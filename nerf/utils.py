import os
import glob
import tqdm
import math
import imageio
import psutil
from pathlib import Path
import random
import shutil
import tensorboardX

import numpy as np
import matplotlib
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchmetrics import PearsonCorrCoef

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

from lib.lpips.LPIPS import LPIPS
from lib.lpips.lpips_metric import rgb_lpips
from kornia.morphology import erosion
from PIL import Image
from sam_utils import get_sam_everything_mask, assign_sam_everything_mask
from derender_utils import gaussian_kernel, image_derender, region_aware_albedo
from PASD import infer_pasd

from diffusers import StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline


def metric_psnr(pred, gt):
    return -10. * np.log10(np.mean(np.square(pred - gt)))

def metric_lpips(pred, gt):
    return rgb_lpips(pred, gt, net_name='vgg', device='cpu')

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Region_Aware_Materials(nn.Module):
    def __init__(self, num_classes, init_roughness=None, init_specular=None, init_albedo_gray=None):
        super(Region_Aware_Materials, self).__init__()
        init_roughness_ = torch.ones(num_classes) * 0.5
        if init_roughness is not None:
            init_roughness_[1:] = init_roughness # the first place is for non-class

        init_specular_ = torch.ones(num_classes) * 0.1
        if init_specular is not None:
            init_specular_[1:] = init_specular

        init_albedo_gray_ = torch.ones(num_classes) * 0.5
        if init_albedo_gray is not None:
            init_albedo_gray_[1:] = init_albedo_gray

        self.roughness = nn.Parameter(init_roughness_.cuda(), requires_grad=True).cuda()
        self.specular = nn.Parameter(init_specular_.cuda(), requires_grad=True).cuda()
        self.albedo_gray = nn.Parameter(init_albedo_gray_.cuda(), requires_grad=True).cuda()


class Trainer(object):
    def __init__(self,
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=False, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        ### Materials
        if self.opt.use_svbrdf and not self.opt.test:
            self.model.prepare_svbrdf(num_lgt_sgs=32)
            image_numpy = self.rgb[0].permute(1,2,0).cpu().numpy()
            spec_estimation, albedo_estimation, spec_strength = image_derender(image_numpy, self.opt.image, self.opt.image[:-9] + '_derender')
            spec_strength = min(1.0, spec_strength*1.0) # XXX
            self.spec_estimation = torch.from_numpy((spec_estimation * spec_strength).astype(np.float32)).permute(2,0,1).unsqueeze(0).to(self.device)
            self.albedo_estimation = torch.from_numpy(albedo_estimation.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(self.device)

        ### SAM
        if self.opt.global_sam and not self.opt.test:
            self.sam_predictor, self.sam_mask_generator, self.sam_mask_generator_lowthresh = self.init_sam()
            self.run_global_SAM(loader=None, default_view=True, derender_dir=self.opt.image[:-9] + '_derender')
            self.model.prepare_class_predictor(num_classes=len(self.sam_labels) + 1, detach=True)
            self.CE_loss = nn.CrossEntropyLoss(reduction='none')

            ## Region-aware materials
            N_regions = len(self.sam_labels) + 1
            # masks = [ann['segmentation'] for ann in self.sam_default_view_masks]
            region_roughness = np.ones(len(self.sam_labels)) * 0.5
            region_speculars = np.ones(len(self.sam_labels)) * 0.23
            self.model.RegionMaterials = Region_Aware_Materials(N_regions, torch.from_numpy(region_roughness),
                                                                torch.from_numpy(region_speculars))
            albedo_region, region_rgb_means = region_aware_albedo(albedo_estimation, self.opt.lambda_region_albedo, self.sam_default_view_masks)
            self.albedo_estimation = torch.from_numpy(albedo_region.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(self.device)
            self.region_rgb_means = torch.from_numpy(region_rgb_means.astype(np.float32)).to(self.device)
        else:
            self.sam_default_view_spec_terms = None

        if self.opt.image is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # Save a copy of images in the experiment workspace
            if opt.images is not None:
                for image_file in opt.images:
                    shutil.copyfile(image_file, os.path.join(self.workspace, os.path.basename(image_file)))

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        # save albedo image
        if hasattr(self, 'albedo_estimation'):
            albedo_est_ = (self.albedo_estimation[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.workspace, 'albedo.png'), cv2.cvtColor(albedo_est_, cv2.COLOR_RGB2BGR))

        self.default_view_data, self.ref_view_mask, self.ref_view_rgb, self.ref_view_depth, self.ref_view_normal = [None] * 5

        # perceptual loss initialization
        self.lpips_loss_func = None
        if self.opt.lambda_super_reso > 0:
            self.lpips_loss_func = LPIPS().to(self.device)

        # smooth gaussian
        ksize = self.opt.h // 16 + 1
        self.gaussian_conv = nn.Conv2d(1, 1, kernel_size=(ksize, ksize), stride=1, padding=ksize//2, bias=False)
        self.gaussian_conv.weight.data = torch.from_numpy(gaussian_kernel(ksize, 0.5))[None,None].float().cuda()
        self.gaussian_conv.weight.requires_grad = False

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        h = int(self.opt.known_view_scale * self.opt.h)
        w = int(self.opt.known_view_scale * self.opt.w)

        # load processed image
        for image in self.opt.images:
            assert image.endswith('_rgba.png') # the rest of this code assumes that the _rgba image has been passed.

        rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in self.opt.images]
        rgba_hw = np.stack([cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
        
        mask_threshold = 0.6 if self.opt.dmtet else 0.9 # hyper-parameter, important!!!
        rgb_mask = (rgba_hw[..., 3:] > mask_threshold).astype(np.float32)
        rgb_hw = rgba_hw[..., :3] * rgb_mask + (1 - rgb_mask)

        self.rgb_numpy = rgb_hw
        self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().to(self.device)
        self.mask = torch.from_numpy(rgba_hw[..., 3] > mask_threshold).to(self.device)
        kernel = torch.ones(3, 3).to(self.device)
        self.mask_erosion = erosion(self.mask[None].float(), kernel).squeeze() > 0.9
        print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

        # load depth
        depth_paths = [image.replace('_rgba.png', '_depth.png') for image in self.opt.images]
        depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
        depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
        self.depth = torch.from_numpy(depth.astype(np.float32) / 255).to(self.device)  # TODO: this should be mapped to FP16
        print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

        # load normal
        normal_paths = [image.replace('_rgba.png', '_normal.npy') for image in self.opt.images]
        normals = [np.load(normal_path) for normal_path in normal_paths]
        normal = np.stack([cv2.resize(n, (w, h), interpolation=cv2.INTER_AREA) for n in normals])
        self.normal = torch.from_numpy(normal * 2. - 1.).to(self.device).squeeze() # fixme: only apply to single ref image
        print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

        # encode embeddings for zero123
        rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])

        normal_256 = np.stack([cv2.resize(n, (256, 256), interpolation=cv2.INTER_AREA) for n in normals])
        normal_256 = normal_256 * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
        normal_256 = torch.from_numpy(normal_256).permute(0,3,1,2).contiguous().to(self.device)
        guidance_embeds_normal = self.guidance['zero123'].get_img_embeds(normal_256)
        self.embeddings['zero123']['normal'] = {
            'zero123_ws' : self.opt.zero123_ws,
            'c_crossattn' : guidance_embeds_normal[0],
            'c_concat' : guidance_embeds_normal[1],
            'ref_polars' : self.opt.ref_polars,
            'ref_azimuths' : self.opt.ref_azimuths,
            'ref_radii' : self.opt.ref_radii,
        }

        rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
        rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().to(self.device)
        guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
        self.embeddings['zero123']['default'] = {
            'zero123_ws' : self.opt.zero123_ws,
            'c_crossattn' : guidance_embeds[0],
            'c_concat' : guidance_embeds[1],
            'ref_polars' : self.opt.ref_polars,
            'ref_azimuths' : self.opt.ref_azimuths,
            'ref_radii' : self.opt.ref_radii,
        }

        # get image caption
        with open(self.opt.images[0].replace('_rgba.png', '_caption.txt'), 'r') as f:
            self.image_caption = f.read()
        print(f'image caption: {self.image_caption}')

        if self.opt.lambda_super_reso > 0:
            if self.opt.sr_pipeline == 'sd_i2i':
                model_sr = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16,
                    variant="fp16", use_safetensors=True).to(self.device)
                model_sr.set_progress_bar_config(disable=True)
                self.model_sr = model_sr
            else:
                self.sr_args = infer_pasd.PASD_args()
                self.model_sr = infer_pasd.load_pasd_pipeline(self.sr_args, self.device, True)

    def init_sam(self):
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        predictor = SamPredictor(sam)
        print('Successfully loaded SAM Predictor')
        mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=500)
        mask_generator_lowthresh = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.86,min_mask_region_area=500)
        print('Successfully loaded SAM Mask Generator')
        return predictor, mask_generator, mask_generator_lowthresh

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def train_step(self, data, save_guidance_path:Path=None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """

        mode = 'geometry' if (self.opt.use_svbrdf and self.global_step <= self.opt.start_sg_render) else 'appearance'
        self.opt.guidance_scale = 25 if mode == 'geometry' else 5

        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.global_step % self.opt.known_view_interval == 0

        # SAM loss
        do_sam_loss = self.opt.global_sam and self.global_step % self.opt.sam_loss_interval == 0 and mode == 'appearance' \
                        and (hasattr(self, 'sam_view_data') or do_rgbd_loss)

        # SR loss
        do_sr_loss = self.opt.lambda_super_reso > 0 and self.epoch >= self.opt.start_super_reso and \
                      not do_rgbd_loss and self.global_step % self.opt.sr_loss_interval == 0 and not do_sam_loss

        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data

        if do_sr_loss:
            sr_idx = random.randint(0, self.opt.lambda_super_reso_views-1)
            data = self.sr_view_data[sr_idx]

        if do_sam_loss and not do_rgbd_loss:
            sam_idx = random.randint(0, len(self.sam_view_data)-1)
            if self.sam_view_label[sam_idx].sum() > 0:
                data = self.sam_view_data[sam_idx]
            else:
                do_sam_loss = False

        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (self.opt.exp_end_iter - self.opt.exp_start_iter)

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0*exp_iter_ratio)
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                    self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                    self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + 2.0*exp_iter_ratio)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]
        pose = data['pose']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if self.opt.use_svbrdf and self.global_step > self.opt.start_sg_render:
            ambient_ratio = 1.0
            shading = 'svbrdf'
            as_latent = False
            binarize = False
            bg_color = torch.ones((B * N, 3), device=rays_o.device)

        elif do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False
            bg_color = torch.rand((B * N, 3), device=rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale #* (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif do_sr_loss:
            ambient_ratio = 1.0
            shading = 'lambertian' # use lambertian instead of albedo to get normal
            as_latent = False
            binarize = False
            bg_color = torch.ones((B * N, 3), device=rays_o.device)

        elif exp_iter_ratio <= self.opt.latent_iter_ratio:
            ambient_ratio = 1.0
            shading = 'normal'
            as_latent = True
            binarize = False
            bg_color = None

        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = self.opt.min_ambient_ratio + (1.0-self.opt.min_ambient_ratio) * random.random()
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and rand > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg

        if mode == 'geometry':
            shading = 'albedo' if do_rgbd_loss else 'normal'
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize,
                                    predict_class=self.opt.global_sam, pose=data['pose'])
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

        # known view loss
        if do_rgbd_loss:
            gt_mask = self.mask # [B, H, W]
            gt_rgb = self.rgb   # [B, 3, H, W]
            gt_normal = self.normal # [B, H, W, 3]
            gt_depth = self.depth   # [B, H, W]

            if len(gt_rgb) > self.opt.batch_size:
                gt_mask = gt_mask[choice]
                gt_rgb = gt_rgb[choice]
                gt_normal = gt_normal[choice]
                gt_depth = gt_depth[choice]

            # color loss
            mask = gt_mask[:, None].expand(-1,3,-1,-1)
            loss = self.opt.lambda_rgb * F.mse_loss(pred_rgb[mask], gt_rgb[mask])

            # mask loss
            loss = loss + self.opt.lambda_mask * F.mse_loss(pred_mask[:, 0], gt_mask.float())

            # albedo and spcular regularization at the reference view
            if self.opt.derender_reg and shading == 'svbrdf':
                pred_spec = outputs['spec_color'].reshape(B, H, W, 3).permute(0,3,1,2)
                pred_albedo = outputs['albedo'].reshape(B, H, W, 3).permute(0,3,1,2)
                spec_loss = self.opt.lambda_derender_reg * F.mse_loss(pred_spec[mask], self.spec_estimation[mask])
                albedo_weight = (1 - self.spec_estimation[mask])
                albedo_loss = self.opt.lambda_derender_reg * (albedo_weight * F.mse_loss(pred_albedo[mask], self.albedo_estimation[mask], reduction='none')).mean()
                loss = loss + spec_loss + albedo_loss

            # normal loss
            if self.opt.lambda_normal > 0 and 'normal_image' in outputs:
                # monosdf implementation
                l1, cos = self.model.get_normal_loss(outputs['normal_image'].reshape(self.normal.shape), gt_normal, pose,
                                                         self.mask_erosion, convert=True, smooth=False)
                est_normal_loss = (l1 + cos)

                loss = loss + self.opt.lambda_normal * est_normal_loss

            # relative depth loss
            if self.opt.lambda_depth > 0:
                valid_gt_depth = gt_depth[gt_mask] # [B,]
                valid_pred_depth = pred_depth[:, 0][gt_mask] # [B,]
                lambda_depth = self.opt.lambda_depth * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * (1 - self.pearson(valid_pred_depth, valid_gt_depth))

        # novel view loss
        else:

            loss = 0

            if not do_sr_loss:

                polar = data['polar']
                azimuth = data['azimuth']
                radius = data['radius']

                embedding = self.embeddings['zero123']['normal'] if mode == 'geometry' and self.opt.guidance_image == 'normal' else self.embeddings['zero123']['default']
                loss = loss + self.guidance['zero123'].train_step(embedding, pred_rgb, polar, azimuth, radius, guidance_scale=self.opt.guidance_scale,
                                                                  as_latent=as_latent, grad_scale=self.opt.lambda_guidance, save_guidance_path=save_guidance_path, global_step=self.global_step)

            else: # Apply super-resolution loss
                if self.sr_view_img[sr_idx] is None:
                    pred_rgb_sr = self.run_sr_module(pred_rgb[0].detach().clamp(0,1).cpu().numpy(), False, f'ep{self.epoch:0>4d}_{sr_idx}')
                    self.sr_view_img[sr_idx] = pred_rgb_sr
                else:
                    pred_rgb_sr = self.sr_view_img[sr_idx]
                    # save pred rgb for validation
                    # pred_rgb_img = (pred_rgb[0].detach().permute(1, 2, 0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                    # cv2.imwrite(os.path.join(self.workspace, 'SR', f'ep{self.epoch:0>4d}_{sr_idx}_{self.global_step}.png'), cv2.cvtColor(pred_rgb_img, cv2.COLOR_RGB2BGR))
                pred_rgb_sr = torch.from_numpy(pred_rgb_sr).unsqueeze(0).to(self.device)
                pred_mask_bi = (pred_mask > 0.5).expand(-1,3,-1,-1)
                sr_loss = self.lpips_loss_func.forward(pred_rgb, pred_rgb_sr)
                sr_loss_pixel = F.mse_loss(pred_rgb[pred_mask_bi], pred_rgb_sr[pred_mask_bi])
                loss += self.opt.lambda_super_reso * (sr_loss + sr_loss_pixel)

        # albedo regularization at the novel views
        if hasattr(self, 'sam_view_label') and self.opt.lambda_albedo > 0 and not do_rgbd_loss:
            assert self.opt.global_sam
            pred_albedo = outputs['albedo'].reshape(B, H, W, 3).permute(3,0,1,2) # 3, 1, H, W
            pred_class_ = outputs['pred_class'].reshape(B, H, W, self.model.num_classes)
            with torch.no_grad():
                prob_mask = nn.Softmax(dim=-1)(pred_class_.squeeze()).max(dim=-1)[0] > 0.5
            labels = pred_class_.argmax(-1)[0]
            albedo_loss = 0
            for i in self.sam_labels:
                label_mask = (labels == i).detach()
                label_mask_gaussian = self.gaussian_conv(label_mask[None, None].float())
                calc_mask = label_mask_gaussian.squeeze() > 0.1
                calc_mask = calc_mask & prob_mask
                if calc_mask.sum() < 100:
                    continue
                pred_albedo_ = torch.zeros_like(pred_albedo)
                pred_albedo_[:,:,label_mask] = pred_albedo[:,:,label_mask]
                spatial_gaussian = (self.gaussian_conv(pred_albedo_) / (label_mask_gaussian + 1e-6)).permute(1, 2, 3, 0)[0]
                albedo_loss += ((spatial_gaussian[calc_mask] - self.region_rgb_means[i-1])**2).mean()
            loss += self.opt.lambda_albedo * albedo_loss

        # Apply loss for semantic label learning
        if do_sam_loss:
            if do_rgbd_loss:
                gt_img_label = self.sam_default_view_label
            else:
                gt_img_label = self.sam_view_label[sam_idx]

            gt_label = F.one_hot(gt_img_label.long(), num_classes=self.model.num_classes)
            ce_loss = self.CE_loss(outputs['pred_class'].reshape(B, H, W, self.model.num_classes).permute(0,3,1,2), gt_label.permute(2,0,1)[None].float())
            ce_loss = ce_loss[0,gt_img_label>0].mean()
            if do_rgbd_loss:
                ce_loss *= 100
            loss += self.opt.lambda_sam_loss * ce_loss

        # regularizations
        if not self.opt.dmtet:

            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:
                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
                lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)
                loss = loss + lambda_entropy * loss_entropy

            if self.opt.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
                # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
                # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
                # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
                # total-variation
                loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                              (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
                loss = loss + self.opt.lambda_2d_normal_smooth * loss_smooth

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient

            if self.opt.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
                loss_normal_perturb = outputs['loss_normal_perturb']
                loss = loss + self.opt.lambda_3d_normal_smooth * loss_normal_perturb

        else:

            if self.opt.lambda_mesh_normal > 0:
                loss = loss + self.opt.lambda_mesh_normal * outputs['normal_loss']

            if self.opt.lambda_mesh_laplacian > 0:
                loss = loss + self.opt.lambda_mesh_laplacian * outputs['lap_loss']

        return pred_rgb, pred_depth, loss

    def _vis_segment(self, pred_class, file_name='seg.png'):
        semantic_colors = torch.from_numpy(np.asarray(matplotlib.colormaps[self.opt.colormap].colors)).float().cuda()
        semantic_rgb = semantic_colors[pred_class.long()]
        semantic_rgb = (semantic_rgb.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(file_name, cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR))

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data, vis=True):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.opt.use_svbrdf and self.global_step > self.opt.start_sg_render:
            shading = 'svbrdf'
        elif self.opt.dmtet:
            shading = 'albedo'
        else:
            shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio,
                                    shading=shading, predict_class=self.opt.global_sam, pose=data['pose'])
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        if shading == 'svbrdf' and vis:
            outputs_albedo = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio,
                                        shading='albedo', predict_class=self.opt.global_sam, pose=data['pose'])
            pred_albedo = outputs_albedo['image'].reshape(B, H, W, 3)
            outputs_albedo = outputs['albedo']
            pred_albedo = outputs_albedo.reshape(B, H, W, 3)
            pred_rgb = torch.concat([pred_rgb, pred_albedo],1)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
            pred_alpha = outputs['weights_sum'].reshape(B, H, W, 1)
            pred_normal = pred_normal * pred_alpha
        else:
            pred_normal = None

        if 'pred_class' in outputs:
            pred_class = outputs['pred_class'].reshape(B, H, W, len(self.sam_labels)+1).argmax(-1)
            semantic_colors = torch.from_numpy(np.asarray(matplotlib.colormaps[self.opt.colormap].colors)).float().cuda()
            pred_others = semantic_colors[pred_class.long()]
            # specular = self.model.RegionMaterials.specular[pred_class]
            # pred_others = torch.cat([semantic_rgb, specular[...,None].repeat(1,1,1,3)], 1)
            if 'spec_color' in outputs:
                specular = outputs['spec_color'].reshape(B, H, W, 3)
                pred_others = torch.cat([pred_others, specular], 1)
        else:
            pred_others = None

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, pred_normal, pred_others, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        if self.opt.use_svbrdf and self.global_step > self.opt.start_sg_render:
            shading = 'svbrdf'
        elif self.opt.dmtet:
            shading = 'albedo'
        else:
            shading = data['shading'] if 'shading' in data else 'albedo'

        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color, predict_class=self.opt.global_sam, pose=data['pose'])

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        if 'normal_image' in outputs:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)
            pred_alpha = outputs['weights_sum'].reshape(B, H, W, 1)
            pred_normal = pred_normal * pred_alpha
        else:
            pred_normal = None

        if 'pred_class' in outputs:
            pred_class = outputs['pred_class'].reshape(B, H, W, len(self.sam_labels)+1).argmax(-1)
            semantic_colors = torch.from_numpy(np.asarray(matplotlib.colormaps[self.opt.colormap].colors)).float().cuda()
            pred_others = semantic_colors[pred_class.long()]
            if 'spec_color' in outputs:
                specular = outputs['spec_color'].reshape(B, H, W, 3)
                pred_others = torch.cat([pred_others, specular], 2)
            if 'albedo' in outputs:
                pred_albedo = outputs['albedo'].reshape(B, H, W, 3)
                pred_others = torch.cat([pred_others, pred_albedo], 2)
        else:
            pred_others = None

        return pred_rgb, pred_depth, pred_normal, pred_others

    def save_mesh(self, loader=None, save_path=None, suffix=''):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh' + suffix)

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        # self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)
        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution,
                               decimate_target=self.opt.decimate_target,
                               speculars=self.sam_default_view_spec_terms,
                               semantic=self.opt.global_sam)

        self.log(f"==> Finished saving mesh.")

    def save_envmap(self, name=None):
        if name == None:
            savepath = os.path.join(self.workspace, 'envmap.exr')
        else:
            savepath = name
        self.model.export_envmap(savepath)

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            if self.opt.global_sam and self.epoch in self.opt.sam_update_epoch:
                self.run_global_SAM(loader=train_loader, derender_dir=self.opt.image[:-9] + '_derender')

            if self.opt.lambda_super_reso > 0 and self.epoch >= self.opt.start_super_reso and self.epoch % 5 == 0:
                self.sr_view_data = train_loader._data.get_sr_view_data(self.opt.lambda_super_reso_views)
                self.sr_view_img = [None for i in range(self.opt.lambda_super_reso_views)]

            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []
            all_preds_other = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, preds_other = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if preds_normal is not None:
                    rot = data['pose'][0, :3, :3].permute(1, 0).cpu().numpy()
                    rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ rot
                    pred_normal = (rot @ preds_normal[0][..., None].cpu().numpy())[..., 0]
                    weight_mask = ((pred_normal > 1e-5) | (pred_normal < -1e-5)).astype(pred_normal.dtype)
                    pred_normal = to8b((0.5 * pred_normal + 0.5)*weight_mask + (1-weight_mask))
                else:
                    pred_normal = None
                
                if preds_other is not None:
                    pred_other = (preds_other[0].detach().cpu().numpy() * 255).astype(np.uint8)
                else:
                    pred_other = None
                
                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_normal.append(pred_normal)
                    all_preds_other.append(pred_other)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    if pred_normal is not None:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'), pred_normal)
                    if pred_other is not None:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_other.png'), pred_other)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            if all_preds_normal[0] is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8, macro_block_size=1)
            if all_preds_other[0] is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_other.mp4'), all_preds_other, fps=25, quality=8, macro_block_size=1)

        if hasattr(self.model, 'lgtSGs'):
            self.save_envmap(os.path.join(save_path, f'{name}_envmap.exr'))
        self.log(f"==> Finished Test.")

    def relight(self, loader, save_path=None, name=None, write_video=True):

        def rotate_envmap(lgtSGs, angle):
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler('yxz', [angle, 0, 0], degrees=True)
            try:
                rotation = r.as_matrix()
            except:
                rotation = r.as_dcm()
            lgtSGLobes = lgtSGs[:, :3] / (np.linalg.norm(lgtSGs[:, :3], axis=-1, keepdims=True) + 1e-8)
            lgtSGLambdas = np.abs(lgtSGs[:, 3:4])
            lgtSGMus = np.abs(lgtSGs[:, 4:])
            lgtSGLobes_rot = np.matmul(lgtSGLobes, rotation.T)
            lgtSGs_rot = np.concatenate((lgtSGLobes_rot, lgtSGLambdas, lgtSGMus), axis=-1).astype(np.float32)
            return lgtSGs_rot

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []

        env_map = np.load(self.opt.relight_sg)
        with torch.no_grad():

            nums = len(loader)
            data = next(iter(loader))
            for i in range(nums):

                env_map_rot = rotate_envmap(env_map, 360/nums*i)
                self.load_envmap(env_map_rot)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if preds_normal is not None:
                    rot = data['pose'][0, :3, :3].permute(1, 0).cpu().numpy()
                    rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ rot
                    pred_normal = (rot @ preds_normal[0][..., None].cpu().numpy())[..., 0]
                    weight_mask = ((pred_normal > 1e-5) | (pred_normal < -1e-5)).astype(pred_normal.dtype)
                    pred_normal = to8b((0.5 * pred_normal + 0.5)*weight_mask + (1-weight_mask))
                else:
                    pred_normal = None

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_normal.append(pred_normal)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    if pred_normal is not None:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'), pred_normal)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            if all_preds_normal[0] is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8, macro_block_size=1)

        self.save_envmap(os.path.join(save_path, f'{name}_envmap.exr'))
        self.log(f"==> Finished Test.")

    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs


    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, mvp, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        mvp = torch.from_numpy(mvp).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:

            # update grid every 16 steps
            if (self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                    save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, loss = self.train_step(data, save_guidance_path=save_guidance_path)

            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)
                pred_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()

            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, preds_other, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_normal = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_normal.png')
                    save_path_others = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_other.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    if preds_normal is not None:
                        rot = data['pose'][0, :3, :3].permute(1, 0).cpu().numpy()
                        rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ rot # fixme!!!!!
                        pred_normal = (rot @ preds_normal[0][..., None].cpu().numpy())[..., 0]
                        cv2.imwrite(save_path_normal, to8b(0.5 * pred_normal + 0.5))

                    if preds_other is not None:
                        pred_other = (preds_other[0].detach().cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(save_path_others, cv2.cvtColor(pred_other, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    @torch.cuda.amp.autocast(enabled=False)
    def run_sr_module(self, img_lr, save_image=False, out_name=None):
        '''
        :param img_lr ndarray [3, H, W] [0, 1]
        return ndarray [3, H, W]
        '''
        _, h, w = img_lr.shape
        # turn ndarray into PIL Image
        img_lr = Image.fromarray((img_lr.transpose(1, 2, 0)*255).astype(np.uint8)).convert('RGB')
        # run super-resolution 
        with torch.no_grad():
            if self.opt.sr_pipeline == 'sd_i2i':
                img_sr = self.model_sr(prompt=self.image_caption, image=img_lr.resize((768,768)), strength=self.opt.sr_strength).images[0]
            else:
                validation_prompt = self.image_caption + self.sr_args.added_prompt
                img_sr = self.model_sr(self.sr_args, validation_prompt, img_lr.resize((768,768)),
                                num_inference_steps=self.sr_args.num_inference_steps, guidance_scale=self.sr_args.guidance_scale,
                                negative_prompt=self.sr_args.negative_prompt, conditioning_scale=self.sr_args.conditioning_scale).images[0]
                img_sr = infer_pasd.wavelet_color_fix(img_sr, img_lr)
        img_sr = img_sr.resize((h, w))
        if save_image:
            os.makedirs(os.path.join(self.workspace, 'SR'), exist_ok=True)
            if out_name is None:
                sr_path = os.path.join(self.workspace, 'SR', f'{self.global_step}_sr.png')
                lr_path = os.path.join(self.workspace, 'SR', f'{self.global_step}_lr.png')
            else:
                sr_path = os.path.join(self.workspace, 'SR', f'{out_name}_sr.png')
                lr_path = os.path.join(self.workspace, 'SR', f'{out_name}_lr.png')
            img_sr.save(sr_path)
            img_lr.save(lr_path)
        img_sr = np.array(img_sr).transpose(2,0,1).astype(np.float32) / 255.

        return img_sr

    def run_global_SAM(self, loader=None, default_view=False, derender_dir=''):
        os.makedirs(os.path.join(self.workspace, 'SAM'), exist_ok=True)

        # run SAM on the default view, segment everything in the image.
        if default_view:
            feat_guidance = 'hsv' if self.opt.use_hsv else None
            img_numpy = self.rgb[0].permute(1,2,0).detach().cpu().numpy()
            img_mask = self.mask[0].detach().cpu().numpy()
            self.sam_labels, self.sam_default_view_masks, self.sam_default_view_colors, self.sam_default_view_label, self.sam_default_view_spec_terms = \
                get_sam_everything_mask(img_numpy, img_mask, self.sam_mask_generator, self.sam_predictor, self.device, feat_guidance=feat_guidance, min_color_similarity=self.opt.sam_color_threshold,
                                        name=os.path.join(self.workspace, 'SAM/default_view'), derender_dir=derender_dir, colormap=self.opt.colormap,
                                        material_masks=self.opt.material_masks)

        # run SAM on other views for training
        if loader is not None:
            print('>>> Start to update SAM labels for training.')
            rand_idx = np.random.randint(0, len(loader), self.opt.sam_n_views).tolist()
            self.sam_view_data = []
            self.sam_view_label = []
            with torch.no_grad():
                for i, data in tqdm.tqdm(enumerate(loader)):
                    if i not in rand_idx:
                        continue
                    bs = data['rays_o'].shape[0]
                    idx = np.random.randint(0, bs)
                    for k in data.keys():
                        if k != 'H' and k!= 'W':
                            data[k] = data[k][idx:idx+1]
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, preds_normal, _, loss = self.eval_step(data, vis=True)
                        hw = min(preds.shape[1], preds.shape[2])
                        img_numpy = preds[0,-hw:].cpu().detach().numpy() # get albedo part if exists
                        img_mask = preds_depth[0].cpu().detach().numpy() > 0.5
                        feat_guidance = 'hsv' if self.opt.use_hsv else None
                        image_label = assign_sam_everything_mask(img_numpy, img_mask, self.sam_mask_generator_lowthresh,self.device, self.sam_default_view_colors,
                                                                 min_color_similarity=self.opt.sam_color_threshold * 2, feat_guidance=feat_guidance,
                                                                 name=os.path.join(self.workspace, 'SAM/ep-{}_view-{}'.format(self.epoch, i)),
                                                                 colormap=self.opt.colormap)
                        self.sam_view_data.append(data)
                        self.sam_view_label.append(image_label)

            print('>>> Finished update SAM labels on {} views for training.'.format(self.opt.sam_n_views))

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        # init missing modules
        if 'class_net.net.2.bias' in checkpoint_dict['model']:
            self.model.prepare_svbrdf(num_lgt_sgs=32)
            N_regions = checkpoint_dict['model']['class_net.net.2.bias'].shape[0]
            self.sam_labels = list(range(1, N_regions))
            self.model.prepare_class_predictor(num_classes=N_regions, detach=True)
            region_roughness = np.ones(N_regions-1) * 0.5
            region_speculars = np.ones(N_regions-1) * 0.23
            self.model.RegionMaterials = Region_Aware_Materials(N_regions, torch.from_numpy(region_roughness),
                                                                torch.from_numpy(region_speculars))
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def load_envmap(self, envmap):
        self.model.load_envmap(envmap)

def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems
