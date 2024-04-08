import glob
import math
import os
import random
import zipfile
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn


def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available() and (cuda_device_id is not None or args.gpu_any) else 'cpu'

    print(f"Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}")
    return cfgs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def archive_code(arc_path, filetypes=['.py', '.yml']):
    print(f"Archiving code to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '**', '*'+ftype), recursive=True))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def draw_bbox(im, size):
    b, c, h, w = im.shape
    h2, w2 = (h-size)//2, (w-size)//2
    marker = np.tile(np.array([[1.],[0.],[0.]]), (1,size))
    marker = torch.FloatTensor(marker)
    im[:, :, h2, w2:w2+size] = marker
    im[:, :, h2+size, w2:w2+size] = marker
    im[:, :, h2:h2+size, w2] = marker
    im[:, :, h2:h2+size, w2+size] = marker
    return im


def save_videos(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.mp4', cycle=False):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,1,3,4,2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[...,::-1]*255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.png', offset=None):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    if offset is None:
        offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    imgs = imgs.transpose(0,2,3,1)
    for i, img in enumerate(imgs):
        if 'depth' in suffix:
            im_out = np.uint16(img[...,::-1]*65535.)
        else:
            im_out = np.uint8(img[...,::-1]*255.)
        cv2.imwrite(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), im_out)


def save_array(file, array):
    array = array.transpose(1, 2, 0)
    array_out = np.uint16(array[...,::-1]*65535.)
    cv2.imwrite(str(file), array_out)


def save_image(file, img):
    img = img.transpose(1, 2, 0)
    im_out = np.uint8(img[...,::-1]*255.)
    cv2.imwrite(str(file), im_out)


def save_npy(file, array):
    np.save(str(file), array)


def load_array(file):
    array = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return np.array(array, dtype=np.float32) / 65535.


def load_image(file):
    # array = Image.open(file)
    # return np.array(array, dtype=np.float32) / 255.
    array = cv2.cvtColor(cv2.imread(str(file), -1), cv2.COLOR_BGR2RGB)
    max_v = 65535. if array.dtype == np.uint16 else 255.
    array = np.array(array, dtype=np.float32) / max_v
    return array


def load_npy(file):
    return np.load(file)


def save_txt(out_fold, data, prefix='', suffix='', sep_folder=True, ext='.txt'):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) +1

    [np.savetxt(os.path.join(out_fold, prefix+'%05d'%(i+offset)+suffix+ext), d, fmt='%.6f', delimiter=', ') for i,d in enumerate(data)]


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b,1,1))**2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b,1,1))**2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask if mask is not None else dist


def mask_mean(t: torch.Tensor, m: torch.Tensor, dim=None, keepdim=False):
    t = t.clone()
    t[m] = 0
    els = 1
    if dim is None or len(dim)==0:
        dim = list(range(len(t.shape)))
    for d in dim:
        els *= t.shape[d]
    return torch.sum(t, dim=dim, keepdim=keepdim) / (els - torch.sum(m.to(torch.float), dim=dim, keepdim=keepdim))


def positional_encoding(x, L=4):
    if len(x.shape) == 2:
        b, c = x.shape
        powers = 2 ** torch.arange(L).to(device=x.device, dtype=torch.float32).view(1, 1, -1) * math.pi
        powers = x.unsqueeze(2) * powers
        return torch.cat([torch.sin(powers), torch.cos(powers)], dim=2).reshape(b, -1)
    else:
        b, c, h, w = x.shape
        powers = 2 ** torch.arange(L).to(device=x.device, dtype=torch.float32).view(1, 1, -1, 1, 1) * math.pi
        powers = x.unsqueeze(2) * powers
        return torch.cat([torch.sin(powers), torch.cos(powers)], dim=2).reshape(b, -1, h, w)


def _resize_square(image, size):
    if len(image.shape) == 3:
        return F.interpolate(image.unsqueeze(1),(size, size), mode='bilinear').squeeze(1)
    else:
        return F.interpolate(image, (size, size), mode='bilinear')


def resize_square(image, size):
    if not isinstance(image, list):
        return _resize_square(image, size)
    else:
        return [_resize_square(img, size // (2 ** i)) for i, img in enumerate(image)]


def _resize_normals_square(normals, size):
    normals = F.interpolate(normals.permute(0, 3, 1, 2), size=(size, size), mode='bilinear').permute(0, 2, 3, 1)
    v_lengths = normals.pow(2).sum(-1, keepdims=True).sqrt()
    return normals / v_lengths


def resize_normals_square(normals, size):
    if not isinstance(normals, list):
        return _resize_normals_square(normals, size)
    else:
        return [_resize_normals_square(ns, size // (2 ** i)) for i, ns in enumerate(normals)]


unsqueezer = partial(torch.unsqueeze, dim=0)


def map_fn(batch, fn):
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = map_fn(batch[k], fn)
        return batch
    elif isinstance(batch, list):
        return [map_fn(e, fn) for e in batch]
    else:
        return fn(batch)


def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)


def plot_image_grid(images, rows, cols, directions=None, imsize=(2, 2), title=None, show=True):
    fig, axs = plt.subplots(rows, cols, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(rows * imsize[0], cols * imsize[1]))
    for i, image in enumerate(images):
        axs[i % rows][i // rows].axis("off")
        if directions is not None:
            axs[i % rows][i // rows].arrow(32, 32, directions[i][0] * 16, directions[i][1] * 16, color='red', length_includes_head=True, head_width=2., head_length=1.)
        axs[i % rows][i // rows].imshow(image, aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    return fig


def show_save(save_path, show=True, save=False):
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class UnlistLayer(nn.Module):
    def forward(self, x):
        return x[0]


class ListLayer(nn.Module):
    def forward(self, x):
        return [x]


# Taken from https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, mask=None):
        if mask is None:
            mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        else:
            mse = mask_mean((img1 - img2) ** 2, mask, dim=[1, 2, 3])
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def shrink_mask(mask, shrink=3):
    mask = F.avg_pool2d(mask.to(torch.float32), kernel_size=shrink, padding=shrink // 2, stride=1)
    return (mask == 1.).to(torch.float32)


def get_mask(size, border=5, device=None):
    mask = torch.ones(size, dtype=torch.float32)
    mask = shrink_mask(mask, border)
    if device is not None:
        mask = mask.to(device)
    return mask


def get_patch_heatmap(img, disc, patch_size, stride=16):
    b, c, h, w = img.shape
    steps = (h - patch_size) // stride
    heatmap = img.new_zeros((b, 1, steps, steps))
    for x in range(steps):
        for y in range(steps):
            heatmap[:, :, y, x] = disc(img[:, :, y*stride:y*stride+patch_size, x*stride:x*stride+patch_size])
    return heatmap


def detach(t):
    if isinstance(t, tuple):
        return tuple(t_.detach() for t_ in t)
    else: return t.detach()


def xy_to_xyz_norm(xy):
    xyz = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=-1)
    return xyz / torch.norm(xyz, dim=-1, keepdim=True)


def get_grid(H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2).float() # flip h,w to x,y
    return grid


def get_patches(im, num_patch=None, patch_size=128, scale=(0.25,0.5)):
    b, c, h, w = im.shape
    if num_patch is None:
        num_patch = 1
        squeeze = True
    else:
        squeeze = False
    wh = torch.rand(b*num_patch, 2) *(scale[1]-scale[0]) + scale[0]
    xy0 = torch.rand(b*num_patch, 2) *(1-wh) *2 -1  # -1~1-wh
    xy_grid = get_grid(patch_size, patch_size, normalize=True).repeat(b*num_patch,1,1,1)  # -1~1
    xy_grid = xy0.view(b*num_patch,1,1,2) + (xy_grid+1) *wh.view(b*num_patch,1,1,2)
    patches = torch.nn.functional.grid_sample(im.repeat(num_patch,1,1,1), xy_grid.to(im.device), mode='bilinear').view(num_patch,b,c,patch_size,patch_size).transpose(1,0)
    if squeeze:
        patches = patches.squeeze(1)
    return patches  # BxNxCxHxW


def blur(im, kernel, padding=3):
    if im.device != kernel.device:
        kernel = kernel.to(im.device)
    return torch.cat([F.conv2d(im[:, :1], kernel, padding=padding),
                              F.conv2d(im[:, 1:2], kernel, padding=padding),
                              F.conv2d(im[:, 2:], kernel, padding=padding)], dim=1)


def get_ball(direction, resolution=256, specularity=-1, background=1):
    x = torch.linspace(-1, 1, resolution, device=direction.device)
    y = torch.linspace(-1, 1, resolution, device=direction.device)
    xx, yy = torch.meshgrid(y, x)
    xy = torch.stack([-yy, -xx], dim=0)
    mask = torch.norm(xy, dim=0, keepdim=True) > 1.
    xy[mask.expand(2, -1, -1)] = 0
    z = 1 - torch.sum(xy**2, dim=0, keepdim=True)
    xyz = torch.cat([xy, z], dim=0)
    shading = (direction.view(3, 1, 1) * xyz).sum(0).clamp(min=0)

    if specularity >= 0:
        cos_theta = (direction.view(3, 1, 1) * xyz).sum(0)
        reflect_d = (2 * cos_theta * xyz - direction.view(3, 1, 1))
        specular = reflect_d[-1, :, :].clamp(min=0) * (cos_theta > 0).to(torch.float32)
        specular_shading = specular.pow(specularity)
        shading += specular_shading
        shading = shading.pow(1 / 2.2)

    shading[mask[0]] = background
    return shading