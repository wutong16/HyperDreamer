
import os 
import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import random
import torchvision
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Union, List
from diffusers import AutoencoderKL

from pytorch_lightning import seed_everything
from editing.src.cldm.model import create_model, load_state_dict
from editing.src.cldm.ddim_hacked import DDIMSampler
from editing.src import utils
from editing.src.sam3d_utils import SAM3D_Mesh
from editing.src.configs.train_config import TrainConfig
from editing.src.models.textured_mesh import TexturedMeshModel
from editing.src.training.views_dataset import ViewsDataset, MultiviewDataset
from editing.src.utils import make_path, tensor2numpy
from editing.src.utils import get_view_direction



class Tex3D:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.edit_step = 0
        self.sample_size = 1024
        self.guess_mode = False
        self.total_labels = []
        self.total_colors = []
        self.make_savedirs()
        self.text = self.cfg.guide.text
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.view_dirs = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
        self.a_prompt = ', best quality, extremely detailed, hightly detail, high resolution'
        self.n_prompt = ', longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        if self.cfg.optim.seed==-1:
            self.cfg.optim.seed = random.randint(1, 100000)

        seed_everything(self.cfg.optim.seed)
        
        self.init_logger() 
        self.init_cldm()
        # self.vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="vae").to(self.device)
        # from src.ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
        self.vae = self.cldm_model_guide.first_stage_model
        self.sam3d = self.init_sam()
        self.dataloaders = self.init_dataloaders()
        self.mesh_model = self.init_mesh_model()
        if self.cfg.guide.edit_model == "Depth_Aware":
            self.depth_diffusion = self.init_diffusion()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))
        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
    
    def make_savedirs(self):
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.edit_path = make_path(self.cfg.log.exp_dir / 'edit' /'vis')
        self.edit_eval_path = make_path(self.cfg.log.exp_dir / 'edit' /'val')
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.texture_img_path = make_path(self.exp_path / 'texture_img')
        self.final_edit_renders_path = make_path(self.exp_path /'edit'/'results')
    
    def init_cldm(self):
        ##ControlNetv11-normal
        model_name_guide = 'control_v11p_sd15_normalbae'
        self.cldm_model_guide = create_model(f'pretrained/controlnet/{model_name_guide}.yaml').cpu()
        self.cldm_model_guide.load_state_dict(load_state_dict('pretrained/controlnet/v1-5-pruned.ckpt', location=self.device), strict=False)
        self.cldm_model_guide.load_state_dict(load_state_dict(f'pretrained/controlnet/{model_name_guide}.pth', location=self.device), strict=False)
        self.cldm_model_guide = self.cldm_model_guide.to(self.device)
        self.ddim_sampler_guide = DDIMSampler(self.cldm_model_guide)
        
        logger.info('========= Successfully load Cldm Multi Models =========')
        
    def init_sam(self):
        sam3d = SAM3D_Mesh(self.cfg)
        return sam3d

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  initial_texture_path = self.cfg.guide.initial_texture,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model
    
    def init_diffusion(self) -> Any:
        from src.stable_diffusion_depth import StableDiffusion
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device, alternate_views=self.cfg.render.alternate_views).dataloader()

        segment_dataloader = MultiviewDataset(self.cfg.render, device=self.device, alternate_views=False).dataloader()
        val_loader = ViewsDataset(self.cfg.render, device=self.device, size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device, size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader, 'segment':segment_dataloader,
                       'val_large': val_large_loader}
        
        utils.visualize_viewpoints(self.exp_path, dataloaders['train'], 'train')
        utils.visualize_viewpoints(self.exp_path, dataloaders['val'], 'val')
        utils.visualize_viewpoints(self.exp_path, dataloaders['val_large'], 'val_large')
        
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)


    def apply_normal(self, img, mask ,H, W):
        ############ normal bae ############
        normal = ((img + 1) * 0.5).clip(0, 1)  #[1,3,w,w]
        normal = normal.permute(0,2,3,1).squeeze(0)
        normal = normal.detach().cpu().numpy() #[w,w,3]
        normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        normal = torch.from_numpy(normal_image.copy()).float().cuda() / 255.0
        normal = normal.unsqueeze(0).permute(0,3,1,2).to(self.device)
        normal = normal * mask        #[1,3,w,w]

        normal = F.interpolate(normal, (H, W), mode='bilinear', align_corners=False) 
        return normal
    
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents)
            # imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs)
        # posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents      
    
    def eval_render(self, data, use_texture_mask=False, use_texture_original=False):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim),use_texture_mask=use_texture_mask)
        
        mask = outputs['mask']
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        #z_normals = outputs['normal_est'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        """
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)        #[1,1,1024,1024]
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0, 0, 0], z_normals=z_normals,
                                                                                    light_coef=0.3) * uncolored_mask
        """
        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache']
                                                     )
        if use_texture_mask:
            outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                dims=(dim, dim), 
                                                background=torch.Tensor([0.8, 0.8, 0.8]).to(self.device),
                                                use_texture_mask=True,
                                                render_cache=outputs['render_cache']
                                                )
            meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                        background=torch.Tensor([0.8, 0.8, 0.8]).to(self.device),
                                        use_texture_mask=True, render_cache=outputs['render_cache'])
            
        elif use_texture_original:
            outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                dims=(dim, dim), 
                                                background=torch.Tensor([0.8, 0.8, 0.8]).to(self.device),
                                                use_texture_original=True,
                                                render_cache=outputs['render_cache']
                                                )
            meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                        background=torch.Tensor([0, 0, 0]).to(self.device),
                                        use_texture_original=True, render_cache=outputs['render_cache'])
        
        else:
            outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                dims=(dim, dim), use_median=True,
                                                render_cache=outputs['render_cache']
                                                )
            meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                background=torch.Tensor([0, 0, 0]).to(self.device),
                                                use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()
        return rgb_render, texture_rgb, depth_render, pred_z_normals  #preds, textures, depths, normals
    
    
    def paint(self):

        #### Segmentation on 3dmesh and text-based local editing 
        
        self.mesh_model.train()
        self.mesh_model.original_texture_img = self.mesh_model.texture_img.clone().detach()
        self.mesh_model.meta_texture_img = nn.Parameter(torch.zeros_like(self.mesh_model.texture_img))
        
        
        input_point = np.array(eval(self.cfg.guide.input_point))
        input_label = np.array(eval(self.cfg.guide.input_label))
                    
        self.mesh_model.train()
        pbar = tqdm(total=len(self.dataloaders['segment']), initial=self.edit_step,
                bar_format='{desc}: {percentage:3.0f}% Editing step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        logger.info('-------------- NOW Generate the 3D Segmentation Regions --------------')
        for data in self.dataloaders['segment']:
            self.edit_step += 1
            pbar.update(1)
            
            if self.edit_step == 1:
                pred_mask_edit, original_rgb_render = self.generate_3D_SAM(data, points=input_point,labels=input_label)
            else:
                pred_mask_edit, original_rgb_render = self.generate_3D_SAM(data)



            self.sam3d.save_sam_img(pred_mask_edit, original_rgb_render, path=self.edit_path, now_step=self.edit_step)
            

        logger.info('------------------ Generate the 3D Segmentation Regions has been saved ------------------')
        self.evaluate(self.dataloaders['val_large'], self.final_edit_renders_path, save_as_video=True,use_texture_mask=True)
        self.evaluate(self.dataloaders['val_large'], self.final_edit_renders_path, save_as_video=True,use_texture_original=True)

        
        self.edit_step = 0
        self.mesh_model.train()
        self.mesh_model.preparing_editing_texture()
        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.edit_step,
                bar_format='{desc}: {percentage:3.0f}% Editing step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        logger.info('-------------- NOW Editing the Initial 3D Object --------------')
        logger.info(f'-------------- Editting Model: {self.cfg.guide.edit_model}--------------')
        
        for data in self.dataloaders['train']:
            self.edit_step += 1

            pbar.update(1)
            pred_rgb_edit = self.edit_viewpoint(data)
        self.full_eval(self.final_edit_renders_path)
        logger.info('Finished Editing ^_^')
        logger.info('------------------ Inpainted 3D Object has been saved ------------------')





    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False, use_texture_mask=False, use_texture_original=False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
            
        pbar = tqdm(total=len(dataloader), initial=1,
                    bar_format='{desc}: {percentage:3.0f}% Exporting Inpainted 3D Object Results step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')    
        
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data, use_texture_mask, use_texture_original)

            pred = tensor2numpy(preds[0])
            pbar.update(1)
            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
 

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"result_{name}.mp4", video,
                                                            fps=25,
                                                            quality=8, macro_block_size=1)
            if use_texture_mask:
                dump_vid(all_preds, 'mask')
            elif use_texture_original:
                dump_vid(all_preds, 'original')
            else:
                dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(output_dir / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset) #做了视线的offset
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {round(np.rad2deg(theta))}, phi: {round(np.rad2deg(phi))}, radius: {radius}')

        background = torch.Tensor([0.8, 0.8, 0.8]).to(self.device) 
        


        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background) #mesh_model:TexturedMeshModel()   重点需要改的地方，改变输出
        
        render_cache = outputs['render_cache']  
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']    # [1,1,w,w]


        ## Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render_median = outputs['image'] # Render again with the median value to use as rgb
        rgb_render=rgb_render_median
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)
       
        
        normal = outputs['normals'].clamp(0, 1)
        normal_raw = outputs['normals']#[1,3,w,w]
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1) #[1,1,w,w]
        z_normals_cache = meta_output['image'].clamp(0, 1)       #[1,3,w,w]
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2] #[1,1,w,w]
        mask=outputs['mask']


        self.log_train_image(rgb_render_raw, 'rendered_input')      #[1,3,w,w]   value[0-1]
        self.log_train_image(depth_render[0,0], 'depth',colormap=True) #[1,1,w,w] -> [w,w] value[0-1]
        generate_mask, refine_mask, junction_mask, update_mask, shaded_input = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=mask,
                                                                        mask=outputs['mask'])
        
        self.log_train_image(generate_mask.repeat(1,3,1,1), name='generate_mask')
        self.log_train_image(refine_mask.repeat(1,3,1,1), name='refine_mask')
        self.log_train_image(junction_mask.repeat(1,3,1,1), name='junction_mask')
        self.log_train_image(update_mask.repeat(1,3,1,1), name='update_mask')
        
        H = W = self.sample_size  #H=W=1024

        shape = (4, H//8, W//8)
        ddim_steps = 20
        strength = self.cfg.guide.strength
        scale = self.cfg.guide.guidance_scale
        
        
        dirs = data['dir']
        print("dirs:",dirs)
        if self.cfg.guide.append_direction:
            # text_prompt = "the {} view of " + self.text
            text_prompt = self.text + ", {} view"
            text_prompt = text_prompt.format(self.view_dirs[dirs])
        else:
            text_prompt = self.text
        logger.info(f'Text: {text_prompt}')
        #######
        
        control = self.apply_normal(depth_render, mask , H, W)
        self.log_train_image(control, name='normal')
        
        cond = {"c_concat": [control], "c_crossattn": [self.cldm_model_guide.get_learned_conditioning([text_prompt + ', ' + self.a_prompt])]}  #normal [1,3,h,w]   h=w=1024=train_grid_size
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.cldm_model_guide.get_learned_conditioning([self.n_prompt])]}
        rgb_render_1024 = F.interpolate(rgb_render, (H, W), mode='bilinear', align_corners=False)     #[1,3,H//8,W//8]
        x0 = self.encode_imgs(rgb_render_1024)        
        # x0 = self.cldm_model_guide.get_first_stage_encoding(self.cldm_model_guide.encode_first_stage(rgb_render_1024))  
        self.cldm_model_guide.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        
        update_mask_cldm = F.interpolate(update_mask, (H//8, W//8), mode='bilinear', align_corners=False)   #[1,1,H//8,W//8]  
        update_mask_cldm = update_mask_cldm.permute(0,2,3,1).squeeze(0).cpu().numpy()
        update_mask_cldm = torch.from_numpy(update_mask_cldm[:,:,::-1].copy())
        update_mask_cldm = update_mask_cldm.unsqueeze(0).permute(0,3,1,2).to(self.device)
        update_mask_cldm = F.interpolate(update_mask_cldm, (H//8, W//8), mode='bilinear', align_corners=False)     #[1,3,H//8,W//8]
        noise_same = torch.randn_like(x0)
            
       

        samples, intermediates = self.ddim_sampler_guide.sample(ddim_steps, 1,
                                                            shape, cond, verbose=False, eta=0, mask=update_mask_cldm, x0=x0,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond,
                                                            noise=noise_same)
        
        samples_output = self.decode_latents(samples) #[1,3,h,w]=[1,3,1024,1024] ####

        self.log_train_image(samples_output, name='cldm_samples_output')

        #rgb_output = inpaint_out if self.paint_step > 1 else samples_output 
        rgb_output = samples_output        
        rgb_output = F.interpolate(rgb_output, (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size), mode='bilinear', align_corners=False)
        
        
        # Project back
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, 
                                               background = background,
                                               rgb_output = rgb_output,  
                                               object_mask = mask, 
                                               update_mask = update_mask, 
                                               z_normals = z_normals,
                                               z_normals_cache = z_normals_cache)
        self.log_train_image(fitted_pred_rgb, name='fitted')

        return fitted_pred_rgb
    
    

    def generate_3D_SAM(self, data: Dict[str, Any], points=None, labels=None, phi_offset=None):
        
        logger.info(f'--- Generate_3D_SAM step #{self.edit_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        if phi_offset is not None:
            phi = phi + np.deg2rad(phi_offset)

        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        
        logger.info(f'Generate_3D_SAM from theta: {round(np.rad2deg(theta))}, phi: {round(np.rad2deg(phi))}, radius: {radius}')
        # Render from viewpoint
        background = torch.Tensor([0, 0, 0]).to(self.device) 
        mask_outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background, use_texture_mask=True)
        render_cache = mask_outputs['render_cache']
        mask_texture_img = mask_outputs['texture_map']

        original_outputs = self.mesh_model.render(render_cache=render_cache, background=background, use_texture_original=True)
        mask_zero_outputs = self.mesh_model.render(render_cache=render_cache, background=background, use_texture_mask_zero=True)
        
        mask = mask_outputs['mask']
        edit_mask = mask_outputs['image']
        edit_zero_mask = mask_zero_outputs['image']
        original_rgb_render = original_outputs['image']
        mask_zero_texture_img = mask_zero_outputs['texture_map']
        ###
        edit_mask = edit_mask * mask
        edit_zero_mask = edit_zero_mask * mask
        Incomplete_sam_view = self.sam3d.save_sam_img(edit_mask, original_rgb_render, path=self.edit_path, now_step=self.edit_step, name='Incomplete_sam_view')
        
        #convert to [1,1,w,w]
        edit_mask_gray = utils.rgb2gray(edit_mask)
        edit_mask_gray = edit_mask_gray * mask
        
        edit_zero_mask_gray = utils.rgb2gray(edit_zero_mask)
        edit_zero_mask_gray = edit_zero_mask_gray * mask
        
        if self.edit_step > 1:
            input_point, input_label = self.sam3d.mask_to_SAMInput(edit_mask_gray[0,0], edit_zero_mask_gray[0,0])
            #self.log_train_image(edit_zero_mask, name='edit_zero_mask', path=self.edit_path, now_step=self.edit_step)

        
        if points is not None and labels is not None:
            input_point = points
            input_label = labels
        
        edit_generation_mask = self.sam3d.get_sam_mask(original_rgb_render, self.sam3d.predictor, input_point=input_point, input_label=input_label)
        
        edit_generation_mask = edit_generation_mask * mask
        edit_generation_mask = edit_generation_mask * (torch.abs(1 - edit_zero_mask_gray))
        edit_generation_mask[torch.bitwise_or(edit_generation_mask == 1, edit_mask_gray == 1)] = 1

        edit_generation_zero_mask = torch.zeros_like(edit_generation_mask)
        edit_generation_zero_mask[torch.bitwise_and(edit_generation_mask == 0, mask == 1)] = 1
        
        # Project back (mask_texture)
        pred_mask_edit, _ = self.project_back(render_cache=render_cache, 
                                               background = background,
                                               rgb_output = edit_generation_mask,  
                                               object_mask = mask, 
                                               update_mask = mask,
                                               use_texture_mask = True)
        #if self.edit_step ==1:
        pred_zeromask_edit, _ = self.project_back(render_cache=render_cache, 
                                            background = background,
                                            rgb_output = edit_generation_zero_mask,  
                                            object_mask = mask, 
                                            update_mask = mask,
                                            use_texture_mask_zero = True)
        
        self.log_train_image(edit_mask_gray.repeat(1,3,1,1), name='edit_mask_raw', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(edit_generation_mask.repeat(1,3,1,1), name='edit_mask_sam', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(original_rgb_render, name='original_rgb_render', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(pred_mask_edit, name='pred_mask_edit', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(pred_zeromask_edit, name='pred_zeromask_edit', path=self.edit_path, now_step=self.edit_step)
        torchvision.utils.save_image(mask_zero_texture_img, os.path.join(self.texture_img_path, f'mask_zero_texture_{self.edit_step:04d}.jpg'))
        torchvision.utils.save_image(mask_texture_img, os.path.join(self.texture_img_path, f'mask_texture_{self.edit_step:04d}.jpg'))
        return pred_mask_edit, original_rgb_render
    
    def edit_viewpoint(self, data: Dict[str, Any], phi_offset=None):
        
        logger.info(f'--- Editing step #{self.edit_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        
        if phi_offset is not None:
            phi = phi + np.deg2rad(phi_offset)

        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset) #view offset
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
    
        
        edit_model = self.cfg.guide.edit_model   ## two Options ['Depth_Aware','Normal_Aware'] 
        
        logger.info(f'Editing from theta: {round(np.rad2deg(theta))}, phi: {round(np.rad2deg(phi))}, radius: {radius}')

        background = torch.Tensor([0, 0, 0]).to(self.device) 
        background_gray = torch.Tensor([0.8, 0.8, 0.8]).to(self.device) 


        edited_outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background_gray); render_cache = edited_outputs['render_cache']         
        fulledit_mask_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_texture_mask=True, render_cache=render_cache)
        fulledit_mask = fulledit_mask_outputs['image']
        fulledit_mask = utils.rgb2gray(fulledit_mask)
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)
        

        mask = edited_outputs['mask']
        normal_raw = edited_outputs['normals']#[1,3,w,w]
        z_normals = edited_outputs['normals'][:, -1:, :, :].clamp(0, 1) #[1,1,w,w]
        edited_texture_img = edited_outputs['texture_map']
        edited_rgb_render = edited_outputs['image']
        depth_render = edited_outputs['depth']
        z_normals_cache = meta_output['image'].clamp(0, 1)       #[1,3,w,w]
        self.log_train_image(edited_rgb_render, name='edited_rgb_render_raw', path=self.edit_path, now_step=self.edit_step)

        torchvision.utils.save_image(edited_texture_img, os.path.join(self.texture_img_path, f'edited_texture_img_{self.edit_step:04d}.jpg'))


        generate_mask, refine_mask, junction_mask, update_mask, shaded_rgb = self.calculate_trimap(rgb_render_raw=edited_rgb_render,
                                                                             depth_render=depth_render,
                                                                             z_normals=z_normals,
                                                                             z_normals_cache=z_normals_cache,
                                                                             edited_mask=fulledit_mask,
                                                                             mask=mask,
                                                                             editting=True)
        
        ## generate_mask, refine_mask, junction_mask: [1,1,w,w]
        exact_edit_generation_mask = update_mask * fulledit_mask

        self.log_train_image(generate_mask.repeat(1,3,1,1), name='generate_mask', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(refine_mask.repeat(1,3,1,1), name='refine_mask', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(junction_mask.repeat(1,3,1,1), name='junction_mask', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(update_mask.repeat(1,3,1,1), name='update_mask', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(fulledit_mask.repeat(1,3,1,1), name='fulledit_mask', path=self.edit_path, now_step=self.edit_step)        
        self.log_train_image(exact_edit_generation_mask.repeat(1,3,1,1), name='exact_edit_generation_mask', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(z_normals_cache[0, 0], name='z_normals_cache', colormap=True, path=self.edit_path, now_step=self.edit_step)
        #self.log_train_image(shaded_rgb, 'shaded_input', path=self.edit_path, now_step=self.edit_step)
        
        

        angle_overhead = np.deg2rad(self.cfg.render.overhead_range)
        angle_front = np.deg2rad(self.cfg.render.front_range)
        thetas = torch.FloatTensor([theta]).to(self.device)
        phis = torch.FloatTensor([phi]).to(self.device) 
        dirs = get_view_direction(thetas, phis, overhead=angle_overhead, front=angle_front)
        
        if self.cfg.guide.append_direction:
            text_prompt = self.cfg.guide.edit_text + ", {} view."
            text_prompt = text_prompt.format(self.view_dirs[dirs])
        else:
            text_prompt = self.cfg.guide.edit_text
        edit_text = text_prompt + self.a_prompt
        guidance_scale = self.cfg.guide.guidance_scale
        ddim_steps = self.cfg.guide.ddim_steps
        strength = self.cfg.guide.strength

        ##draw editing region to shaded colorrgb
        #index_generate_mask = fulledit_mask.type(torch.bool)
        #edited_rgb_render[index_generate_mask.repeat(1,3,1,1)] = shaded_rgb[index_generate_mask.repeat(1,3,1,1)]


        exact_edit_generation_mask_1024 = F.interpolate(exact_edit_generation_mask, (1024, 1024), mode='bilinear', align_corners=False)
        edited_rgb_render_1024 = F.interpolate(edited_rgb_render, (1024, 1024), mode='bilinear', align_corners=False)

        logger.info(f'edit_text: {edit_text}')

        
        if edit_model == 'Depth_Aware':
            # Crop to inner region based on object mask
            min_h, min_w, max_h, max_w = utils.get_nonzero_region(mask[0, 0])
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render = crop(edited_rgb_render)
            cropped_depth_render = crop(depth_render)
            cropped_update_mask = crop(update_mask)
            cropped_rgb_output, steps_vis = self.depth_diffusion.img2img_step(edit_text, cropped_rgb_render.detach(),
                                                                        cropped_depth_render.detach(),
                                                                        guidance_scale=self.cfg.guide.guidance_scale,
                                                                        strength=1.0, update_mask=cropped_update_mask,
                                                                        fixed_seed=self.cfg.optim.seed,
                                                                        check_mask=None,
                                                                        intermediate_vis=self.cfg.log.vis_diffusion_steps)

            
            cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                            (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                            mode='bilinear', align_corners=False)

            # Extend rgb_output to full image size
            edited_inpaint_image = edited_rgb_render.clone()
            edited_inpaint_image[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
            
        elif edit_model == 'Normal_Aware':
            H = W = self.sample_size 
            shape = (4, H//8, W//8) 
            ##single image detect normal (surface_normal_uncertainty)
            torchvision.utils.save_image(edited_texture_img, os.path.join(self.texture_img_path, f'edited_texture_img_{self.edit_step:04d}.jpg'))
            control = self.apply_normal(normal_raw, mask , H, W)

            x0 = self.encode_imgs(edited_rgb_render_1024)
            # x0 = self.cldm_model_guide.get_first_stage_encoding(self.cldm_model_guide.encode_first_stage(edited_rgb_render_1024))  


            cond = {"c_concat": [control], "c_crossattn": [self.cldm_model_guide.get_learned_conditioning([text_prompt + ', ' + self.a_prompt])]}  #normal [1,3,h,w]   h=w=1024=train_grid_size
            un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.cldm_model_guide.get_learned_conditioning([self.n_prompt])]}
                         
            self.cldm_model_guide.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            update_mask_cldm = F.interpolate(exact_edit_generation_mask_1024, (H//8, W//8), mode='bilinear', align_corners=False) 
            
            update_mask_cldm = update_mask_cldm.permute(0,2,3,1).squeeze(0).cpu().numpy()
            update_mask_cldm = torch.from_numpy(update_mask_cldm[:,:,::-1].copy())
            update_mask_cldm = update_mask_cldm.unsqueeze(0).permute(0,3,1,2).to(self.device)
            update_mask_cldm = F.interpolate(update_mask_cldm, (H//8, W//8), mode='bilinear', align_corners=False) 
            
            samples, intermediates = self.ddim_sampler_guide.sample(ddim_steps, 1,
                                                                    shape, cond, verbose=False, eta=0, mask=update_mask_cldm, x0=x0,
                                                                    unconditional_guidance_scale=guidance_scale,
                                                                    unconditional_conditioning=un_cond)
            # edited_inpaint_image_1024 = self.cldm_model_guide.decode_first_stage(samples)
            edited_inpaint_image_1024 = self.decode_latents(samples) #[1,3,h,w] ####
                                       

            edited_inpaint_image = F.interpolate(edited_inpaint_image_1024, (mask.shape[2], mask.shape[3]), mode='bilinear', align_corners=False)
            self.log_train_image(control, name='normal', path=self.edit_path, now_step=self.edit_step)
          
        # Project back (edited_texture)
        pred_rgb_edit, _ = self.project_back(render_cache=render_cache, 
                                             background = background,
                                             rgb_output = edited_inpaint_image,  
                                             object_mask = mask, 
                                             update_mask = exact_edit_generation_mask, 
                                             z_normals = z_normals,
                                             z_normals_cache = z_normals_cache,
                                             editting=True)
        
        
        self.log_train_image(edited_rgb_render, name='edited_inpaint_before_img', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(edited_inpaint_image, name='edited_inpaint_after_img', path=self.edit_path, now_step=self.edit_step)
        self.log_train_image(pred_rgb_edit, name='pred_rgb_edit', path=self.edit_path, now_step=self.edit_step)
        return pred_rgb_edit

    
    def text_inpainting(self, theta, phi, radius, viewImg):
        background = torch.Tensor([0, 0, 0]).to(self.device) 
        print('theta: ',theta,' phi: ',phi, 'radius: ',radius)
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        theta = torch.FloatTensor([theta]).to(self.device)
        phi = torch.FloatTensor([phi]).to(self.device)

        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background); 
        render_cache = outputs['render_cache']         
        mask = outputs['mask']
        H, W = mask.shape[2], mask.shape[3] 
        print(viewImg.shape)
        viewImg = F.interpolate(viewImg, (H, W), mode='bilinear', align_corners=False) 
        #viewImg = viewImg * mask
        print(H,W, viewImg.shape)
        pred_rgb_edit, _ = self.project_back(render_cache=render_cache, 
                                        background = background,
                                        rgb_output = viewImg,  
                                        object_mask = mask, 
                                        update_mask = mask,
                                        editting=True)
        return pred_rgb_edit
    
    def calculate_trimap(self, 
                         rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, 
                         z_normals_cache: torch.Tensor, 
                         edited_mask: torch.Tensor,
                         mask: torch.Tensor,
                         editting = False):
        

        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)                                  #[1,1200,1200]
            

        exact_generate_mask = (diff < 0.1).float().unsqueeze(0) #[1,1,1200,1200]
        generate_mask_raw = exact_generate_mask
        generate_dialte_kernelsize = self.cfg.guide.generate_dialte_kernelsize

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((generate_dialte_kernelsize, generate_dialte_kernelsize), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0) 

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        # Generate the refine mask based on the z normals, and the edited mask
     
        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        
        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1
        update_mask[torch.bitwise_and(object_mask == 0, exact_generate_mask == 0)] = 0


        ######## Need Modified#######
        
        # Creating kernel
        junction_dialte_kernelsize = self.cfg.guide.junction_dialte_kernelsize
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_junction_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (junction_dialte_kernelsize, junction_dialte_kernelsize))
        
        # opening: Separate objects, eliminate small areas (sparse points)
        exact_generate_mask_numpy = exact_generate_mask.permute(0,2,3,1).squeeze(0).cpu().numpy()
        exact_generate_mask_numpy = (exact_generate_mask_numpy.copy() * 255).astype(np.uint8)
        
        exact_generate_mask_numpy = cv2.morphologyEx(exact_generate_mask_numpy, cv2.MORPH_OPEN, kernel)
        #exact_generate_mask = torch.from_numpy(exact_generate_mask_numpy).float()                      #
        #exact_generate_mask = exact_generate_mask.unsqueeze(0).unsqueeze(0).clamp(0,1).to(self.device) #
        
        #dilation - erosion，to get the contour of object：
        junction_mask = cv2.morphologyEx(exact_generate_mask_numpy, cv2.MORPH_GRADIENT, kernel)
        junction_mask = cv2.dilate(junction_mask, kernel_junction_dilate)
        junction_mask = torch.from_numpy(junction_mask).float()
        junction_mask = junction_mask.unsqueeze(0).unsqueeze(0).clamp(0,1).to(self.device)
        junction_mask = torch.bitwise_and(junction_mask==1, object_mask==1).float()
        
        if mask.equal(generate_mask_raw):
            junction_mask = torch.zeros_like(generate_mask_raw).float()
        refine_mask = torch.bitwise_or(junction_mask==1, refine_mask==1).float()
        
        """
        #Dialte generate mask
        exact_generate_mask_numpy = utils.tensor_toImage(exact_generate_mask) #[w,w,1]
        exact_generate_mask_numpy = cv2.dilate(exact_generate_mask_numpy, kernel_generation_dilate) #[w,w]
        exact_generate_mask = torch.from_numpy(exact_generate_mask_numpy).float()
        exact_generate_mask = exact_generate_mask.unsqueeze(0).unsqueeze(0).clamp(0,1).to(self.device)
        """
        
        
        exact_generate_mask = edited_mask * exact_generate_mask
        generate_mask = edited_mask * generate_mask
        refine_mask = edited_mask * refine_mask
        junction_mask = edited_mask * junction_mask
        
        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[174 / 255.0, 249 / 255.0, 211 / 255.0], z_normals=z_normals) #green region: keep
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1 
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 168 / 255.0, 211 / 255.0], #red region: generate 
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask 
            
            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask #gray region:其实就是generate region,应用到shaded_input中
        
            if self.paint_step or self.edit_step > 1 :
                refinement_color_shaded = utils.color_with_shade(color=[185 / 255.0, 225 / 255.0, 238 / 255.0], #blue region: refine
                                                                 z_normals=z_normals)
                #only_old_mask_for_vis：blue(refine) region
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                       1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)

            if editting:
                self.log_train_image(trimap_vis, 'trimap', path=self.edit_path, now_step=self.edit_step)
                self.log_train_image(shaded_rgb_vis, 'shaded_input', path=self.edit_path, now_step=self.edit_step)
            else:
                self.log_train_image(shaded_rgb_vis, 'shaded_input')
                self.log_train_image(trimap_vis, 'trimap')
        
        #exact_generate_mask = torch.bitwise_or(exact_generate_mask==1, generate_mask_raw==1).float()
        update_mask = torch.bitwise_or(generate_mask==1, refine_mask==1).float()
        update_mask = edited_mask * update_mask

        return exact_generate_mask, refine_mask, junction_mask, update_mask, shaded_rgb_vis

    def project_back(self, render_cache: Dict[str, Any], 
                     background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, 
                     update_mask: torch.Tensor, 
                     z_normals: torch.Tensor=None,
                     z_normals_cache: torch.Tensor=None,
                     use_texture_mask: bool = False,
                     use_texture_mask_zero: bool = False,
                     use_texture_original: bool = False,
                     editting: bool = False):
        
        object_mask_raw = object_mask
        ####
        if use_texture_mask_zero:
            object_mask = torch.from_numpy(
                cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
                object_mask.device).unsqueeze(0).unsqueeze(0)
        else:
            object_mask = torch.from_numpy(
                cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((3, 3), np.uint8))).to(
                object_mask.device).unsqueeze(0).unsqueeze(0)
        
        ####
        render_update_mask = object_mask.clone()

        render_update_mask[update_mask == 0] = 0
        
        if z_normals is not None and z_normals_cache is not None:
            blurred_render_update_mask = torch.from_numpy(
                cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
                render_update_mask.device).unsqueeze(0).unsqueeze(0)
            blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

            # Do not get out of the object
            blurred_render_update_mask[object_mask == 0] = 0

            if self.cfg.guide.strict_projection:
                blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
                # Do not use bad normals
                z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
                blurred_render_update_mask[z_was_better] = 0

            render_update_mask = blurred_render_update_mask
            
            if editting:
                self.log_train_image(rgb_output * render_update_mask, 'project_back_input', path=self.edit_path, now_step=self.edit_step)
            else:
                self.log_train_image(rgb_output * render_update_mask, 'project_back_input')
            # Update the normals
            z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(use_texture_mask, 
                                                                use_texture_mask_zero,
                                                                use_texture_original), lr=self.cfg.optim.lr, 
                                                                betas=(0.9, 0.99), eps=1e-15)
      
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache, 
                                             use_texture_mask=use_texture_mask, 
                                             use_texture_mask_zero=use_texture_mask_zero,
                                             use_texture_original=use_texture_original)
            rgb_render = outputs['image']
            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() + (
                    (masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)).mean()

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, 
                                                  render_cache=render_cache,
                                                  use_texture_mask=use_texture_mask,
                                                  use_texture_mask_zero=use_texture_mask_zero,
                                                  use_texture_original=use_texture_original)
            current_z_normals = meta_outputs['image']
            
            if z_normals is not None and z_normals_cache is not None:
                current_z_mask = meta_outputs['mask'].flatten()
                masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False, path=None, now_step=None):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3] #[1,1,1200,1200]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            if path is None and now_step is None:
                Image.fromarray((tensor * 255).astype(np.uint8)).save(
                    self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')
            else:
                Image.fromarray((tensor * 255).astype(np.uint8)).save(
                    path / f'{now_step:04d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)
