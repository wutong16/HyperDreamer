'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''
import os
import os.path

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
# sys.path.insert(0, '../zero123')
from zero123.ldm.models.diffusion.ddim import DDIMSampler
from zero123.ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import argparse
# from tools.camera_pose_visualizer import CameraPoseVisualizer
import json
_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3: Zero-shot One Image to 3D Object'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location=device)
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z, T=None):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            if T is None:
                T = torch.tensor([math.radians(x), math.sin(
                    math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)

            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()



def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].

    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run(models, device, cam_vis=None, return_what='gen',
             x=0.0, y=0.0, z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256, output_dir='outputs/tmp/', view_idx=0):
    '''
    :param raw_im (PIL Image).
    '''
    
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        to_return = [None] * 10
        description = ('###  <span style="color:red"> Unfortunately, '
                       'potential NSFW content was detected, '
                       'which is not supported by our model. '
                       'Please try again with a different image. </span>')
        if 'angles' in return_what:
            to_return[0] = 0.0
            to_return[1] = 0.0
            to_return[2] = 0.0
            to_return[3] = description
        else:
            to_return[0] = description
        return to_return

    else:
        print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, preprocess)

    # if np.random.rand() < 0.3:
    #     description = ('Unfortunately, a human, a face, or potential NSFW content was detected, '
    #                    'which is not supported by our model.')
    #     if vis_only:
    #         return (None, None, description)
    #     else:
    #         return (None, None, None, description)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    if 'rand' in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0

    if cam_vis is not None:
        cam_vis.polar_change(x)
        cam_vis.azimuth_change(y)
        cam_vis.radius_change(z)
        cam_vis.encode_image(show_in_im1)
        new_fig = cam_vis.update_figure()
    else:
        new_fig = None

    if 'vis' in return_what:
        description = ('The viewpoints are visualized on the top right. '
                       'Click Run Generation to update the results on the bottom right.')

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif 'gen' in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models['turncam'])
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                      ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        save_dir = os.path.join(output_dir, 'view_{:03d}'.format(view_idx))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        radius_m = 2.5 + z * 1.5
        c2w = pose_spherical(x, y, radius_m).numpy()

        meta = dict(frames=[])
        for i, img in enumerate(output_ims):
            img.save(os.path.join(save_dir, 'img_{:02d}.png'.format(i)))
            meta['frames'].append(dict(file_path='view_{:03d}/img_{:02d}.png'.format(view_idx, i), transform_matrix=c2w))

        with open(os.path.join(save_dir, 'transforms.json'), 'w', encoding='utf-8') as fp:
            json.dump(meta, fp, indent=4, cls=NumpyEncoder)
            print('saving to ', save_dir)

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T

def polar2RT(polar_deg, azimuth_deg, radius_m):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.
    base_radius = 2.5
    zoom_scale = 1.5
    radius_m = base_radius + radius_m * zoom_scale

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)
    camera_T = np.asarray([cam_x, cam_y, cam_z])
    c2w = np.concatenate([camera_R, camera_T[..., None]], 1)
    bottom = np.zeros(4)
    bottom[-1] = 1.
    c2w_ = np.concatenate([c2w, bottom[None]], 0)

    return c2w_

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = c2w @ torch.Tensor(np.array([[1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]]))
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w



def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='../pretrained/105000.ckpt',
        config='configs/sd-objaverse-finetune-c_concat-256.yaml',
        img_file=None):

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        device_idx = int(sys.argv[1])
        print('new device_idx:', device_idx)

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    # Reduce NSFW false positives.
    # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
    # models['nsfw'].concept_embeds_weights:
    # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
    #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
    # models['nsfw'].special_care_embeds_weights:
    # [0.1950, 0.2000, 0.2200].
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    models['nsfw'].concept_embeds_weights *= 1.07
    models['nsfw'].special_care_embeds_weights *= 1.07

    angle_x, angle_y, angle_z = 0., 0., 0.
    raw_im = Image.open(img_file)

    # visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [-5, 5])
    # c2w = polar2RT(0, 0, angle_z)
    # visualizer.extrinsic2pyramid(c2w, 'red', 0.5)
    # c2w = polar2RT(0, 90, angle_z)
    # visualizer.extrinsic2pyramid(c2w, 'blue', 0.5)
    # c2w = polar2RT(90, 0, angle_z)
    # visualizer.extrinsic2pyramid(c2w, 'green', 0.5)
    # visualizer.show()
    # visualizer.imsave('sample_camera.png')
    output_dir = 'outputs/demo_minion_dense'
    raw_im.save(os.path.join(output_dir, 'raw_img.png'))
    with open(os.path.join(output_dir, 'raw_transforms.json'), 'w') as f:
        json.dump(dict(transform_matrix=pose_spherical(0,0,0).numpy().tolist()), f)

    view_idx = 0
    ''' # nerf camera distribution
    for angle_x in range(-90, 91, 15):
        for angle_y in range(-180, 180, 15):
            if angle_x == 0 and angle_y == 0:
                print(view_idx)
                continue
            main_run(models, device, None, 'angles_gen',
                     x=angle_x, y=angle_y, z=angle_z,
                     raw_im=raw_im, preprocess=True,
                     scale=3.0, n_samples=1, ddim_steps=75, ddim_eta=1.0,
                     precision='fp32', h=256, w=256, output_dir=output_dir, view_idx=view_idx)
            view_idx += 1
    '''
    # GET3D distribution
    rotation_list = []
    elevation_list = []
    for angle_x in range(-30, 0, 3):
        for angle_y in range(-180, 180, 15):
            if angle_x == 0 and angle_y == 0:
                print(view_idx)
                continue
            main_run(models, device, None, 'angles_gen',
                     x=angle_x, y=angle_y, z=angle_z,
                     raw_im=raw_im, preprocess=True,
                     scale=3.0, n_samples=4, ddim_steps=75, ddim_eta=1.0,
                     precision='fp32', h=256, w=256, output_dir=output_dir, view_idx=view_idx)

            rotation = angle_y + 180
            elevation = -angle_x
            rotation_list.append(rotation)
            elevation_list.append(elevation)
            view_idx += 1

    np.save(os.path.join(output_dir, 'rotation'),
            rotation_list)
    np.save(os.path.join(output_dir, 'elevation'),
            elevation_list)

    exit()
    with open('instructions.md', 'r') as f:
        article = f.read()

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                image_block = gr.Image(type='pil', image_mode='RGBA',
                                       label='Input image of single object')
                preprocess_chk = gr.Checkbox(
                    True, label='Preprocess image automatically (remove background and recenter object)')
                # info='If enabled, the uploaded image will be preprocessed to remove the background and recenter the object by cropping and/or padding as necessary. '
                # 'If disabled, the image will be used as-is, *BUT* a fully transparent or white background is required.'),

                gr.Markdown('*Try camera position presets:*')
                with gr.Row():
                    left_btn = gr.Button('View from the Left', variant='primary')
                    above_btn = gr.Button('View from Above', variant='primary')
                    right_btn = gr.Button('View from the Right', variant='primary')
                with gr.Row():
                    random_btn = gr.Button('Random Rotation', variant='primary')
                    below_btn = gr.Button('View from Below', variant='primary')
                    behind_btn = gr.Button('View from Behind', variant='primary')

                gr.Markdown('*Control camera position manually:*')
                polar_slider = gr.Slider(
                    -90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)')
                # info='Positive values move the camera down, while negative values move the camera up.')
                azimuth_slider = gr.Slider(
                    -180, 180, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)')
                # info='Positive values move the camera right, while negative values move the camera left.')
                radius_slider = gr.Slider(
                    -0.5, 0.5, value=0.0, step=0.1, label='Zoom (relative distance from center)')
                # info='Positive values move the camera further away, while negative values move the camera closer.')

                samples_slider = gr.Slider(1, 8, value=4, step=1,
                                           label='Number of samples to generate')

                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1,
                                             label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=75, step=5,
                                             label='Number of diffusion inference steps')

                with gr.Row():
                    vis_btn = gr.Button('Visualize Angles', variant='secondary')
                    run_btn = gr.Button('Run Generation', variant='primary')

                desc_output = gr.Markdown('The results will appear on the right.', visible=_SHOW_DESC)

            with gr.Column(scale=1.1, variant='panel'):

                vis_output = gr.Plot(
                    label='Relationship between input (green) and output (blue) camera poses')

                gen_output = gr.Gallery(label='Generated images from specified new viewpoint')
                gen_output.style(grid=2)

                preproc_output = gr.Image(type='pil', image_mode='RGB',
                                          label='Preprocessed input image', visible=_SHOW_INTERMEDIATE)

        gr.Markdown(article)

        cam_vis = CameraVisualizer(vis_output)

        vis_btn.click(fn=partial(main_run, models, device, cam_vis, 'vis'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, preprocess_chk],
                      outputs=[desc_output, vis_output, preproc_output])

        run_btn.click(fn=partial(main_run, models, device, cam_vis, 'gen'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, preprocess_chk,
                              scale_slider, samples_slider, steps_slider],
                      outputs=[desc_output, vis_output, preproc_output, gen_output])

        # NEW:
        preset_inputs = [image_block, preprocess_chk,
                         scale_slider, samples_slider, steps_slider]
        preset_outputs = [polar_slider, azimuth_slider, radius_slider,
                          desc_output, vis_output, preproc_output, gen_output]
        import ipdb; ipdb.set_trace()
        left_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                  0.0, -90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        above_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   -90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        right_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   0.0, 90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        random_btn.click(fn=partial(main_run, models, device, cam_vis, 'rand_angles_gen',
                                    -1.0, -1.0, -1.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        below_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        behind_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                    0.0, 180.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)

    demo.launch(enable_queue=True, share=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_file', default='../3drec/data/nerf_wild/minion/train/r_0.png')
    args = parser.parse_args()

    run_demo(img_file=args.img_file)
