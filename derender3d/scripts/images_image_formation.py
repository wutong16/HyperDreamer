import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset
from derender3d.model import Derender3D
from derender3d.utils import unsqueezer, map_fn, to, get_ball

os.system("nvidia-smi")

cp_path = Path('results') / 'models' / 'celebahq_nr' / 'checkpoint005.pth'

test_path = Path('datasets') / 'celebahq' / 'imgs_cropped' / 'test'
test_path_precompute = Path('datasets') / 'celebahq' / 'unsup3d' / 'test'
out = Path('results') / 'images' / 'image_formation' / 'celebahq'

gpu_id = 0

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

out.mkdir(exist_ok=True, parents=True)
image_size = 256

frames = 1
rot_start = -math.pi / 2
rot_time = 8
d_start = 1.2
d_speed = .0 / 100
a_min = .0
a_max = .0
b_min = 1
b_max = 1

p_normal = True
p_depth = True
p_bump = True
p_normal_noref = True
p_diff = True
p_spec = False
p_input = True
p_albedo = True
p_ball = False
p_specs = False
p_cls_diff = True
p_cls_spec = True
p_shadow = False

resolution = 256

indices = [2, 6, 8, 17, 19, 33, 42, 51, 55, 196, 253]
failure_cases = [201, 207, 217, 235]
challenging = [30, 32, 33, 55, 72, 82, 140, 160, 170, 219, 240, 245, 349, 263]
comparison = [41, 84, 307, 514, 1045]
first_figure = [199, 245, 263, 358]
ablation_shape = [17, 65, 449]

indices = first_figure[1:2]

dry_run = True


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.lam_flip_start_epoch = 0


def get_xy(frame):
    rotation = rot_start + frame / rot_time * math.pi * 2.
    d = d_start + frame * d_speed
    return [math.sin(rotation) * d, math.cos(rotation) * d]


def save_plot(img, file_name=None, grey=False):
    if dry_run:
        plt.imshow(img)
        plt.title(file_name)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))


def main():
    print('Loading dataset')

    dataset = ImageDataset(str(test_path), image_size=image_size, crop=None, is_validation=True, precomputed_dir=test_path_precompute, cfgs={'min_depth': .9, 'max_depth': 1.1})

    print('Loading checkpoint')
    cp = torch.load(cp_path, map_location=device)

    print('Building model')
    model = Derender3D({
            'device': device,
            'predict_geometry': 'hr_depth',
            'image_size': 256,
            'use_gan': False,
            'autoencoder_depth': 9,
            'not_load_nets': ['netDisc'],
            'if_module_params': {'spec_alpha': 'single', 'spec_strength': 'single', 'spec_alpha_max': 64, 'neural_refinement': True, 'shadow': False, 'nr_spec': True, 'shadow_cap': .1, 'nr_nf': 32, 'nr_depth': 6,'nr_albedo': False}
        })

    models = [model]
    cps = [cp]

    for model, cp in zip(models, cps):
        model.trainer = DummyTrainer()
        model.load_model_state(cp)
        model.to_device(device)
        model.set_eval()

        for index in indices:
            data_dict = dataset.__getitem__(index)
            map_fn(data_dict, unsqueezer)
            data_dict = to(data_dict, device)

            for model in models:
                data_dict_ = dict(data_dict)

                for i in tqdm(range(frames+1)):
                    light_d = torch.tensor([get_xy(i)]).to(device).repeat(1, 1)
                    lam = i / max((frames - 1), 1)
                    light_a = (1 - lam) * a_min + lam * a_max
                    light_b = (1 - lam) * b_min + lam * b_max
                    light_d_norm = torch.cat([light_d, torch.ones_like(light_d)[:, :1]], dim=1)
                    light_d_norm = light_d_norm / torch.norm(light_d_norm, dim=1, keepdim=True)
                    # light = torch.cat([torch.tensor([[light_a, light_b]]).to(device), light_d], dim=-1)
                    light = light_d

                    if i == frames:
                        light = None

                    with torch.no_grad():
                        model.forward(data_dict_, light=light)

                    recon_im = model.data_dict['recon_im_nr'][0][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                    light_d = light_d.cpu()

                    ball = get_ball(light_d_norm[0].cpu()).numpy()

                    save_plot(recon_im, str(out / f'{index:06d}_{i}.jpg'), False)
                    if p_ball:
                        save_plot(ball, str(out / f'shading_{i}.jpg'), True)

                    diffuse = model.data_dict['neural_shading'][0][0, 0].detach().cpu().numpy()
                    if p_diff:
                        save_plot(diffuse, str(out / f'{index:06d}_{i}_diff.jpg'), grey=True)
                    if p_spec:
                        save_plot(spec, str(out / f'{index:06d}_{i}_spec.jpg'), grey=True)
                    if p_specs:
                        specs = [s[0, 0].detach().cpu().numpy() for s in model.data_dict['spec_maps']]
                        for j, s in enumerate(specs):
                            save_plot(s, str(out / f'{index:06d}_{i}_specs_{j}.jpg'), grey=True)
                    if p_cls_diff:
                        diffuse = model.data_dict['recon_diffuse_shading'][0][0, 0].detach().cpu().numpy()
                        save_plot(diffuse, str(out / f'{index:06d}_{i}_cls_diff.jpg'), grey=True)
                    if p_cls_spec:
                        spec = model.data_dict['recon_specular_shading'][0][0, 0].detach().cpu().numpy()
                        save_plot(spec, str(out / f'{index:06d}_{i}_cls_spec.jpg'), grey=True)
                    if p_shadow:
                        shadow = model.data_dict['recon_shadow'][0, 0].detach().cpu().numpy()
                        save_plot(shadow, str(out / f'{index:06d}_{i}_shadow.jpg'), grey=True)

                    if i == 0:
                        input_im = model.data_dict['input_im'][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                        albedo = model.data_dict['recon_albedo'][0][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                        depth = (model.data_dict['recon_depth'][0][0].detach().squeeze().cpu().numpy() - model.min_depth) / (model.max_depth - model.min_depth)
                        normal = model.data_dict['recon_normal'][0][0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                        if p_input:
                            save_plot(input_im, str(out / f'{index:06d}_input.jpg'), False)
                        if p_albedo:
                            save_plot(albedo, str(out / f'{index:06d}_albedo.jpg'), False)
                        if p_depth:
                            save_plot(depth, str(out / f'{index:06d}_depth.jpg'), True)
                        if p_normal:
                            save_plot(normal, str(out / f'{index:06d}_normal.jpg'), False)
                        if p_bump:
                            bump = model.data_dict['recon_bump'][0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                            save_plot(bump, str(out / f'{index:06d}_bump.jpg'), False)
                        if p_normal_noref:
                            normal_noref = model.data_dict['recon_normal_noref'][0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                            save_plot(normal_noref, str(out / f'{index:06d}_normal_noref.jpg'), False)


if __name__ == '__main__':
    main()