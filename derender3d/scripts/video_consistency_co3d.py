import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset
from derender3d.model import Derender3D
from derender3d.utils import to

os.system("nvidia-smi")

cp_path = Path('results') / 'models' / 'co3d' / 'checkpoint010.pth'
co3d_base = Path('datasets') / 'co3d'
cosy_base = Path('datasets') / 'cosy'
photos_base = Path('datasets') / 'photos'

category = 'hydrant'


if category != 'cosy' and category != 'photos':
    test_path = co3d_base / f'extracted_{category}' / 'imgs_cropped' / 'val'
    test_path_precompute = co3d_base / f'extracted_{category}' / 'precomputed' / 'val'
elif category == 'cosy':
    test_path = cosy_base / 'ims' / 'val'
    test_path_precompute = cosy_base / 'precomputed' / 'val'
else:
    test_path = photos_base / 'imgs_cropped' / 'val'
    test_path_precompute = None


out = Path('results') / 'videos' / category


gpu_id = 0

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

save = True

out.mkdir(exist_ok=True, parents=True)
image_size = 256

frames = 50
rot_start = -math.pi / 2
rot_time = 50
d_start = 1.0
d_speed = 0
a_min = .2
a_max = .2
b_min = .7
b_max = .7

resolution = 256

DRY_RUN = False

s_diff = True
s_spec = True
s_input = True
s_normal = True


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.lam_flip_start_epoch = 0


def main():
    print('Loading dataset')

    dataset_params = {'flip_normal_x': True, 'flip_normal_y': True}

    dataset = ImageDataset(str(test_path), image_size=image_size, crop=None, is_validation=True, precomputed_dir=test_path_precompute, cfgs={'min_depth': .9, 'max_depth': 1.1, 'dataset_params': dataset_params})
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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
            'compute_loss': False,
            'compute_metrics': False,
            'metrics_module': 'DecompositionMetrics',
            'if_module_params': {'spec_alpha': 'single', 'spec_strength': 'single', 'spec_strength_min': 0.1, 'neural_refinement': False}
        })

    models = [model]
    cps = [cp]


    for model, cp in zip(models, cps):
        model.trainer = DummyTrainer()
        model.load_model_state(cp)
        model.to_device(device)
        model.set_eval()

        file_name = f'{category}.mp4'
        file_name_input = f'{category}_input.mp4'
        file_name_diff = f'{category}_diff.mp4'
        file_name_spec = f'{category}_spec.mp4'
        file_name_normal = f'{category}_normal.mp4'

        video = []
        video_input = []
        video_diff = []
        video_spec = []
        video_normal = []

        for (i, data_dict) in tqdm(enumerate(dataloader)):
            data_dict = to(data_dict, device)

            with torch.no_grad():
                model.forward(data_dict)

            recon_im = model.data_dict['input_im'][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5

            mask = model.data_dict['lr_recon_im_mask'][0, 0].cpu().numpy() > 0

            if mask.shape[-1] != recon_im.shape[-1]:
                mask = np.broadcast_to(np.expand_dims(mask, -1), recon_im.shape)

            recon_im = np.array(recon_im)

            recon_diff = model.data_dict['recon_diffuse_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()
            recon_spec = model.data_dict['recon_specular_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()
            if 'neural_shading' in model.data_dict:
                recon_spec *= model.data_dict['neural_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()

            recon_normal = model.data_dict['recon_normal'][0][0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy() * .5 + .5

            recon_diff[~mask] = 0
            recon_spec[~mask] = 0
            recon_normal[~mask] = 0

            recon_im = (recon_im * 255.).astype(np.uint8)
            recon_diff = (recon_diff * 255).astype(np.uint8)
            recon_spec = (recon_spec * 255).astype(np.uint8)
            recon_normal = (recon_normal * 255).astype(np.uint8)
            combined = np.concatenate((recon_im, recon_normal, recon_diff, recon_spec), axis=1)

            video_input.append(recon_im)
            video_diff.append(recon_diff)
            video_spec.append(recon_spec)
            video_normal.append(recon_normal)
            video.append(combined)

        if not DRY_RUN:
            video = ImageSequenceClip(video, fps=10)
            video.write_videofile(str(out / file_name))
            video.close()

            if s_input:
                video_input = ImageSequenceClip(video_input, fps=10)
                video_input.write_videofile(str(out / file_name_input))
                video_input.close()
            if s_diff:
                video_diff = ImageSequenceClip(video_diff, fps=10)
                video_diff.write_videofile(str(out / file_name_diff))
                video_diff.close()
            if s_spec:
                video_spec = ImageSequenceClip(video_spec, fps=10)
                video_spec.write_videofile(str(out / file_name_spec))
                video_spec.close()
            if s_normal:
                video_normal = ImageSequenceClip(video_normal, fps=10)
                video_normal.write_videofile(str(out / file_name_normal))
                video_normal.close()


if __name__ == '__main__':
    main()