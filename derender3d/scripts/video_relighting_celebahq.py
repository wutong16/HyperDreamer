import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset
from derender3d.model import Derender3D
from derender3d.utils import unsqueezer, map_fn, to, get_ball

os.system("nvidia-smi")

cp_path = Path('results') / 'models' / 'celebahq_nr' / 'checkpoint005.pth'

test_path = Path('datasets') / 'celebahq' / 'imgs_cropped' / 'test'
test_path_precompute = Path('datasets') / 'celebahq' / 'unsup3d' / 'test'

out = Path('results') / 'videos' / 'relighting' / 'celebahq'

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
d_start = 1
d_speed = 0 / 100
a_min = .0
a_max = .0
b_min = .4
b_max = .4

resolution = 256

indices = [2, 6, 8, 17, 19, 33, 42, 51, 55, 196, 253]
failure_cases = [201, 207, 217, 235]
challenging = [30, 32, 33, 55, 72, 82, 140, 160, 170, 219, 240, 245, 349, 263]
comparison = [41, 84, 307, 514, 1045]
first_figure = [199, 245, 263, 358]
ablation_shape = [17, 65, 449]
validation = [7]

relighting_video = [42, 196, 207]

indices = relighting_video

s_ball = True

s_diff = True
s_spec = True


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.lam_flip_start_epoch = 0


def get_xy_(frame):
    rotation = rot_start + frame / rot_time * math.pi * 2.
    d = d_start + frame * d_speed
    v = torch.tensor([math.sin(rotation) * d, math.cos(rotation) * d, 1])
    v /= torch.norm(v)
    return list(v)


def get_xy(frame):
    rotation = rot_start + frame / rot_time * math.pi * 2.
    d = d_start + frame * d_speed
    return [math.sin(rotation) * d, math.cos(rotation) * d]


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
            'if_module_params': {'spec_alpha': 'single', 'spec_strength': 'single', 'spec_alpha_max': 64, 'spec_taylor': False, 'neural_refinement': True, 'shadow': False, 'nr_spec': True, 'shadow_cap': .1, 'nr_nf': 32, 'nr_depth': 6, 'nr_albedo': False, 'nr_shadow': False}
        })

    models = [model]
    cps = [cp]

    if s_ball:
        print('Rendering lighting')
        video_ball = []
        for i in range(frames):
            light_d = torch.tensor(get_xy_(i)).to(device)
            ball = get_ball(light_d, specularity=16, background=0)
            ball = ball.unsqueeze(-1).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()
            ball = (ball * 255).astype(np.uint8)
            video_ball.append(ball)
        video_ball = ImageSequenceClip(video_ball, fps=10)
        video_ball.write_videofile(str(out / 'lighting.mp4'))
        video_ball.close()
        print('Completed lighting video')


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
                file_name = f'{index:06d}.mp4'
                file_name_diff = f'{index:06d}_diff.mp4'
                file_name_spec = f'{index:06d}_spec.mp4'

                video = []
                video_diff = []
                video_spec = []

                data_dict_ = dict(data_dict)

                for i in tqdm(range(frames)):
                    light_d = torch.tensor([get_xy(i)]).to(device).repeat(1, 1)
                    lam = i / (frames - 1)
                    light_a = (1 - lam) * a_min + lam * a_max
                    light_b = (1 - lam) * b_min + lam * b_max
                    light = torch.cat([torch.tensor([[light_a, light_b]]).to(device), light_d], dim=-1)
                    light = light_d

                    with torch.no_grad():
                        model.forward(data_dict_, light=light)

                    recon_im = model.data_dict['recon_im_nr'][0][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                    light_d = light_d.cpu()

                    recon_diff = model.data_dict['recon_diffuse_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()
                    recon_spec = model.data_dict['recon_specular_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()
                    if 'neural_shading' in model.data_dict:
                        recon_spec *= model.data_dict['neural_shading'][0][0].permute(1, 2, 0).expand(-1, -1, 3).cpu().clamp(0, 1).numpy()

                    recon_im = (recon_im * 255.).astype(np.uint8)
                    recon_diff = (recon_diff * 255).astype(np.uint8)
                    recon_spec = (recon_spec * 255).astype(np.uint8)

                    video.append(recon_im)
                    video_diff.append(recon_diff)
                    video_spec.append(recon_spec)

                    if i == 0:
                        input_im = model.data_dict['input_im'][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                        albedo = model.data_dict['recon_albedo'][0][0].permute(1, 2, 0).cpu().clamp(-1, 1).numpy() / 2. + .5
                        normal = model.data_dict['recon_normal'][0][0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
                        cv2.imwrite(str(out / f'{index:06d}_input.jpg'), cv2.cvtColor((input_im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(out / f'{index:06d}_albedo.jpg'), cv2.cvtColor((albedo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(out / f'{index:06d}_normal.jpg'), cv2.cvtColor((normal * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                video = ImageSequenceClip(video, fps=10)
                video.write_videofile(str(out / file_name))
                video.close()

                if s_diff:
                    video_diff = ImageSequenceClip(video_diff, fps=10)
                    video_diff.write_videofile(str(out / file_name_diff))
                    video_diff.close()
                if s_spec:
                    video_spec = ImageSequenceClip(video_spec, fps=10)
                    video_spec.write_videofile(str(out / file_name_spec))
                    video_spec.close()


if __name__ == '__main__':
    main()