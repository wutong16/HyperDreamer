import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset

os.system("nvidia-smi")

test_path = Path('datasets') / 'celebahq' / 'imgs_cropped' / 'test'
test_path_precompute = Path('datasets') / 'celebahq' / 'unsup3d' / 'test'

out = Path('results') / 'images' / 'data' / 'celebahq'

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

indices = [2, 6, 8, 17, 19, 33, 42, 51, 55, 196, 253]
failure_cases = [201, 207, 217, 235]
challenging = [30, 32, 33, 55, 72, 82, 140, 160, 170, 219, 240, 245, 349, 263]
comparison = [41, 84, 307, 514, 1045]
first_figure = [199, 245, 263, 358]

indices = indices + comparison

dry_run = False


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.lam_flip_start_epoch = 0


def save_plot(img, file_name=None, grey=False):
    if dry_run:
        plt.imshow(img)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))


def main():
    print('Loading dataset')

    dataset = ImageDataset(str(test_path), image_size=image_size, crop=None, is_validation=True, precomputed_dir=test_path_precompute, cfgs={'min_depth': 0, 'max_depth': 1})

    for index in indices:
        data_dict = dataset.__getitem__(index)

        input_im = data_dict['input_im'].permute(1, 2, 0).numpy()

        lr_depth = data_dict['recon_depth'].permute(1, 2, 0).squeeze().numpy()
        lr_normal = data_dict['recon_normal'].permute(1, 2, 0).numpy() * .5 + .5
        lr_albedo = data_dict['recon_albedo'].permute(1, 2, 0).numpy() * .5 + .5
        lr_mask = data_dict['recon_im_mask'].permute(1, 2, 0).squeeze().numpy()

        save_plot(input_im, str(out / f'{index:06d}_input.jpg'), False)
        save_plot(lr_depth, str(out / f'{index:06d}_lr_depth.jpg'), True)
        save_plot(lr_normal, str(out / f'{index:06d}_lr_normal.jpg'), False)
        save_plot(lr_albedo, str(out / f'{index:06d}_lr_albedo.jpg'), False)
        save_plot(lr_mask, str(out / f'{index:06d}_lr_mask.jpg'), True)


if __name__ == '__main__':
    main()