import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.extend(['.', '..'])
from derender3d.dataloaders import ImageDataset
from derender3d.meters import StandardMetrics
from derender3d.model import Derender3D
from derender3d.model_loader import LoaderModel
from derender3d.utils import unsqueezer, map_fn, to

os.system("nvidia-smi")

test_path = Path('datasets') / 'cosy' / 'ims' / 'val'
test_path_precompute = Path('datasets') / 'cosy' / 'precomputed' / 'val'

external_path = Path('external')

out = Path('out_img/comparison_additional/dpr')

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

resolution = 256


indices = list(range(40))

dry_run = True


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.lam_flip_start_epoch = 0


def getSingleImageShapeAndSVBRDF():
    depth_path = external_path / 'SingleImageShapeAndSVBRDF' / 'output' / 'depth'
    normal_path = external_path / 'SingleImageShapeAndSVBRDF' / 'output' / 'normal'
    albedo_path = external_path / 'SingleImageShapeAndSVBRDF' / 'output' / 'albedo'

    model = LoaderModel({
        'depth_folder': str(depth_path),
        'normal_folder': str(normal_path),
        'albedo_folder': str(albedo_path),
        'metrics_module': 'DecompositionMetrics'
    })

    return model


def getSIRFS():
    albedo_path = external_path / 'SIRFS' / 'results' / 'reflectance_map_div2'
    normal_path = external_path / 'SIRFS' / 'results' / 'normal_map'

    model = LoaderModel({
        'normal_folder': str(normal_path),
        'albedo_folder': str(albedo_path),
        'metrics_module': 'DecompositionMetrics'
    })

    return model


def getShapeNetIntrinsics():
    albedo_path = external_path / 'shapenet-intrinsics' / 'train' / 'out' / 'albedo'

    model = LoaderModel({
        'albedo_folder': str(albedo_path),
        'metrics_module': 'DecompositionMetrics'
    })

    return model


def getNeuralRelighting():
    depth_path = external_path / 'neuralrelighting' / 'cosy' / 'depth'
    normal_path = external_path / 'neuralrelighting' / 'cosy' / 'normal'
    albedo_path = external_path / 'neuralrelighting' / 'cosy' / 'albedo'

    model = LoaderModel({
        'depth_folder': str(depth_path),
        'normal_folder': str(normal_path),
        'albedo_folder': str(albedo_path),
        'metrics_module': 'DecompositionMetrics'
    })

    return model


def getDerender3D():
    cp_path = Path('results') / 'models' / 'co3d' / 'checkpoint010.pth'

    model = Derender3D({
        'device': device,
        'predict_geometry': 'hr_depth',
        'image_size': 256,
        'use_gan': False,
        'not_load_nets': ['netDisc'],
        'compute_loss': False,
        'autoencoder_depth': 9,
        'metrics_module': 'DecompositionMetrics',
        'if_module_params': {'spec_alpha': 'single', 'spec_strength': 'single', 'light_y_down': False}
    })

    model.load_model_state(torch.load(cp_path, map_location=device))

    return model


def main():
    print('Loading dataset')

    dataset = ImageDataset(str(test_path), image_size=image_size, crop=None, is_validation=True, precomputed_dir=test_path_precompute, cfgs={'min_depth': .9, 'max_depth': 1.1, 'dataset_params': {'flip_normal_x': False, 'flip_normal_y': False}})

    metrics = StandardMetrics()

    print('Building model')

    # Pick right model
    model = getDerender3D()

    model.trainer = DummyTrainer()
    model.to_device(device)
    model.set_eval()

    for index in tqdm(indices):
        data_dict = dataset.__getitem__(index)
        map_fn(data_dict, unsqueezer)
        data_dict = to(data_dict, device)

        data_dict_ = dict(data_dict)

        with torch.no_grad():
            metrics_dict = model.forward(dict(data_dict_))
        metrics.update(metrics_dict)

    print(metrics)


if __name__ == '__main__':
    main()
