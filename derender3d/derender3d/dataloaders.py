import math
import os
from pathlib import Path

import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..derender3d import utils
from ..derender3d.renderer.utils import get_rotation_matrix


def get_set_path(data_dir, set):
    if not isinstance(data_dir, list) and not isinstance(data_dir, tuple):
        set_data_dir = os.path.join(data_dir, set)
        assert os.path.isdir(set_data_dir), "Data directory does not exist: %s" % set_data_dir
    else:
        set_data_dir = [os.path.join(tvdd, set) for tvdd in data_dir]
        for dir in set_data_dir:
            assert os.path.isdir(dir), "Training data directory does not exist: %s" % dir

    return set_data_dir


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 64)
    crop = cfgs.get('crop', None)
    test_esrc = cfgs.get('test_esrc', None)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    train_val_precomputed_dir = cfgs.get('train_val_precomputed_dir', None)
    train_val_extracted_dir = cfgs.get('train_val_extracted_dir', None)
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')
    esrc_data_dir = cfgs.get('esrc_data_dir', None)

    load_gt_depth = cfgs.get('load_gt_depth', False)
    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    train_loader = vis_loader = val_loader = test_loader = None
    if load_gt_depth:
        get_loader = lambda **kargs: get_paired_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    else:
        get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, cfgs=cfgs)

    if run_train:
        train_data_dir = get_set_path(train_val_data_dir, 'train')
        print(f"Loading training data from {train_data_dir}")

        precomputed_dir = get_set_path(train_val_precomputed_dir, 'train') if train_val_precomputed_dir is not None else None
        extracted_dir =  get_set_path(train_val_extracted_dir, 'train') if train_val_extracted_dir is not None else None

        train_loader = get_loader(data_dir=train_data_dir, is_validation=False, precomputed_dir=precomputed_dir, extracted_dir=extracted_dir)

        vis_data_dir = get_set_path(train_val_data_dir, 'val')
        print(f"Loading validatioVisualizationn data from {vis_data_dir}")

        precomputed_dir = get_set_path(train_val_precomputed_dir, 'val') if train_val_precomputed_dir is not None else None
        extracted_dir = get_set_path(train_val_extracted_dir, 'val') if train_val_extracted_dir is not None else None

        vis_loader = get_loader(data_dir=vis_data_dir, is_validation=True, precomputed_dir=precomputed_dir, extracted_dir=extracted_dir)

        if not test_esrc:
            val_loader = vis_loader
        else:
            val_loader = DataLoader(ESRCDataset(os.path.join(esrc_data_dir, 'test'), image_size, crop, is_validation=True), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    if run_test:
        if not test_esrc:
            assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
            print(f"Loading testing data from {test_data_dir}")
            test_loader = get_loader(data_dir=test_data_dir, is_validation=True)
        else:
            test_loader = DataLoader(ESRCDataset(os.path.join(esrc_data_dir, 'test'), image_size, crop, is_validation=True), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, vis_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, precomputed_dir=None, extracted_dir=None, cfgs=None):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        if precomputed_dir is not None:
            self.precomputed_dir = Path(precomputed_dir)
        else:
            self.precomputed_dir = None
        self.extracted_dir = Path(extracted_dir) if extracted_dir is not None else None
        self.paths = make_dataset(data_dir)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

        if cfgs is not None:
            self.min_depth = cfgs.get('min_depth', 0.9)
            self.max_depth = cfgs.get('max_depth', 1.1)

        dataset_params = cfgs.get('dataset_params', {}) if cfgs is not None else {}
        self.flip_normal_x = dataset_params.get('flip_normal_x', False)
        self.flip_normal_y = dataset_params.get('flip_normal_y', False)
        self.flip_normal_z = dataset_params.get('flip_normal_z', False)
        self.flip_light_x = dataset_params.get('flip_light_x', False)
        self.flip_light_y = dataset_params.get('flip_light_y', False)
        self.flip_light_z = dataset_params.get('flip_light_z', False)
        self.bias = dataset_params.get('bias', None)
        self.bias_amplitude = dataset_params.get('bias_amplitude', 1)

        self.foreground_exists = (self.precomputed_dir / 'foreground_mask').exists() if self.precomputed_dir is not None else False
        print('Foreground masks exist:', self.foreground_exists)

        self.spec_shading_exists = (self.precomputed_dir / 'spec_shading').exists() if self.precomputed_dir is not None else False
        print('Spec shading maps exist:', self.spec_shading_exists)

        self.rng_state_store = [None for _ in range(len(self))]

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def transform_array(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        if hflip:
            img = np.array(np.flip(img, axis=1))
        return tfs.functional.to_tensor(img).squeeze().to(dtype=torch.float)

    def transform_normal(self, normal, hflip=False):
        if hflip:
            normal = np.array(np.flip(normal, axis=1))
            normal[:, :, 0] = 1 - normal[:, :, 0]

        if self.flip_normal_x or self.flip_normal_y or self.flip_normal_z:
            normal = normal * 2 - 1
            normal = normal * np.array([[[-1. if self.flip_normal_x else 1., -1. if self.flip_normal_y else 1., -1. if self.flip_normal_z else 1.]]])
            normal = normal * .5 + .5

        return tfs.functional.to_tensor(normal).to(dtype=torch.float)

    def transform_light(self, light, hflip=False):
        if hflip:
            light[2] *= -1
        if self.flip_light_x or self.flip_light_y or self.flip_light_z:
            if self.flip_light_x:
                light[2] *= -1
            if self.flip_light_y:
                light[3] *= -1
            if self.flip_light_z and len(light) == 6:
                light[4] *= -1
        return torch.tensor(light, dtype=torch.float)

    def transform_view(self, view, hflip=False):
        if hflip and len(view.shape) == 1:
            view[1] *= -1
            view[2] *= -1
            view[3] *= -1
        return torch.tensor(view, dtype=torch.float)

    def transform_grid(self, grid, hflip=False):
        if hflip:
            grid[:, :, 1] *= -1
        return torch.tensor(grid)

    def recon_to_canon_light(self, light, view):
        if len(light) == 4:
            light_d = torch.cat([light[2:], torch.ones((1))])
            light_d /= torch.norm(light_d)
            light = torch.cat([light[:2], light_d])
        rot_mat = get_rotation_matrix(view[None, 0], view[None, 1], view[None, 3])
        canon_light = torch.cat([light[:2], (torch.inverse(rot_mat) @ light[None, 2:, None]).squeeze()], dim=-1)
        return canon_light

    def apply_bias(self, index, data_dict):
        if self.rng_state_store[index] is None:
            self.rng_state_store[index] = torch.random.get_rng_state()
        else:
            torch.random.set_rng_state(self.rng_state_store[index])

        if self.bias == "normals_pp" or self.bias == "normals_ps" or self.bias == "normals_pd" or self.bias == "normals_pd_det":
            normals = data_dict["recon_normal"]

            if self.bias == "normals_pp":
                angle_deviations_ab = (torch.randn_like(normals[:2, :, :]) * self.bias_amplitude) / 180 * math.pi
            elif self.bias == "normals_ps":
                angle_deviations_ab = ((torch.ones_like(normals[:2, :, :]) * torch.randn_like(normals[:2, :1, :1])) * self.bias_amplitude) / 180 * math.pi
            elif self.bias == "normals_pd":
                if not hasattr(self, "angle_deviations_ab"):
                    self.angle_deviations_ab = ((torch.ones_like(normals[:2, :, :]) * torch.randn_like(normals[:2, :1, :1])) * self.bias_amplitude) / 180 * math.pi
                angle_deviations_ab = self.angle_deviations_ab
            else:
                angle_deviations_ab = torch.ones_like(normals[:2, :, :]) * normals.new_tensor(self.bias_amplitude).view(2, 1, 1) / 180 * math.pi

            curr_a = torch.atan(normals[0] / normals[2])
            curr_b = torch.atan(normals[1] / normals[2])

            new_a = curr_a + angle_deviations_ab[0, :, :]
            new_b = curr_b + angle_deviations_ab[1, :, :]

            tan_a = torch.tan(new_a)
            tan_b = torch.tan(new_b)

            new_normals = torch.zeros_like(normals)
            new_normals[2, :, :] = torch.sqrt(1 / (1 + tan_a ** 2 + tan_b ** 2))
            new_normals[0, :, :] = tan_a * new_normals[2, :, :]
            new_normals[1, :, :] = tan_b * new_normals[2, :, :]

            data_dict["recon_normal"] = new_normals

        elif self.bias == "albedo_ps" or self.bias == "albedo_pd" or self.bias == "albedo_pd_det":
            albedo = data_dict["recon_albedo"]

            if self.bias == "albedo_ps":
                brightness_change = torch.ones_like(albedo[:1, :, :]) * (torch.randn((1, 1, 1)) * self.bias_amplitude)
            elif self.bias == "albedo_pd":
                if not hasattr(self, "brightness_change"):
                    self.brightness_change = torch.ones_like(albedo[:1, :, :]) * (torch.randn((1, 1, 1)) * self.bias_amplitude)
                brightness_change = self.brightness_change
            else:
                brightness_change = torch.ones_like(albedo[:1, :, :]) * self.bias_amplitude

            new_albedo = torch.clamp(((albedo * .5 + .5) + brightness_change) * 2. - 1., -1, 1)

            data_dict["recon_albedo"] = new_albedo

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        img = Image.open(fpath).convert('RGB')
        hflip = not self.is_validation and torch.rand(1).item()>0.5
        data_dict = {'input_im': self.transform(img, hflip=hflip), 'index': torch.tensor(index)}

        w, h = img.size

        if self.precomputed_dir is not None:
            file_name_png = f'{index:06d}.png'
            file_name_npy = f'{index:06d}.npy'
            data_dict['recon_albedo'] = self.transform_array(utils.load_image(self.precomputed_dir / 'recon_albedo' / file_name_png), hflip)[:3, :, :] * 2. - 1.
            _, h, w = data_dict['recon_albedo'].shape
            # data_dict['canon_depth'] = self.transform_array(utils.load_array(self.precomputed_dir / 'canon_depth' / file_name_png), hflip).squeeze(0) * (self.max_depth - self.min_depth) + self.min_depth
            data_dict['recon_depth'] = self.transform_array(utils.load_array(self.precomputed_dir / 'recon_depth' / file_name_png), hflip).reshape(-1, h, w)[:1, :, :] * (self.max_depth - self.min_depth) + self.min_depth
            # data_dict['canon_normal'] = self.transform_normal(utils.load_array(self.precomputed_dir / 'canon_normal' / file_name_png), hflip) * 2. - 1.
            data_dict['recon_normal'] = self.transform_normal(utils.load_array(self.precomputed_dir / 'recon_normal' / file_name_png), hflip) * 2. - 1.
            # data_dict['canon_albedo'] = self.transform_array(utils.load_image(self.precomputed_dir / 'canon_albedo' / file_name_png), hflip) * 2. - 1.
            # data_dict['canon_im'] = self.transform_array(utils.load_image(self.precomputed_dir / 'canon_im' / file_name_png), hflip) * 2. - 1.
            # data_dict['recon_im'] = self.transform_array(utils.load_image(self.precomputed_dir / 'recon_im' / file_name_png), hflip) * 2. - 1.
            # data_dict['conf_sigma_l1'] = self.transform_array(utils.load_array(self.precomputed_dir / 'conf_sigma_l1' / file_name_png), hflip)
            # data_dict['conf_sigma_percl'] = self.transform_array(utils.load_array(self.precomputed_dir / 'conf_sigma_percl' / file_name_png), hflip)
            # data_dict['canon_diffuse_shading'] = self.transform_array(utils.load_array(self.precomputed_dir / 'canon_diffuse_shading' / file_name_png), hflip)
            # data_dict['recon_diffuse_shading'] = self.transform_array(utils.load_array(self.precomputed_dir / 'recon_diffuse_shading' / file_name_png), hflip)
            data_dict['recon_im_mask'] = self.transform_array(utils.load_array(self.precomputed_dir / 'recon_im_mask' / file_name_png), hflip).reshape(-1, h, w)[:1, :, :]
            if self.foreground_exists:
                data_dict['foreground_mask'] = self.transform_array(utils.load_array(self.precomputed_dir / 'foreground_mask' / file_name_png), hflip).reshape(-1, h, w)[:1, :, :]
            if self.spec_shading_exists:
                data_dict['recon_specular_shading'] = ((self.transform_array(utils.load_array(self.precomputed_dir / 'spec_shading' / file_name_png), hflip).reshape(-1, h, w)[:1, :, :] - .5) * 2).clamp(0, 1)
            data_dict['canon_light'] = self.transform_light(utils.load_npy(self.precomputed_dir / 'canon_light' / file_name_npy), hflip)
            data_dict['view'] = self.transform_view(utils.load_npy(self.precomputed_dir / 'view' / file_name_npy), hflip)
            # data_dict['grid_2d_from_canon'] = self.transform_grid(utils.load_npy(self.precomputed_dir / 'grid_2d_from_canon' / file_name_npy), hflip)
        if self.extracted_dir is not None:
            file_name_png = f'{index:06d}.png'
            file_name_npy = f'{index:06d}.npy'
            data_dict['recon_albedo'] = self.transform_array(utils.load_image(self.extracted_dir / 'extracted_albedo' / file_name_png), hflip) * 2. - 1.
            recon_light = self.transform_light(utils.load_npy(self.extracted_dir / 'extracted_light' / file_name_npy), hflip)
            data_dict['canon_light'] = self.recon_to_canon_light(recon_light, data_dict['view'])

        if self.bias is not None:
            self.apply_bias(index, data_dict)

        return data_dict

    def __len__(self):
        return max(self.size, 1)

    def name(self):
        return 'ImageDataset'


def get_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, precomputed_dir=None, extracted_dir=None, cfgs=None):

    if isinstance(data_dir, list) or isinstance(data_dir, tuple):
        datasets = [ImageDataset(data_dir[i], image_size=image_size, crop=crop, is_validation=is_validation, precomputed_dir=precomputed_dir[i] if precomputed_dir is not None else None, extracted_dir=extracted_dir[i] if extracted_dir is not None else None, cfgs=cfgs) for i in range(len(data_dir))]
        dataset = MultiDataset(datasets)
    else:
        dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation, precomputed_dir=precomputed_dir, extracted_dir=extracted_dir, cfgs=cfgs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


## paired AB image dataset ##
def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    A_dname, B_dname = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_B = os.path.join(dir, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and torch.rand(1).item()>0.5
        return self.transform(img_A, hflip=hflip), self.transform(img_B, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'


def get_paired_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, AB_dnames=None, AB_fnames=None):

    dataset = PairedDataset(data_dir, image_size=image_size, crop=crop, \
        is_validation=is_validation, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=False
    )
    return loader


class ESRCDataset(ImageDataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, use_views=(0, 2, 4)):
        super().__init__(data_dir, image_size, crop, is_validation)

        # file_list = sorted(os.walk(data_dir))
        file_list = sorted(os.listdir(data_dir))
        ids = sorted(list(set([file_name[:5] for file_name in file_list])))
        ims_per_id = {id: [f for f in file_list if f.startswith(id)] for id in ids}

        pairs = []

        for id in ids:
            for view in use_views:
                for direction in ('L', 'S', 'R'):
                    matching_ims = [im for im in ims_per_id[id] if f'V{view}{direction}' in im]
                    count = len(matching_ims)
                    if count >= 2:
                        for i in range(count):
                            for j in range(count):
                                if i != j:
                                    pairs += [(matching_ims[i], matching_ims[j])]

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img0 = Image.open(os.path.join(self.root, pair[0])).convert('RGB')
        img1 = Image.open(os.path.join(self.root, pair[1])).convert('RGB')
        hflip = not self.is_validation and torch.rand(1).item() > 0.5
        data_dict = {'input_im': self.transform(img0, hflip=hflip), 'target_im': self.transform(img1, hflip=hflip), 'index': torch.tensor(index)}

        data_dict['recon_depth'] = torch.tensor(np.ones_like(data_dict['input_im'][:1]), dtype=torch.float32)
        data_dict['recon_normal'] = torch.tensor(np.ones_like(data_dict['input_im']), dtype=torch.float32)
        data_dict['recon_albedo'] = torch.tensor(np.ones_like(data_dict['input_im']), dtype=torch.float32)
        data_dict['recon_im'] = torch.tensor(np.ones_like(data_dict['input_im']), dtype=torch.float32)
        data_dict['recon_diffuse_shading'] = torch.tensor(np.ones_like(data_dict['input_im'][:1]), dtype=torch.float32)
        data_dict['recon_im_mask'] = torch.tensor(np.ones_like(data_dict['input_im'][:1]), dtype=torch.float32)
        data_dict['canon_light'] = torch.tensor(np.zeros((5,)), dtype=torch.float32)
        data_dict['view'] = torch.tensor(np.zeros((6,)), dtype=torch.float32)

        return data_dict


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(ds) for ds in self.datasets]
        self.len = sum(self.lengths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dataset_index = 0
        while dataset_index < len(self.datasets)-1 and index >= self.lengths[dataset_index]:
            index -= self.lengths[dataset_index]
            dataset_index += 1
        if index >= self.lengths[dataset_index]:
            raise IndexError
        return self.datasets[dataset_index].__getitem__(index)
