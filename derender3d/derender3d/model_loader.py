import os

from .loss import *
from .networks import *

EPS = 1e-7


class LoaderModel():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')

        self.network_names = [k for k in vars(self) if k.startswith('net')]

        self.metrics_module = cfgs.get('metrics_module', 'ImageMetrics')
        self.metrics_module = globals()[self.metrics_module](**{**cfgs.get('metrics_module_params', {}), **cfgs.get('metrics_params', {}), **{'model': self}})

        self.other_param_names = ['metrics_module']

        self.depth_folder = cfgs.get('depth_folder', None)
        self.normal_folder = cfgs.get('normal_folder', None)
        self.albedo_folder = cfgs.get('albedo_folder', None)

    def init_optimizers(self):
        self.optimizer_names = []

    def load_model_state(self, cp):
        pass

    def load_optimizer_state(self, cp):
        pass

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def set_progress(self, curr_epoch, total_epochs, sample=0, total_samples=1):
        pass

    def backward(self):
        pass

    def forward(self, data_dict, light=None, view=None):
        """Feedforward once."""
        input_im = data_dict['input_im'].to(self.device) * 2. - 1.
        if 'target_im' in data_dict:
            target_im = data_dict['target_im'].to(self.device) * 2. - 1.
        else:
            target_im = None

        data_dict = {'lr_' + k: v.to(self.device) for k, v in data_dict.items()}

        data_dict['input_im'] = input_im
        data_dict['target_im'] = target_im
        b, c, h, w = input_im.shape

        depths = []
        normals = []
        albedos = []

        indices = data_dict['lr_index']
        for index in indices:
            file_name = f'{index:06d}.png'

            if self.depth_folder is not None:
                depth = torch.tensor(utils.load_image(os.path.join(self.depth_folder, file_name)), device=input_im.device).permute(2, 0, 1)
                depths.append(depth)
            else:
                depths.append(torch.zeros_like(input_im[0, :1, :, :]))

            if self.normal_folder is not None:
                normal = torch.tensor(utils.load_image(os.path.join(self.normal_folder, file_name)), device=input_im.device).permute(2, 0, 1) * 2 - 1
                normals.append(normal)
            else:
                normals.append(torch.zeros_like(input_im[0, :, :, :]))

            if self.albedo_folder is not None:
                albedo = torch.tensor(utils.load_image(os.path.join(self.albedo_folder, file_name)), device=input_im.device).permute(2, 0, 1) * 2 - 1
                albedos.append(albedo)
            else:
                albedos.append(torch.zeros_like(input_im[0, :, :, :]))

        recon_depth = torch.stack(depths)
        recon_normal = torch.stack(normals)
        recon_albedo = torch.stack(albedos)

        data_dict['recon_depth'] = [recon_depth]
        data_dict['recon_normal'] = [recon_normal]
        data_dict['recon_albedo'] = [recon_albedo]

        metrics_dict = self.metrics_module(data_dict, target=('input_im' if target_im is None else 'target_im'))
        if 'recon_im_nr' in data_dict:
            data_dict_ = dict(data_dict)
            data_dict_['recon_im'] = data_dict['recon_im_nr']
            metrics_dict.update(self.metrics_module(data_dict_, suffix='nr', target=('input_im' if target_im is None else 'target_im')))

        self.data_dict = data_dict
        self.metrics_dict = metrics_dict

        return metrics_dict

    def visualize(self, logger, total_iter, max_bs=25, prefix='', numbers_only=False):
        pass

    def save_results(self, save_dir):
        pass
