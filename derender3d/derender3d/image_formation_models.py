import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import networks, utils
from .utils import get_mask, IdentityLayer

EPS = 1e-7
GAMMA = 2.2


def to_gamma_space(img):
    return img ** GAMMA


def from_gamma_space(img):
    return img.clamp(min=EPS) ** (1 / GAMMA)


def compute_shadow(depth, light_dir, fov=10):
    start = time.time()
    depth = depth.detach()
    light_dir = light_dir.detach()
    image_size = depth.shape[-1]
    c = torch.linspace(-1, 1, image_size, device=depth.device) * math.tan(fov / 360 * math.pi)
    xx, yy = torch.meshgrid(c, c)
    xx = xx.view(1, 1, image_size, image_size).expand(depth.shape[0], -1, -1, -1)
    yy = yy.view(1, 1, image_size, image_size).expand(depth.shape[0], -1, -1, -1)
    positions = torch.cat([yy, xx, torch.ones_like(yy)], dim=1) * depth
    num_steps = image_size
    step = -light_dir.view(-1, 3, 1, 1) / image_size
    is_shadow = torch.zeros_like(depth, dtype=torch.bool)  # 0: unkown, 1: shadow
    for i in range(num_steps):
        positions += step
        pix_pos = (positions[:, :2] / positions[:, 2:]) / math.tan(fov / 360 * math.pi)
        sampled_depth = F.grid_sample(depth, pix_pos.permute(0, 2, 3, 1), padding_mode='border')
        diff = sampled_depth - positions[:, 2:, :, :]
        is_shadow = is_shadow | (diff < 0)
    end = time.time()
    # print('shadow took', end - start)
    return is_shadow


class ReconPhongIF(nn.Module):
    def __init__(self, model, fov=10, spec_alpha='single', spec_strength='single', spec_alpha_max=128, spec_strength_min=None, spec_taylor=False,
                 spec_softplus=False, detach_spec=False, neural_refinement=False, nr_spec=True, nr_spec_alphas=(1, 4, 16, 64),
                 nr_albedo=False, nr_detach=True, nr_nf=64, nr_depth=5, nr_shadow=True, shadow=False, shadow_cap=.1,
                 shadow_smooth=7, use_gamma_space=True, light_y_down=False, autoencoder_depth=5):
        super().__init__()
        self.model = model
        self.predict_specular_alpha = (spec_alpha == 'single') or (spec_alpha == 'map')
        if not self.predict_specular_alpha:
            self.spec_alpha = spec_alpha
        else:
            self.spec_alpha_mode = spec_alpha
        self.predict_specular_strength = (spec_strength == 'single') or (spec_strength == 'map') or (spec_strength == 'mask')
        if not self.predict_specular_strength:
            self.spec_strength = spec_strength
        else:
            self.spec_strength_mode = spec_strength
        self.spec_strength_min = spec_strength_min
        self.spec_taylor = spec_taylor
        self.spec_softplus = spec_softplus
        self.netA = networks.AutoEncoder(cin=3, cout=(4 if not self.predict_specular_strength else 5), nf=64, in_size=model.image_size, activation=IdentityLayer, depth=autoencoder_depth)
        self.netL = networks.Encoder(cin=3, cout=(5 if not spec_strength == 'single' and spec_strength != 'mask' else 6), nf=32, in_size=model.image_size, activation=IdentityLayer)
        self.view_d = self.get_view_d(fov, model.image_size)
        self.view_d.requires_grad_(False)
        self.detach_spec = detach_spec
        self.neural_refinement = neural_refinement
        self.nr_spec = nr_spec
        self.nr_spec_alphas = nr_spec_alphas
        self.nr_albedo = nr_albedo
        self.nr_detach = nr_detach
        self.nr_nf = nr_nf
        self.nr_depth = nr_depth if nr_depth is not None else 5
        self.nr_shadow = nr_shadow
        if not self.spec_softplus:
            self.spec_alpha_rescaler = lambda x: ((x * .5 + .5) * (math.sqrt(spec_alpha_max) - 1) + 1) ** 2
        else:
            self.spec_alpha_rescaler = lambda x: x * 16 + 1.
        self.shadow = shadow
        self.shadow_cap = shadow_cap
        self.shadow_smooth = shadow_smooth
        if self.shadow:
            self.smooth_kernel = utils.get_gaussian_kernel(self.shadow_smooth, channels=1)
        if self.neural_refinement:
            self.netNR = networks.AutoEncoder(cin=3, cout=1, nf=self.nr_nf, depth=self.nr_depth, in_size=model.image_size, activation=nn.Sigmoid, last_layer_relu=False)

        self.network_names = ('netA', 'netL', 'netNR')
        self.use_gamma_space = use_gamma_space
        self.light_y_down = light_y_down

        ## Freeze sub-networks
        for net in self.model.freeze_nets:
            if not net in self.network_names: continue
            for param in getattr(self, net).parameters():
                param.requires_grad = False

    def load_model_state(self, cp):
        cp = {k:v for k, v in cp.items() if not any(k.startswith(nln) for nln in self.model.not_load_nets)}
        self.load_state_dict(cp, strict=False)

    def get_view_d(self, fov, image_size):
        x = torch.linspace(-1, 1, image_size) * math.tan(fov / 360 * math.pi)
        y = torch.tensor(x)
        xx, yy = torch.meshgrid(x, y)
        v = torch.stack([yy, xx, torch.ones_like(xx)], dim=0)
        v = v / torch.norm(v, p=2, dim=0, keepdim=True)
        v = v.flip([1, 2])
        return v.unsqueeze(0)

    def forward(self, data_dict, light):
        input_im = data_dict['input_im']
        b, c, h, w = input_im.shape

        recon_normal = data_dict['recon_normal'][0]

        if self.view_d.device != recon_normal.device:
            self.view_d = self.view_d.to(recon_normal.device)
        view_d = self.view_d.expand(b, -1, -1, -1)

        if 'netA_out' not in data_dict:
            recon_albedo_specular_notanh = self.netA(input_im)[0]
            data_dict['netA_out'] = recon_albedo_specular_notanh
        else:
            recon_albedo_specular_notanh = data_dict['netA_out']
        recon_albedo_specular = torch.tanh(recon_albedo_specular_notanh)
        recon_albedo = recon_albedo_specular[:, :3, :, :]

        data_dict['recon_albedo'] = [recon_albedo]

        if self.use_gamma_space:
            recon_albedo = to_gamma_space(recon_albedo / 2 + 0.5)
        else:
            recon_albedo = recon_albedo / 2 + .5
        if 'netL_out' not in data_dict:
            netL_input = input_im if data_dict['target_im'] is None else data_dict['target_im']
            recon_light_notanh = self.netL(netL_input)
            data_dict['netL_out'] = recon_light_notanh
        else:
            recon_light_notanh = data_dict['netL_out']
        recon_light = torch.tanh(recon_light_notanh)

        if not self.predict_specular_alpha:
            spec_alpha = (torch.ones_like(recon_light[:, :1]) * self.spec_alpha).view(b, 1, 1, 1)
        else:
            if self.spec_alpha_mode == 'single':
                spec_alpha = recon_light_notanh[:, :1].view(b, 1, 1, 1)
            else:
                spec_alpha = recon_albedo_specular_notanh[:, 3:4, :, :]
            if not self.spec_softplus:
                spec_alpha = torch.tanh(spec_alpha)
            else:
                spec_alpha = F.softplus(spec_alpha)
            spec_alpha = self.spec_alpha_rescaler(spec_alpha)

        if not self.predict_specular_strength:
            spec_strength = recon_albedo_specular.new_tensor(self.spec_strength).view(1, 1, 1, 1).expand(b, 1, 1, 1)
        else:
            if self.spec_strength_mode == 'map':
                spec_strength = recon_albedo_specular[:, 4:] * .5 + .5
            elif self.spec_strength_mode == 'mask':
                spec_strength = ((recon_light[:, 5:6].view(b, 1, 1, 1) * .5 + .5) * .5)
                spec_mask = F.upsample(F.avg_pool2d(recon_albedo_specular[:, 4:] * .5 + .5, 4), scale_factor=(4, 4))
                spec_strength = spec_strength * spec_mask
            else:
                spec_strength = (recon_light[:, 5:6].view(b, 1, 1, 1) * .5 + .5) * .5

            if self.spec_strength_min is not None:
                spec_strength = spec_strength * 2 * (0.5 - self.spec_strength_min) + self.spec_strength_min
        # import ipdb; ipdb.set_trace()
        recon_light_a = recon_light[:, 1:2] / 2 + 0.5
        recon_light_b = recon_light[:, 2:3] / 2 + 0.5
        if not self.light_y_down:
            recon_light_d = torch.cat([recon_light[:, 3:5], torch.ones(b, 1).to(input_im.device)], 1)
        else:
            recon_light_d = torch.cat([recon_light[:, 3:4], -torch.ones(b, 1).to(input_im.device), recon_light[:, 4:5]], 1)

        if light is not None:
            if light.shape[-1] == 2:
                recon_light_d = torch.cat([light, torch.ones(b, 1).to(input_im.device)], 1)
            elif light.shape[-1] == 4:
                recon_light_a = light[:, :1].to(input_im.device)
                recon_light_b = light[:, 1:2].to(input_im.device)
                recon_light_d = torch.cat([light[:, 2:4], torch.ones(b, 1).to(input_im.device)], 1)
            else:
                recon_light_a = light[:, :1].to(input_im.device)
                recon_light_b = light[:, 1:2].to(input_im.device)
                recon_light_d = light[:, 2:5].to(input_im.device)
        recon_light_d = recon_light_d / torch.norm(recon_light_d, p=2, dim=1, keepdim=True)

        data_dict['recon_light'] = torch.cat([recon_light_a, recon_light_b, recon_light_d], dim=-1)
        data_dict['recon_light_spec_alpha'] = spec_alpha
        data_dict['recon_light_spec_strength'] = spec_strength

        cos_theta = (recon_normal * recon_light_d.view(-1, 3, 1, 1)).sum(1, keepdim=True)
        recon_diffuse_shading = cos_theta.clamp(min=0)

        specular_mask = get_mask(recon_diffuse_shading.shape, 5, recon_diffuse_shading.device)

        reflect_d = (2 * cos_theta * recon_normal - recon_light_d.view(b, 3, 1, 1))
        if self.detach_spec:
            reflect_d = reflect_d.detach()
        specular = (view_d * reflect_d).sum(1, keepdim=True).clamp(min=0) * (cos_theta > 0).to(torch.float32) * specular_mask
        if not self.spec_taylor:
            recon_specular_shading = specular.clamp(min=EPS, max=1-EPS).pow(spec_alpha)
        else:
            log_specular = specular.clamp(min=EPS, max=1-EPS).log()
            alpha_log_spec = spec_alpha * log_specular
            recon_specular_shading = (1 + alpha_log_spec + (alpha_log_spec ** 2) / 2 + (alpha_log_spec ** 3) / 6).clamp(0, 1)

        if not self.shadow:
            recon_shading = recon_light_a.view(-1, 1, 1, 1) + recon_light_b.view(-1, 1, 1, 1) * recon_diffuse_shading
            recon_im = recon_albedo * recon_shading + recon_specular_shading * spec_strength
        else:
            recon_shadow = compute_shadow(data_dict['recon_depth'][0], recon_light_d).to(torch.float32)
            if self.smooth_kernel.device != recon_shadow.device: self.smooth_kernel = self.smooth_kernel.to(recon_shadow.device)
            recon_shadow = F.conv2d(recon_shadow, self.smooth_kernel, padding=self.shadow_smooth//2)
            shading_cap_map = recon_shadow * self.shadow_cap + (1 - recon_shadow) * 4.
            shading_cap_map_spec = (1 - recon_shadow) * 4.
            recon_shading = recon_light_a.view(-1, 1, 1, 1) + recon_light_b.view(-1, 1, 1, 1) * torch.min(recon_diffuse_shading, shading_cap_map)
            recon_im = recon_albedo * recon_shading + torch.min(recon_specular_shading, shading_cap_map_spec) * spec_strength
            data_dict['recon_shadow'] = recon_shadow
        # import ipdb; ipdb.set_trace()
        if self.use_gamma_space:
            recon_im = from_gamma_space(recon_im)
        recon_im = torch.clamp(recon_im, 0, 1)
        recon_im = 2. * recon_im - 1.

        if self.neural_refinement:
            if not self.nr_albedo:
                neural_refinement_in = recon_im
            else:
                neural_refinement_in = recon_albedo
            if self.nr_detach:
                neural_refinement_in = neural_refinement_in.detach()
                recon_albedo_ = recon_albedo.detach()
            else:
                recon_albedo_ = recon_albedo

            neural_spec_mask = self.netNR(neural_refinement_in)[0]

            if not self.shadow:
                recon_im_nr = recon_albedo_ * recon_shading + recon_specular_shading * spec_strength * neural_spec_mask
            else:
                recon_im_nr = recon_albedo_ * recon_shading + torch.min(recon_specular_shading, shading_cap_map_spec) * spec_strength * neural_spec_mask
            recon_im_nr = from_gamma_space(recon_im_nr) * 2. - 1.

            data_dict['recon_im_nr'] = [recon_im_nr.clamp(-1, 1)]
            data_dict['neural_shading'] = [neural_spec_mask]
        # import ipdb; ipdb.set_trace()
        data_dict['recon_specular'] = [specular]
        data_dict['recon_specular_shading'] = [recon_specular_shading]
        data_dict['recon_diffuse_shading'] = [recon_diffuse_shading]
        data_dict['recon_im'] = [recon_im]

        return data_dict
