import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from lib.sg_render import render_with_sg, compute_envmap
import imageio

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class LightSGs(nn.Module):
    def __init__(self, num_lgt_sgs=32, white_light=True, upper_hemi=False):
        super().__init__()
        self.numLgtSGs = num_lgt_sgs
        print('Number of Light SG: ', self.numLgtSGs)
        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.white_light = white_light
        if self.white_light:
            print('Using white light!')
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5), requires_grad=True).cuda()  # [M, 5]; lobe + lambda + mu
            # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
        else:
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True).cuda()  # [M, 7]; lobe + lambda + mu
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

        def compute_energy(lgtSGs):
            lgtLambda = torch.abs(lgtSGs[:, 3:4])  # [M, 1]
            lgtMu = torch.abs(lgtSGs[:, 4:])  # [M, 3]
            energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
            return energy

        ### uniformly distribute points on a sphere
        def fibonacci_sphere(samples=1):
            '''
            https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
            :param samples:
            :return:
            '''
            points = []
            phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
            for i in range(samples):
                y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment

                x = np.cos(theta) * radius
                z = np.sin(theta) * radius

                points.append([x, y, z])
            points = np.array(points)
            return points

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 20. + torch.abs(self.lgtSGs.data[:, 3:4] * 100.)
        # make sure total energy is around 1.
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
        self.lgtSGs.data[:, 4:] = torch.abs( self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

        self.lgtSGs = nn.Parameter(self.lgtSGs.data, requires_grad=True).cuda()

    def forward(self):
        return self.lgtSGs

    def load_light(self, envmap):
        device = self.lgtSGs.data.device
        envmap = np.load(envmap)
        self.lgtSGs = nn.Parameter(torch.from_numpy(envmap).to(device), requires_grad=True)
        self.numLgtSGs = self.lgtSGs.data.shape[0]
        if self.lgtSGs.data.shape[1] == 7:
            self.white_light = False

    def get_light(self):
        lgt = self.lgtSGs.detach().clone()
        if self.white_light:
            lgt = torch.cat((lgt, lgt[..., -1:], lgt[..., -1:]), dim=-1)
        if self.upper_hemi:
            lgt = self.restrict_lobes_upper(lgt)
        return lgt

class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        if opt.normal_offset:
            self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        if opt.dmtet:
            self.dmtet_decoder = MLP(self.in_dim, 4, 32, 3, False)
        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x, predict_class=False, refine_branch=False):

        # sigma
        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        if refine_branch:
            h = self.sigma_net_plus(enc)
        else:
            h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])
        res = [sigma, albedo]

        if predict_class:
            pred_class = self.predict_class(enc)
            res.append(pred_class)
        return res

    def svbrdf_forward(self, x, predict_class=False, **kwargs):

        enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        roughness = self.mlp_roughness(enc)
        roughness = (torch.tanh(roughness) + 1) / 2
        # print(self.mlp_specular.net[0].weight.var())
        specular_reflectance = self.mlp_specular(enc)
        specular_reflectance = (torch.tanh(specular_reflectance) + 1) / 2

        lgtSGs = self.lgtSGs() # .lgtSGs
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)
        # print('roughness', self.mlp_roughness.net[0].weight.var())
        # print('specular', self.mlp_specular.net[0].weight.var())
        # print('lgtSGs', self.lgtSGs.lgtSGs.var())
        # roughness[:,:] = 0
        # specular_reflectacne[:,:]= 0.4
        ret = dict([
            ('sigma', sigma),
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectance),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', albedo)
        ])
        if predict_class:
            pred_class = self.predict_class(enc)
            ret.update(dict(pred_class=pred_class))
        if self.opt.normal_offset:
            delta_normal = safe_normalize(self.normal_net(enc))
            ret.update(dict(delta_normal=delta_normal))
        return ret


    def prepare_class_predictor(self, num_classes, detach=True):
        self.num_classes = num_classes
        self.class_net = MLP(self.in_dim, num_classes, self.hidden_dim, self.num_layers, bias=True).cuda()
        if detach:
            self.predict_class = lambda x: self.class_net(x.detach())
        else:
            self.predict_class = lambda x: self.class_net(x)

    def prepare_svbrdf(self, num_lgt_sgs=32, white_light=True, upper_hemi=False):

        self.mlp_roughness = MLP(self.in_dim, 1, self.hidden_dim, num_layers=2, bias=True).cuda()
        self.mlp_specular = MLP(self.in_dim, 1, self.hidden_dim, num_layers=2, bias=True).cuda()
        self.lgtSGs = LightSGs(num_lgt_sgs, white_light, upper_hemi)
        self.white_light = white_light
        self.upper_hemi = upper_hemi

    def init_r_s(self, init_roughness=0.5, init_specular=0.23):
        # init roughness
        init_roughness= torch.tensor(init_roughness)
        init_roughness= torch.arctanh(init_roughness*2-1)
        r = self.mlp_roughness.net[-1]
        torch.nn.init.constant_(r.weight, 0)
        torch.nn.init.constant_(r.bias, init_roughness)
        # init specular
        init_specular= torch.tensor(init_specular)
        init_specular= torch.arctanh(init_specular*2-1)
        s = self.mlp_specular.net[-1]
        torch.nn.init.constant_(s.weight, 0)
        torch.nn.init.constant_(s.bias, init_specular)

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal

    def forward(self, x, d, l=None, ratio=1, shading='albedo', predict_class=False, refine_branch=False):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        if not shading == 'svbrdf':
            outputs = self.common_forward(x, predict_class, refine_branch)
            sigma, albedo = outputs[:2]

            if shading == 'albedo':
                normal = None
                color = albedo

            else: # lambertian shading

                # normal = self.normal_net(enc)
                normal = self.normal(x)

                lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
                elif shading == 'normal':
                    color = (normal + 1) / 2
                else: # 'lambertian'
                    color = albedo * lambertian.unsqueeze(-1)

            res = [sigma, color, normal]

            if predict_class:
                pred_class = outputs[-1]
                res.append(pred_class)
        else:
            sg_envmap_material = self.svbrdf_forward(x, predict_class, refine_branch)

            sigma = sg_envmap_material['sigma']
            normal = self.normal(x)
            if self.opt.normal_offset:
                normal = safe_normalize(normal + sg_envmap_material['delta_normal']) # fixme: should calcualte this in sph space!

            if predict_class:
                pred_class = sg_envmap_material['pred_class']

            sg_ret = render_with_sg(
                lgtSGs=sg_envmap_material['sg_lgtSGs'],
                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                roughness=sg_envmap_material['sg_roughness'],
                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                normal=normal, viewdirs=d,)

            color = sg_ret['sg_rgb']
            res = [sigma, color, normal]

            if predict_class:
                res.append(pred_class)

        return res

    def init_tet(self):
        if self.cuda_ray:
            density_thresh = min(self.mean_density, self.density_thresh)
        else:
            density_thresh = self.density_thresh
        if self.opt.density_activation == 'softplus':
            density_thresh = density_thresh * 25
        with torch.no_grad():
            sigma = self.density(self.verts)['sigma']
            mask = sigma > density_thresh
            valid_verts = self.verts[mask]
            self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
            self.verts = self.verts * self.tet_scale

            ref_sdf = self.density(self.verts)['sigma']
            ref_sdf = (ref_sdf - density_thresh).clamp(-1, 1)

        loss_fn = nn.MSELoss()
        params = [
            {'params': self.encoder.parameters()},
            {'params': self.dmtet_decoder.parameters()}
        ]
        optimizer = torch.optim.Adam(params, lr=1e-3, betas=[0.9,0.99],eps=1e-15)
        batch_size = 10240
        for _ in range(4000):
            rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
            p = self.verts[rand_idx]
            ref_value = ref_sdf[rand_idx]
            output = self.dmtet_decoder(self.encoder(p))
            loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def density(self, x, predict_class=False, refine_branch=False):
        # x: [N, 3], in [-bound, bound]

        if predict_class:
            sigma, albedo, pred_class = self.common_forward(x, predict_class=True, refine_branch=refine_branch)
        else:
            sigma, albedo = self.common_forward(x, refine_branch=refine_branch)
            pred_class = None
        return {
            'sigma': sigma,
            'albedo': albedo,
            'pred_class': pred_class
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def load_envmap(self, envmap):
        self.lgtSGs.load_light(envmap)
        self.white_light = self.lgtSGs.white_light

    @torch.no_grad()
    def export_envmap(self, savepath, H=256, W=512):
        lgt = self.lgtSGs.get_light()
        envmap = compute_envmap(lgt, H, W, self.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(savepath, envmap)

    # optimizer utils
    def get_params(self, lr):

        encoder_lr = lr
        params = [
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]

        if hasattr(self, 'class_net'):
            params.append({'params': self.class_net.parameters(), 'lr': lr})
        if hasattr(self, 'sigma_net_plus'):
            params.append({'params': self.sigma_net_plus.parameters(), 'lr': lr})

        if self.opt.use_svbrdf:
            params.append({'params': self.mlp_roughness.parameters(), 'lr': lr})
            params.append({'params': self.mlp_specular.parameters(), 'lr': lr})
            params.append({'params': self.lgtSGs.parameters(), 'lr': lr})

        if hasattr(self, 'RegionMaterials'):
            params.append({'params': self.RegionMaterials.parameters(), 'lr': lr * 0.01})
        if hasattr(self, 'normal_net'):
            params.append({'params': self.normal_net.parameters(), 'lr': lr})

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        if self.opt.dmtet and not self.opt.lock_geo:
            params.append({'params': self.dmtet_decoder.parameters(), 'lr': lr})

        return params
