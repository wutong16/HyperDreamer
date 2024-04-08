from .image_formation_models import *
from .loss import *
from .networks import *
from .renderer import Renderer
from .metrics import *

EPS = 1e-7


class Derender3D():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7 * self.max_depth + 0.3 * self.min_depth))
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.lam_flip = cfgs.get('lam_flip', (0.5, ))
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)

        self.compute_loss = cfgs.get('compute_loss', True)
        self.compute_metrics = cfgs.get('compute_metrics', True)

        self.freeze_nets = cfgs.get('freeze_nets', ())
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.autoencoder_depth = cfgs.get('autoencoder_depth', 5)

        self.predict_geometry = cfgs.get('predict_geometry', 'normal')
        self.use_gan = cfgs.get('use_gan', False)
        self.gan_params = cfgs.get('gan_params', {})
        self.patch_count = self.gan_params.get('patch_count', cfgs.get('batch_size', 16))
        self.patch_size = self.gan_params.get('patch_size', 64)
        self.gan_input = self.gan_params.get('gan_input', ['patch'])
        self.gan_nf = self.gan_params.get('gan_nf', 16)
        if self.use_gan:
            if 'full' in self.gan_input and not 'patch' in self.gan_input:
                self.netDisc = DiscNet(3, 1, nf=self.gan_nf, in_size=self.image_size)
            elif 'full' in self.gan_input and 'patch' in self.gan_input:
                self.netDisc = networks.DoubleDiscNet(3, 1, nf=self.gan_nf, im_size=self.image_size, patch_size=self.patch_size)
            else:
                self.netDisc = DiscNet(3, 1, nf=self.gan_nf, in_size=self.patch_size)
            self.criterionGAN = networks.GANLoss('lsgan')
        self.lam_GAN = self.gan_params.get('lam_GAN', [0.01, 0.01])
        self.lam_GAN_nr = self.gan_params.get('lam_GAN_nr', .0)
        self.gan_gen_disc_ratio = self.gan_params.get('gen_disc_ratio', 1.)
        self.gan_gen_total = 0
        self.gan_disc_total = 0
        self.gan_only_foreground = self.gan_params.get('only_foreground', False)
        self.light_sample_mode = self.gan_params.get('light_sample_mode', 'fixed')
        self.light_range = self.gan_params.get('light_range', .5)
        self.gan_rand_resize = self.gan_params.get('rand_resize', False)
        if (not isinstance(self.lam_GAN, list)) and (not isinstance(self.lam_GAN, tuple)):
            self.lam_GAN = (self.lam_GAN, self.lam_GAN)
        if (not isinstance(self.lam_GAN_nr, list)) and (not isinstance(self.lam_GAN_nr, tuple)):
            self.lam_GAN_nr = (self.lam_GAN_nr, self.lam_GAN_nr)

        if self.predict_geometry == 'depth' or self.predict_geometry == 'hr_depth':
            self.renderer = Renderer(cfgs)

        ## networks and optimizers
        self.netIF = ReconPhongIF(**{**cfgs.get('if_module_params', {}), **cfgs.get('if_params', {}), **{'model': self, 'autoencoder_depth': self.autoencoder_depth}})

        if self.predict_geometry == 'depth':
            self.netD = networks.AutoEncoder(cin=3, cout=1, nf=64, in_size=self.image_size, depth=self.autoencoder_depth)
        if self.predict_geometry == 'hr_depth':
            self.netD = networks.AutoEncoder(cin=3, cout=4, nf=64, in_size=self.image_size, depth=self.autoencoder_depth)
        if self.predict_geometry == 'normal':
            self.netN = networks.AutoEncoder(cin=3, cout=3, nf=64, in_size=self.image_size, depth=self.autoencoder_depth)

        self.network_names = [k for k in vars(self) if k.startswith('net')]

        self.make_optimizer = lambda model: torch.optim.Adam(
            model.parameters(),
            lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        self.loss_module = DecompositionLoss(**{**cfgs.get('loss_module_params', {}), **cfgs.get('loss_params', {}), **{'model': self}})
        self.metrics_module = cfgs.get('metrics_module', 'ImageMetrics')
        ## other parameters
        self.metrics_module = globals()[self.metrics_module](**{**cfgs.get('metrics_module_params', {}), **cfgs.get('metrics_params', {}), **{'model': self}})

        self.other_param_names = ['loss_module', 'metrics_module']
        if self.use_gan:
            self.other_param_names += ['criterionGAN']

        ## Freeze sub-networks
        for net in self.freeze_nets:
            if not net in self.network_names: continue
            for param in getattr(self, net).parameters():
                param.requires_grad = False

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d: (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                if k != 'netIF':
                    getattr(self, k).load_state_dict(cp[k], strict=True)
                else:
                    getattr(self, k).load_model_state(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])

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
        self.loss_module.set_progress(curr_epoch, total_epochs, sample, total_samples)

    def predict_shape(self, data_dict, input_im):
        if self.predict_geometry == 'depth':
            recon_depth = self.netD(input_im)[0]
            recon_depth = self.depth_rescaler(recon_depth)
            recon_normal = self.renderer.get_normal_from_depth(recon_depth.squeeze(1)).permute(0, 3, 1, 2)
            recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
        elif self.predict_geometry == 'hr_depth':
            recon_depth_bump = self.netD(input_im)[0]
            recon_depth = recon_depth_bump[:, :1, :, :]
            recon_bump = recon_depth_bump[:, 1:, :, :]
            recon_bump = recon_bump / torch.norm(recon_bump, p=2, dim=1, keepdim=True)
            data_dict['recon_bump'] = recon_bump
            recon_depth = self.depth_rescaler(recon_depth)
            recon_normal = self.renderer.get_normal_from_depth(recon_depth.squeeze(1)).permute(0, 3, 1, 2)
            data_dict['recon_normal_noref'] = recon_normal
            recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
            recon_normal = recon_normal + recon_bump
            recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
        elif self.predict_geometry == 'normal':
            recon_depth = input_im[:, :1, :, :] * 0
            recon_normal = self.netN(input_im)[0]
            recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
        elif self.predict_geometry == 'low_res':
            recon_depth = F.interpolate(data_dict['lr_recon_depth'], size=(self.image_size, self.image_size), mode='bilinear')
            recon_normal = F.interpolate(data_dict['lr_recon_normal'], size=(self.image_size, self.image_size), mode='bilinear')
            recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
        else:
            recon_depth = input_im[:, :1, :, :] * 0
            recon_normal = input_im * 0
        data_dict['recon_depth'] = [recon_depth]
        data_dict['recon_normal'] = [recon_normal]
        return data_dict

    def get_real_fake_patches(self, data_dict):
        bs, c, h, w = data_dict['input_im'].shape
        # input_im = data_dict['input_im']
        input_im = data_dict['recon_im'][0]
        recon_im_rand = data_dict['recon_im_rand'][0]
        recon_im_nr_rand = data_dict['recon_im_nr_rand'][0] if 'recon_im_nr_rand' in data_dict else None

        if self.gan_only_foreground:
            foreground_mask = (data_dict['lr_foreground_mask'] > .5).view(bs, 1, h, w).expand(-1, 3, -1, -1)
            input_im = torch.tensor(input_im)
            input_im[~foreground_mask] = 0
            recon_im_rand = torch.tensor(recon_im_rand)
            recon_im_rand[~foreground_mask] = 0
            if recon_im_nr_rand is not None:
                recon_im_nr_rand = torch.tensor(recon_im_nr_rand)
                recon_im_nr_rand[~foreground_mask] = 0

        if not self.gan_rand_resize:
            pc = min(self.patch_count, bs)
            ps = self.patch_size
            patch_coords = torch.randint(2, self.image_size - ps - 2, size=(2, self.patch_count, 2)).to(self.device)
            patches_real = torch.stack([input_im[i, :, patch_coords[0, i, 0]:patch_coords[0, i, 0]+ps, patch_coords[0, i, 1]:patch_coords[0, i, 1]+ps] for i in range(pc)], dim=0)
            patches_fake = torch.stack([recon_im_rand[i, :, patch_coords[1, i, 0]:patch_coords[1, i, 0]+ps, patch_coords[1, i, 1]:patch_coords[1, i, 1]+ps] for i in range(pc)], dim=0)
            data_dict['patches_real'] = patches_real
            data_dict['patches_fake'] = patches_fake
            if recon_im_nr_rand is not None:
                patches_fake_nr = torch.stack([recon_im_nr_rand[i, :, patch_coords[1, i, 0]:patch_coords[1, i, 0]+ps, patch_coords[1, i, 1]:patch_coords[1, i, 1]+ps] for i in range(pc)], dim=0)
                data_dict['patches_fake_nr'] = patches_fake_nr
        else:
            pc = min(self.patch_count, bs)
            num_patch = max(1, self.patch_count // pc)
            data_dict['patches_real'] = utils.get_patches(input_im[:pc], num_patch=num_patch, patch_size=self.patch_size).reshape(pc*num_patch, -1, self.patch_size, self.patch_size)
            data_dict['patches_fake'] = utils.get_patches(recon_im_rand[:pc], num_patch=num_patch, patch_size=self.patch_size).reshape(pc*num_patch, -1, self.patch_size, self.patch_size)
            if recon_im_nr_rand is not None:
                data_dict['patches_fake_nr'] = utils.get_patches(recon_im_nr_rand[:pc], num_patch=num_patch, patch_size=self.patch_size).reshape(pc*num_patch, -1, self.patch_size, self.patch_size)

        return data_dict

    def get_gan_input(self, data_dict):
        if 'patch' in self.gan_input:
            data_dict.update(self.get_real_fake_patches(data_dict))
            patches_real = self.data_dict['patches_real']
            patches_fake = self.data_dict['patches_fake']
            patches_fake_nr = self.data_dict['patches_fake_nr'] if 'patches_fake_nr' in self.data_dict else None

        if 'full' in self.gan_input and not 'patch' in self.gan_input:
            real_in = self.data_dict['recon_im'][0][:self.patch_count]
            fake_in = self.data_dict['recon_im_rand'][0][:self.patch_count]
            fake_in_nr = self.data_dict['recon_im_nr_rand'][0][
                         :self.patch_count] if 'recon_im_nr_rand' in self.data_dict else None
        elif 'full' in self.gan_input and 'patch' in self.gan_input:
            real_in = (self.data_dict['recon_im'][0][:self.patch_count], patches_real)
            fake_in = (self.data_dict['recon_im_rand'][0][:self.patch_count], patches_fake)
            fake_in_nr = (self.data_dict['recon_im_rand'][0][:self.patch_count],
                          patches_fake) if 'recon_im_nr_rand' in self.data_dict else None
        else:
            real_in = patches_real
            fake_in = patches_fake
            fake_in_nr = patches_fake_nr
        return real_in, fake_in, fake_in_nr

    def gan_losses(self, data_dict):
        real_in, fake_in, fake_in_nr = self.get_gan_input(data_dict)

        lam_GAN = (1 - self.loss_module._lam) * self.lam_GAN[0] + self.loss_module._lam * self.lam_GAN[1]
        lam_GAN_nr = (1 - self.loss_module._lam) * self.lam_GAN_nr[0] + self.loss_module._lam * self.lam_GAN_nr[1]

        GAN_G_pred = self.netDisc(fake_in)
        loss_G_GAN_fake = self.criterionGAN(GAN_G_pred, True)
        if fake_in_nr is not None:
            GAN_G_pred_fake_nr = self.netDisc(fake_in_nr)
            loss_G_GAN_fake_nr = self.criterionGAN(GAN_G_pred_fake_nr, True)
        else:
            loss_G_GAN_fake_nr = 0

        GAN_D_real_pred = self.netDisc(utils.detach(real_in))
        GAN_D_fake_pred = self.netDisc(utils.detach(fake_in))
        loss_D_GAN_real = self.criterionGAN(GAN_D_real_pred, True)
        loss_D_GAN_fake = self.criterionGAN(GAN_D_fake_pred, False)
        if fake_in_nr is not None:
            GAN_D_fake_pred_nr = self.netDisc(utils.detach(fake_in_nr))
            loss_D_GAN_fake_nr = self.criterionGAN(GAN_D_fake_pred_nr, True)
        else:
            GAN_D_fake_pred_nr = torch.zeros_like(GAN_D_fake_pred)
            loss_D_GAN_fake_nr = torch.zeros_like(loss_D_GAN_fake)

        loss_G_GAN = (1 - lam_GAN_nr) * loss_G_GAN_fake + lam_GAN_nr * loss_G_GAN_fake_nr

        real_acc = (GAN_D_real_pred > .5).to(dtype=torch.float32).mean()
        fake_acc = (GAN_D_fake_pred < .5).to(dtype=torch.float32).mean()
        fake_nr_acc = (GAN_D_fake_pred_nr < .5).to(dtype=torch.float32).mean()

        loss_D_total = (loss_D_GAN_real + loss_D_GAN_fake) * 0.5

        loss_dict = {}
        loss_dict['loss_GAN_G'] = loss_G_GAN * lam_GAN
        loss_dict['loss_GAN_G_fake'] = loss_G_GAN_fake
        loss_dict['loss_GAN_G_fake_nr'] = loss_G_GAN_fake_nr
        loss_dict['loss_GAN_D_real'] = loss_D_GAN_real
        loss_dict['loss_GAN_D_fake'] = loss_D_GAN_fake
        loss_dict['loss_GAN_D_total'] = loss_D_total
        loss_dict['loss_GAN_D_real_acc'] = real_acc
        loss_dict['loss_GAN_D_fake_acc'] = fake_acc
        if fake_in_nr is not None:
            loss_dict['loss_GAN_D_fake_nr'] = loss_D_GAN_fake_nr
            loss_dict['loss_GAN_D_fake_nr_acc'] = fake_nr_acc
        return loss_dict

    def backward(self):
        if self.use_gan:
            real_in, fake_in, fake_in_nr = self.get_gan_input(self.data_dict)

            lam_GAN = (1 - self.loss_module._lam) * self.lam_GAN[0] + self.loss_module._lam * self.lam_GAN[1]
            lam_GAN_nr = (1 - self.loss_module._lam) * self.lam_GAN_nr[0] + self.loss_module._lam * self.lam_GAN_nr[1]

            ## backward G
            self.netDisc.freeze()
            for optim_name in self.optimizer_names:
                if optim_name == 'optimizerDisc':
                    continue
                getattr(self, optim_name).zero_grad()
            GAN_G_pred = self.netDisc(fake_in)
            loss_G_GAN_fake = self.criterionGAN(GAN_G_pred, True)
            if fake_in_nr is not None:
                GAN_G_pred_fake_nr = self.netDisc(fake_in_nr)
                loss_G_GAN_fake_nr = self.criterionGAN(GAN_G_pred_fake_nr, True)
            else:
                loss_G_GAN_fake_nr = 0
            loss_G_GAN = (1 - lam_GAN_nr) * loss_G_GAN_fake + lam_GAN_nr * loss_G_GAN_fake_nr
            loss_G_total = self.loss_total + lam_GAN * loss_G_GAN

            loss_G_total = loss_G_total

            if self.gan_gen_total <= self.gan_disc_total * self.gan_gen_disc_ratio:
                self.gan_gen_total += 1
                loss_G_total.backward()
                for optim_name in self.optimizer_names:
                    if optim_name == 'optimizerDisc':
                        continue
                    getattr(self, optim_name).step()

            ## backward D
            self.netDisc.unfreeze()
            self.optimizerDisc.zero_grad()

            GAN_D_real_pred = self.netDisc(utils.detach(real_in))
            GAN_D_fake_pred = self.netDisc(utils.detach(fake_in))

            loss_D_GAN_real = self.criterionGAN(GAN_D_real_pred, True)
            loss_D_GAN_fake = self.criterionGAN(GAN_D_fake_pred, False)

            if fake_in_nr is not None:
                GAN_D_fake_pred_nr = self.netDisc(utils.detach(fake_in_nr))
                loss_D_GAN_fake_nr = self.criterionGAN(GAN_D_fake_pred_nr, False)
            else:
                loss_D_GAN_fake_nr = 0

            loss_D_total = (loss_D_GAN_real + ((1 - lam_GAN_nr) * loss_D_GAN_fake + lam_GAN_nr * loss_D_GAN_fake_nr)) * 0.5

            if self.gan_gen_total >= self.gan_disc_total * self.gan_gen_disc_ratio:
                self.gan_disc_total += 1
                loss_D_total.backward()
                self.optimizerDisc.step()

            self.loss_dict['loss_GAN_G'] = loss_G_GAN * lam_GAN
            self.loss_dict['loss_GAN_G_fake'] = loss_G_GAN_fake
            self.loss_dict['loss_GAN_G_fake_nr'] = loss_G_GAN_fake_nr
            self.loss_dict['loss_GAN_D_real'] = loss_D_GAN_real
            self.loss_dict['loss_GAN_D_fake'] = loss_D_GAN_fake
            self.loss_dict['loss_GAN_D_total'] = loss_D_total

        else:
            for optim_name in self.optimizer_names:
                getattr(self, optim_name).zero_grad()
            self.loss_total.backward()
            for optim_name in self.optimizer_names:
                getattr(self, optim_name).step()

    def forward(self, data_dict, light=None):
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
        # import ipdb; ipdb.set_trace()
        if ('recon_depth' not in data_dict) or ('recon_normal' not in data_dict):
            data_dict = self.predict_shape(data_dict, input_im)

        data_dict = self.netIF(data_dict, light)

        if self.use_gan:
            data_dict_ = dict(data_dict)
            if self.light_sample_mode == 'fixed':
                rand_light_a = .65 + (torch.randn((b, 1)) * .1).to(self.device)
                rand_light_b = .65 + (torch.randn((b, 1)) * .1).to(self.device)
                rand_light_d = (torch.randn((b, 2)) * self.light_range).to(self.device)
                if not self.netIF.light_y_down:
                    rand_light_d_norm = torch.cat([rand_light_d, torch.ones_like(rand_light_a)], dim=-1)
                else:
                    rand_light_d_norm = torch.cat([rand_light_d[:, :1], -torch.ones_like(rand_light_a), rand_light_d[:, 1:2]], dim=-1)
                rand_light_d_norm = rand_light_d_norm / torch.norm(rand_light_d_norm, keepdim=True, dim=-1)
                rand_light = torch.cat([rand_light_a, rand_light_b, rand_light_d_norm], dim=-1)
            elif self.light_sample_mode == 'mean':
                recon_light = data_dict['recon_light'].detach()
                rand_light_a = recon_light[:, 0].mean() + (torch.randn((b, 1)) * .1).to(self.device)
                rand_light_b = recon_light[:, 1].mean() + (torch.randn((b, 1)) * .1).to(self.device)
                rand_light_d = (torch.randn((b, 2)) * self.light_range).to(self.device)
                if not self.netIF.light_y_down:
                    rand_light_d_norm = torch.cat([rand_light_d, torch.ones_like(rand_light_a)], dim=-1)
                else:
                    rand_light_d_norm = torch.cat(
                        [rand_light_d[:, :1], -torch.ones_like(rand_light_a), rand_light_d[:, 1:2]], dim=-1)
                rand_light_d_norm = rand_light_d_norm / torch.norm(rand_light_d_norm, keepdim=True, dim=-1)
                rand_light = torch.cat([rand_light_a, rand_light_b, rand_light_d_norm], dim=-1)
            else:
                rand_light = data_dict['recon_light'][torch.randperm(b, dtype=torch.long)].detach()

            data_dict_ = self.netIF(data_dict_, rand_light)
            data_dict['recon_im_rand'] = data_dict_['recon_im']
            data_dict['recon_light_rand'] = rand_light
            data_dict['recon_diffuse_shading_rand'] = data_dict_['recon_diffuse_shading']
            data_dict['recon_specular_shading_rand'] = data_dict_['recon_specular_shading']
            if 'recon_im_nr' in data_dict_:
                data_dict['recon_im_nr_rand'] = data_dict_['recon_im_nr']
            if 'neural_shading' in data_dict_:
                data_dict['neural_shading_rand'] = data_dict_['neural_shading']
            if 'neural_specular' in data_dict_:
                data_dict['neural_specular_rand'] = data_dict_['neural_specular']
            if 'patch' in self.gan_input:
                data_dict = self.get_real_fake_patches(data_dict)

        data_dict['recon_im_mask'] = input_im.new_ones((b, 1, h, w))
        data_dict['conf_sigma_l1'] = input_im.new_ones((b, 1, h, w))
        data_dict['conf_sigma_percl'] = input_im.new_ones([b, 1, h // 4, w // 4])

        if self.compute_loss:
            self.loss_total, loss_dict = self.loss_module(data_dict)

        if self.compute_metrics:
            metrics_dict = self.metrics_module(data_dict, target=('input_im' if target_im is None else 'target_im'))
            if 'recon_im_nr' in data_dict:
                data_dict_ = dict(data_dict)
                data_dict_['recon_im'] = data_dict['recon_im_nr']
                metrics_dict.update(self.metrics_module(data_dict_, suffix='nr', target=('input_im' if target_im is None else 'target_im')))
        else:
            metrics_dict = None

        if self.compute_loss and self.compute_metrics:
            metrics_dict['loss'] = self.loss_total

        self.data_dict = data_dict
        if self.compute_loss:
            self.loss_dict = loss_dict
        self.metrics_dict = metrics_dict

        return metrics_dict

    def visualize(self, logger, total_iter, max_bs=25, prefix='', numbers_only=False):
        b, c, h, w = self.data_dict['input_im'].shape
        b0 = min(max_bs, b)

        print(self.loss_dict)

        def log_grid_image(label, im, nrow=int(math.ceil(b0 ** 0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)

        input_im = self.data_dict['input_im'][:b0].detach().cpu() / 2 + 0.5
        recon_im = self.data_dict['recon_im'][0][:b0].detach().cpu() / 2 + 0.5
        if 'recon_im_nr' in self.data_dict:
            recon_im_nr = self.data_dict['recon_im_nr'][0][:b0].detach().cpu() / 2 + 0.5
            recon_im_nr_rand = self.data_dict['recon_im_nr_rand'][0][:b0].detach().cpu() / 2 + .5 if 'recon_im_nr_rand' in self.data_dict else None
        else:
            recon_im_nr = None
            recon_im_nr_rand = None
        if 'neural_shading' in self.data_dict:
            neural_shading = self.data_dict['neural_shading'][0][:b0].detach().cpu().clamp(0, 4) * .25
            neural_shading_rand = self.data_dict['neural_shading_rand'][0][:b0].detach().cpu().clamp(0, 4) * .25 if 'neural_shading_rand' in self.data_dict else None
        else:
            neural_shading = None
            neural_shading_rand = None
        if 'neural_specular' in self.data_dict:
            neural_specular = self.data_dict['neural_specular'][0][:b0].detach().cpu().clamp(0, 1)
            neural_specular_rand = self.data_dict['neural_specular_rand'][0][:b0].detach().cpu().clamp(0, 1) if 'neural_specular_rand' in self.data_dict else None
        else:
            neural_specular = None
            neural_specular_rand = None
        if 'recon_diffuse_shading_rand' in self.data_dict:
            recon_diffuse_shading_rand = self.data_dict['recon_diffuse_shading_rand'][0][:b0].detach().cpu()
        else:
            recon_diffuse_shading_rand = None
        if 'recon_specular_shading_rand' in self.data_dict:
            recon_specular_shading_rand = self.data_dict['recon_specular_shading_rand'][0][:b0].detach().cpu()
        else:
            recon_specular_shading_rand = None

        if self.use_gan:
            b1 = min(max_bs, self.patch_count)
            recon_im_rand = self.data_dict['recon_im_rand'][0][:b0].detach().cpu() / 2 + 0.5
            patches_real = self.data_dict['patches_real'][:b1].detach().cpu() / 2 + .5 if 'patches_real' in self.data_dict else None
            patches_fake = self.data_dict['patches_fake'][:b1].detach().cpu() / 2 + .5 if 'patches_fake' in self.data_dict else None
            patches_fake_nr = self.data_dict['patches_fake_nr'][:b1].detach().cpu() / 2 + .5 if 'patches_fake_nr' in self.data_dict else None
            loss_dict_GAN = self.gan_losses(self.data_dict)
            self.loss_dict.update(loss_dict_GAN)
            # start = time.time()
            # gan_heatmap = utils.get_patch_heatmap(self.data_dict['recon_im_rand'][0], self.netDisc, self.patch_size)
            # end = time.time()
            # print('Heatmap took', end-start, 'seconds.')

        recon_albedo = self.data_dict['recon_albedo'][0][:b0].detach().cpu() / 2 + 0.5
        recon_depth = (self.data_dict['recon_depth'][0][:b0].detach().cpu() - self.min_depth) / (self.max_depth - self.min_depth)
        recon_normal = self.data_dict['recon_normal'][0][:b0].detach().cpu() / 2 + 0.5
        recon_diffuse_shading = self.data_dict['recon_diffuse_shading'][0][:b0].detach().cpu()
        recon_specular = self.data_dict['recon_specular'][0][:b0].detach().cpu()
        recon_specular_shading = self.data_dict['recon_specular_shading'][0][:b0].detach().cpu()

        recon_light = self.data_dict['recon_light'][:b0].detach().cpu().numpy()
        recon_light_spec_alpha = self.data_dict['recon_light_spec_alpha'][:b0].detach().cpu()
        recon_light_spec_strength = self.data_dict['recon_light_spec_strength'][:b0].detach().cpu()
        recon_light_spec_alpha_mean = recon_light_spec_alpha.mean(dim=[-2, -1])
        recon_light_spec_strength_mean = recon_light_spec_strength.mean(dim=[-2, -1])
        recon_light_a = recon_light[:, :1]
        recon_light_b = recon_light[:, 1:2]
        recon_light_d_x = recon_light[:, 2:3]
        recon_light_d_y = recon_light[:, 3:4]

         #lr_recon_diffuse_shading = self.data_dict['lr_recon_diffuse_shading'][:b0].detach().cpu() / 2 + 0.5
        lr_recon_diffuse_shading = torch.zeros_like(recon_diffuse_shading)
        lr_recon_depth = (self.data_dict['lr_recon_depth'][:b0].detach().cpu() - self.min_depth) / (self.max_depth - self.min_depth)
        lr_recon_normal = self.data_dict['lr_recon_normal'][:b0].detach().cpu() / 2 + 0.5
        lr_recon_albedo = self.data_dict['lr_recon_albedo'][:b0].detach().cpu() / 2 + 0.5
        lr_recon_im_mask = self.data_dict['lr_recon_im_mask'][:b0].detach().cpu()

        lr_recon_light = self.data_dict['lr_canon_light'][:b0].detach().cpu().numpy()
        lr_recon_light_a = lr_recon_light[:, :1]
        lr_recon_light_b = lr_recon_light[:, 1:2]

        log_grid_image(f'Image{prefix}/input_im', input_im)
        log_grid_image(f'Image{prefix}/recon_im', recon_im)
        if recon_im_nr is not None: log_grid_image(f'Image{prefix}/recon_im_nr', recon_im_nr)
        if self.use_gan:
            log_grid_image(f'Image{prefix}/recon_im_rand', recon_im_rand)
            if patches_real is not None: log_grid_image(f'Image{prefix}/patches_real', patches_real)
            if patches_fake is not None: log_grid_image(f'Image{prefix}/patches_fake', patches_fake)
            # log_grid_image(f'Image{prefix}/gan_heatmap', gan_heatmap)
            if recon_im_nr_rand is not None: log_grid_image(f'Image{prefix}/recon_im_nr_rand', recon_im_nr_rand)
            if neural_shading_rand is not None: log_grid_image(f'Depth{prefix}/neural_shading_rand', neural_shading_rand)
            if patches_fake_nr is not None: log_grid_image(f'Image{prefix}/patches_fake_nr', patches_fake_nr)

        log_grid_image(f'Image{prefix}/recon_albedo', recon_albedo)
        log_grid_image(f'Image{prefix}/lr_recon_albedo', lr_recon_albedo)
        if recon_light_spec_alpha.shape[-1] > 1: log_grid_image(f'Image{prefix}/recon_light_spec_alpha', recon_light_spec_alpha.clamp(0, 16) / 16.)
        if recon_light_spec_strength.shape[-1] > 1: log_grid_image(f'Image{prefix}/recon_light_spec_strength', recon_light_spec_strength)

        log_grid_image(f'Depth{prefix}/recon_im_mask', lr_recon_im_mask)
        log_grid_image(f'Depth{prefix}/recon_depth', recon_depth)
        log_grid_image(f'Depth{prefix}/recon_normal', recon_normal)
        log_grid_image(f'Depth{prefix}/recon_diffuse_shading', torch.clamp(recon_diffuse_shading, 0, 1))
        log_grid_image(f'Depth{prefix}/recon_specular', torch.clamp(recon_specular, 0, 1))
        log_grid_image(f'Depth{prefix}/recon_specular_shading', torch.clamp(recon_specular_shading, 0, 1))

        if recon_diffuse_shading_rand is not None: log_grid_image(f'Depth{prefix}/recon_diffuse_shading_rand', torch.clamp(recon_diffuse_shading_rand, 0, 1))
        if recon_specular_shading_rand is not None: log_grid_image(f'Depth{prefix}/recon_specular_shading_rand', torch.clamp(recon_specular_shading_rand, 0, 1))

        if neural_shading is not None: log_grid_image(f'Depth{prefix}/neural_shading', neural_shading)
        if neural_specular is not None: log_grid_image(f'Depth{prefix}/neural_specular', neural_specular)
        if neural_specular_rand is not None: log_grid_image(f'Depth{prefix}/neural_specular_rand', neural_specular_rand)
        log_grid_image(f'Depth{prefix}/lr_recon_depth', lr_recon_depth)
        log_grid_image(f'Depth{prefix}/lr_recon_normal', lr_recon_normal)
        log_grid_image(f'Depth{prefix}/lr_recon_diffuse_shading', torch.clamp(lr_recon_diffuse_shading, 0, 1))

        logger.add_scalar(f'Loss{prefix}/loss_total', self.loss_total, total_iter)
        for k, v in self.loss_dict.items():
            logger.add_scalar(f'Loss{prefix}/{k}', v, total_iter)

        for k, v in self.metrics_dict.items():
            logger.add_scalar(f'Metrics{prefix}/{k}', v, total_iter)

        logger.add_histogram(f'Light{prefix}/recon_light_a', recon_light_a, total_iter)
        logger.add_histogram(f'Light{prefix}/recon_light_b', recon_light_b, total_iter)
        logger.add_histogram(f'Light{prefix}/recon_light_d_x', recon_light_d_x, total_iter)
        logger.add_histogram(f'Light{prefix}/recon_light_d_y', recon_light_d_y, total_iter)
        logger.add_histogram(f'Light{prefix}/recon_light_spec_alpha', recon_light_spec_alpha_mean, total_iter)
        logger.add_histogram(f'Light{prefix}/recon_light_spec_strength', recon_light_spec_strength_mean, total_iter)

        logger.add_histogram(f'Light{prefix}/lr_recon_light_a', lr_recon_light_a, total_iter)
        logger.add_histogram(f'Light{prefix}/lr_recon_light_b', lr_recon_light_b, total_iter)

    def save_results(self, save_dir):
        raise NotImplementedError
