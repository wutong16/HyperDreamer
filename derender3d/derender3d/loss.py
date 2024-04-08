from functools import partial

import torch
from pytorch_msssim import ssim
from torch import nn
from torch.nn import functional as F

from ..derender3d import networks, utils
from ..derender3d.renderer.utils import get_rotation_matrix
from ..derender3d.utils import mask_mean

EPS = 1e-7


def masked_l1_loss(arr1, arr2, mask):
    return mask_mean((arr1 - arr2).abs(), ~mask.expand(-1, arr1.shape[1], -1, -1))


def masked_dot_loss(arr1, arr2, mask):
    return mask_mean(1 - (arr1 * arr2).sum(1, keepdims=True), ~mask.expand(-1, 1, -1, -1))


def l1_loss(im1, im2, mask=None):
    loss = (im1 - im2).abs()
    if mask is not None:
        loss = utils.mask_mean(loss, mask)
    else:
        loss = loss.mean()
    return loss


def l1_ssim_loss(im1, im2, mask=None, lam=.5):
    l1_loss = (im1 - im2).abs()
    if mask is not None:
        l1_loss = utils.mask_mean(l1_loss, mask)
    else:
        l1_loss = l1_loss.mean()
    ssim_loss = ((1. - ssim(im1 * .5 + .5, im2 * .5 + .5, data_range=1., size_average=False, nonnegative_ssim=True)) * .5).mean()
    return (1-lam) * l1_loss + lam * ssim_loss


def l1_perc_loss(im1, im2, mask=None, perc_fn=None):
    l1_loss = (im1 - im2).abs()
    if mask is not None:
        l1_loss = utils.mask_mean(l1_loss, mask)
    else:
        l1_loss = l1_loss.mean()
    perc_loss = perc_fn(im1, im2)
    return .5 * l1_loss + .5 * perc_loss


def light_loss(l0, l1, lam=.8):
    ab = F.mse_loss(l0[:, :2], l1[:, :2])
    xyz = (1 - (l0[:, 2:] * l1[:, 2:]).sum(1)).mean()
    return (1 - lam) * ab + lam * xyz, ab, xyz


def albedo_smooth_loss(alb, im, w_l1=1, w_grad_rec=0):
    alb_grad_y = alb[:, :, 1:, :] - alb[:, :, :-1, :]
    alb_grad_x = alb[:, :, :, 1:] - alb[:, :, :, :-1]

    im_grad_y = im[:, :, 1:, :] - im[:, :, :-1, :]
    im_grad_x = im[:, :, :, 1:] - im[:, :, :, :-1]

    l1_loss = alb_grad_y.abs().mean() + alb_grad_x.abs().mean()

    grad_rec_loss = (alb_grad_y - im_grad_y).pow(2).mean() + (alb_grad_x - im_grad_x).pow(2).mean()
    loss = w_l1 * l1_loss + w_grad_rec * grad_rec_loss
    return loss


class Scaler:
    def __init__(self, size):
        self.size = size

    def scale(self, arr, normalize=False):
        if arr.shape[2] != self.size[0] or arr.shape[3] != self.size[1]:
            arr = F.interpolate(arr, self.size, mode='bilinear')
            if normalize:
                arr = arr / ((arr ** 2).sum(1, keepdims=True) ** .5)
        return arr


def get_range(r):
    if isinstance(r, tuple) or isinstance(r, list):
        return tuple(r)
    else:
        return (r, r)


class DecompositionLoss(nn.Module):
    def __init__(self, model, coeff_rec=1., coeff_depth=.25, coeff_normal=1., coeff_diffuse_shading=0., coeff_light=1,
                 coeff_albedo=1., coeff_rec_nr=0, coeff_albedo_smooth=0.0, use_dot_loss=True, downsample=False, blur_albedo=False, rec_loss='l1',
                 nr_rec_loss='l1', only_foreground=False):
        super().__init__()
        self.model = model
        self.coeffs = {
            'rec': coeff_rec,
            'rec_nr': coeff_rec_nr,
            'depth': coeff_depth,
            'normal': coeff_normal,
            'diffuse_shading': coeff_diffuse_shading,
            'light': coeff_light,
            'albedo': coeff_albedo,
            'albedo_smooth': coeff_albedo_smooth
        }
        self.coeffs = {k: get_range(v) for k, v in self.coeffs.items()}
        self.normal_loss = masked_l1_loss if not use_dot_loss else masked_dot_loss
        self._lam = 0
        if not downsample:
            self.scaler = Scaler((model.image_size, model.image_size))
        else:
            self.scaler = Scaler((64, 64))

        self.blur_albedo = blur_albedo
        if self.blur_albedo:
            self._gaussian_kernel = utils.get_gaussian_kernel(kernel_size=5, channels=1)

        self.perc_loss = None
        self.nr_loss_fn = self.get_loss_fn(nr_rec_loss)
        self.rec_loss = self.get_loss_fn(rec_loss)
        # self.smooth_loss = lambda im: (im[:, :, 1:, :] - im[:, :, :-1, :]).abs().mean() + (im[:, :, :, 1:] - im[:, :, :, :-1]).abs().mean()
        self.smooth_loss = albedo_smooth_loss

        self.only_foreground = only_foreground


    def get_loss_fn(self, fn='l1'):
        if fn == 'l1':
            return l1_loss
        elif fn == 'l1_ssim':
            return l1_ssim_loss
        elif fn == 'l1_perc':
            if self.perc_loss is not None:
                self.perc_loss = networks.PerceptualLoss(requires_grad=False)
            return partial(l1_perc_loss, perc_fn=self.perc_loss)

    def set_progress(self, curr_epoch, total_epochs, sample=0, total_samples=1):
        self._lam = (curr_epoch + (sample / total_samples)) / total_epochs
        pass

    def forward(self, data_dict):
        input_im = data_dict['input_im']
        recon_im = data_dict['recon_im'][0]
        if self.coeffs['rec_nr'][1] != 0:
            recon_im_nr = data_dict['recon_im_nr'][0]

        recon_albedo = self.scaler.scale(data_dict['recon_albedo'][0])
        if self.blur_albedo:
            self._gaussian_kernel = self._gaussian_kernel.to(recon_albedo.device)
            recon_albedo = torch.cat([F.conv2d(recon_albedo[:, :1], self._gaussian_kernel, padding=2), F.conv2d(recon_albedo[:, 1:2], self._gaussian_kernel, padding=2), F.conv2d(recon_albedo[:, 2:], self._gaussian_kernel, padding=2)], dim=1)
        recon_depth = (self.scaler.scale(data_dict['recon_depth'][0]) - self.model.min_depth) / (self.model.max_depth - self.model.min_depth) * 2. - 1.
        recon_normal = self.scaler.scale(data_dict['recon_normal'][0], normalize=True)
        recon_light = data_dict['recon_light']
        recon_diffuse_shading = self.scaler.scale(data_dict['recon_diffuse_shading'][0])

        lr_recon_albedo = self.scaler.scale(data_dict['lr_recon_albedo'])
        lr_recon_depth = (self.scaler.scale(data_dict['lr_recon_depth']) - self.model.min_depth) / (self.model.max_depth - self.model.min_depth) * 2. - 1.
        lr_recon_normal = self.scaler.scale(data_dict['lr_recon_normal'], normalize=True)
        lr_canon_light = data_dict['lr_canon_light']
        # lr_recon_diffuse_shading = self.scaler.scale(data_dict['lr_recon_diffuse_shading'])
        lr_recon_diffuse_shading = None

        lr_view = data_dict['lr_view']
        if len(lr_view.shape) != 3:
            rot_mat = get_rotation_matrix(lr_view[:, 0], lr_view[:, 1], lr_view[:, 3])
            lr_recon_light = torch.cat([lr_canon_light[:, :2], (rot_mat @ lr_canon_light[:, 2:].unsqueeze(-1)).squeeze(-1)], dim=-1)
        else:
            lr_recon_light = lr_canon_light

        recon_im_mask_ = data_dict['lr_recon_im_mask']
        if recon_im_mask_.shape[-1] != input_im.shape[-1]:
            recon_im_mask_[:, :, :, [0, 63]] = 0
            recon_im_mask_[:, :, [0, 63], :] = 0
            recon_im_mask = (F.upsample(recon_im_mask_, recon_normal.shape[-2:], mode='bilinear') == 1.)
            recon_im_mask = (utils.shrink_mask(recon_im_mask, 5) > .5)
        else:
            recon_im_mask = recon_im_mask_ > .5

        coeffs = {k: (1 - self._lam) * v[0] + self._lam * v[1] for k, v in self.coeffs.items()}

        border_mask = (utils.get_mask(recon_im_mask.shape, device=recon_im_mask.device) > .5)
        if self.only_foreground:
            foreground_mask = data_dict['lr_foreground_mask'] > .5
            border_mask = border_mask & foreground_mask
        border_mask = (~border_mask).expand(-1, 3, -1, -1)

        reconstruction_loss = self.rec_loss(input_im, recon_im, border_mask) * coeffs['rec']
        albedo_loss = mask_mean((recon_albedo - lr_recon_albedo).abs(), ~recon_im_mask.expand(-1, 3, -1, -1)) * coeffs['albedo'] if coeffs['albedo'] != 0 else reconstruction_loss.new_zeros((1))
        depth_loss = masked_l1_loss(recon_depth, lr_recon_depth, recon_im_mask) * coeffs['depth'] if coeffs['depth'] != 0 else reconstruction_loss.new_zeros((1))
        normal_loss = self.normal_loss(recon_normal, lr_recon_normal, recon_im_mask) * coeffs['normal'] if coeffs['normal'] != 0 else reconstruction_loss.new_zeros((1))
        light_loss = F.mse_loss(recon_light, lr_recon_light) * coeffs['light'] if coeffs['light'] != 0 else reconstruction_loss.new_zeros((1))
        diffuse_shading_loss = masked_l1_loss(recon_diffuse_shading, lr_recon_diffuse_shading, recon_im_mask) * coeffs['diffuse_shading'] if coeffs['diffuse_shading'] != 0 else reconstruction_loss.new_zeros((1))
        rec_nr_loss = self.nr_loss_fn(input_im, recon_im_nr, border_mask) * coeffs['rec_nr'] if coeffs['rec_nr'] != 0 else reconstruction_loss.new_zeros((1))
        albedo_smooth_loss = self.smooth_loss(recon_albedo, input_im) * coeffs['albedo_smooth'] if coeffs['albedo_smooth'] != 0 else reconstruction_loss.new_zeros((1))

        loss_total = reconstruction_loss + albedo_loss + depth_loss + normal_loss + light_loss + diffuse_shading_loss + rec_nr_loss + albedo_smooth_loss

        loss_dict = {
            'reconstruction_loss': reconstruction_loss,
            'albedo_loss': albedo_loss,
            'depth_loss': depth_loss,
            'normal_loss': normal_loss,
            'light_loss': light_loss,
            'diffuse_shading_loss': diffuse_shading_loss,
            'rec_nr_loss': rec_nr_loss,
            'albedo_smooth': albedo_smooth_loss
        }

        return loss_total, loss_dict


class ReconstructionLoss(nn.Module):
    def __init__(self, model, lam_perc=1.) -> None:
        super().__init__()
        self.model = model
        self.lam_perc = lam_perc
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1 - im2).abs()
        if conf_sigma is not None:
            loss = loss * 2 ** 0.5 / (conf_sigma + EPS) + (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def forward(self, data_dict, suffix=None):
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        input_im = data_dict['input_im']
        recon_im = data_dict['recon_im'][0]
        recon_im_mask = data_dict['recon_im_mask']

        conf_sigma_l1 = data_dict['conf_sigma_l1']
        conf_sigma_percl = data_dict['conf_sigma_percl']

        b, c, h, w = input_im.shape
        loss_l1_im = self.photometric_loss(recon_im, input_im, mask=recon_im_mask, conf_sigma=conf_sigma_l1)
        loss_perc_im = self.PerceptualLoss(recon_im, input_im, mask=recon_im_mask, conf_sigma=conf_sigma_percl)
        loss_total = loss_l1_im + self.lam_perc * (loss_perc_im)
        loss_dict = {
            f'loss_l1_im{suffix}': loss_l1_im,
            f'loss_perc_im{suffix}': loss_perc_im,
            f'loss_total{suffix}': loss_total
        }
        return loss_total, loss_dict
