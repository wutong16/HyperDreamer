import math

import torch
from lib.lpips.LPIPS import LPIPS
from pytorch_msssim import SSIM
from torch import nn

from ..derender3d.utils import PSNR, mask_mean


class ImageMetrics(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

        self.psnr = PSNR()
        self.ssim = SSIM(size_average=False, data_range=1.0)
        self.lpips = LPIPS()

    def forward(self, data_dict, suffix=None, rec='recon_im', target='input_im'):
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        metrics_dict = {}

        recon_im = data_dict[rec][0]
        target_im = data_dict[target]

        recon_im = recon_im / 2. + .5
        target_im = target_im / 2. + .5
        with torch.no_grad():
            metric_psnr = self.psnr(recon_im * 255., target_im * 255.).mean()
            metric_ssim = self.ssim(recon_im, target_im).mean()
            metric_lpips = self.lpips(recon_im, target_im, normalize=True).mean()
            metric_l1 = (target_im - recon_im).abs().mean()

        metrics_dict[f"PSNR{suffix}"] = metric_psnr
        metrics_dict[f"SSIM{suffix}"] = metric_ssim
        metrics_dict[f"LPIPS{suffix}"] = metric_lpips
        metrics_dict[f"L1{suffix}"] = metric_l1
        return metrics_dict


class DecompositionMetrics(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ssim = SSIM(size_average=False, data_range=1.0)

    def forward(self, data_dict, suffix=None, rec='recon_im', target='input_im'):
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''

        metrics_dict = {}

        normal_pred = data_dict['recon_normal'][0]
        normal_gt = data_dict['lr_recon_normal']

        albedo_pred = torch.tensor(data_dict['recon_albedo'][0])
        albedo_gt = torch.tensor(data_dict['lr_recon_albedo'])

        compute_spec_metrics = 'lr_recon_specular_shading' in data_dict

        if compute_spec_metrics:
            spec_pred = data_dict['recon_specular_shading'][0]
            spec_gt = data_dict['lr_recon_specular_shading']

        mask = ~(data_dict['lr_recon_im_mask'] > .0)

        normal_l1 = mask_mean(torch.abs(normal_pred - normal_gt).sum(dim=1, keepdim=True), mask, dim=(1, 2, 3)).mean()
        normal_mse = mask_mean((normal_pred - normal_gt) ** 2, mask.expand(-1, 3, -1, -1), dim=(1, 2, 3)).mean()

        normal_dot = mask_mean(1 - (normal_pred * normal_gt).sum(dim=1, keepdim=True), mask, dim=(1, 2, 3)).mean()
        normal_deviation = mask_mean(torch.acos((normal_pred * normal_gt).sum(dim=1, keepdim=True).clamp(0, 1)) / math.pi * 180, mask, dim=(1, 2, 3)).mean()

        albedo_sie = mask_mean(((albedo_pred - albedo_gt) - mask_mean(albedo_pred - albedo_gt, mask.expand(-1, 3, -1, -1), dim=(-2, -1), keepdim=True)) ** 2, mask.expand(-1, 3, -1, -1), dim=(1, 2, 3)).mean()
        albedo_l1 = mask_mean(torch.abs(albedo_pred - albedo_gt).sum(dim=1, keepdim=True), mask, dim=(1, 2, 3)).mean()

        albedo_pred[mask.expand(-1, 3, -1, -1)] = 0
        albedo_gt[mask.expand(-1, 3, -1, -1)] = 0

        albedo_ssim = self.ssim(albedo_pred * .5 + .5, albedo_gt * .5 + .5).mean()

        if compute_spec_metrics:
            spec_l1 = mask_mean(torch.abs(spec_pred - spec_gt), mask, dim=(1, 2, 3)).mean()
            spec_mse = mask_mean((spec_pred - spec_gt) ** 2, mask, dim=(1, 2, 3)).mean()
            spec_sie = mask_mean(((spec_pred - spec_gt) - mask_mean(spec_pred - spec_gt, mask, dim=(-2, -1), keepdim=True)) ** 2, mask, dim=(1, 2, 3)).mean()

        metrics_dict[f"Normal_l1{suffix}"] = normal_l1
        metrics_dict[f"Normal_mse{suffix}"] = normal_mse
        metrics_dict[f"Normal_dot{suffix}"] = normal_dot
        metrics_dict[f"Normal_deviation{suffix}"] = normal_deviation

        metrics_dict[f"Albedo_sie{suffix}"] = albedo_sie
        metrics_dict[f"Albedo_l1{suffix}"] = albedo_l1
        metrics_dict[f"Albedo_ssim{suffix}"] = albedo_ssim

        if compute_spec_metrics:
            metrics_dict[f"Spec_l1{suffix}"] = spec_l1
            metrics_dict[f"Spec_mse{suffix}"] = spec_mse
            metrics_dict[f"Spec_sie{suffix}"] = spec_sie

        return metrics_dict
