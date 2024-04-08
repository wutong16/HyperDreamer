import os
import json
import cv2
import numpy as np
from derender3d.scripts.single_image_decomposition import derender3d_main


def image_derender(image, img_dir, derender_dir):
    if not os.path.exists(derender_dir):
        print("========== run derender3d ============")
        derender3d_main(img_dir, derender_dir)
        print('derendering done.')
    spec_img = cv2.resize(cv2.imread(os.path.join(derender_dir, 'spec.jpg')), image.shape[:2]) / 255.
    albedo_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(derender_dir, 'albedo.jpg')), cv2.COLOR_BGR2RGB), image.shape[:2]) / 255.
    with open(os.path.join(derender_dir, 'spec_params.json'), 'r') as f:
        spec_params = json.load(f)
        spec_strength = spec_params['recon_light_spec_strength']
    return spec_img, albedo_img, spec_strength

def region_aware_albedo(albedo, alpha, masks):
    albedo_r = albedo.copy()
    region_rgb_mean = []
    for mask_ in masks:
        mask = mask_['segmentation']
        region_mean = albedo[mask].mean(axis=0)
        region_rgb_mean.append(region_mean)
        albedo_r[mask] = albedo_r[mask] * (1 - alpha) + region_mean * alpha
    return albedo_r, np.array(region_rgb_mean)

def gaussian_kernel(kernel_size, sigma=1, muu=0):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    normal = 1 / (2.0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal

    gauss /= gauss.sum()

    return gauss
