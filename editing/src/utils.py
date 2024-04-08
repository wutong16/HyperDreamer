import os
import cv2
import random
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from loguru import logger
from typing import List
from pathlib import Path
from matplotlib import cm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from editing.src.configs.train_config import TrainConfig

trainconfig = TrainConfig()
device = trainconfig.optim.device


def get_view_direction(thetas, phis, overhead, front):      #angle_overhead=30.0, angle_front=60.0
    #                   phis [B,];          thetas: [B,]
    # front = 0                                            [0, front)                      #(-front + 360, front]
    # side (left) = 1                                      [front, 180)                    #(front, front + 45]
    # back = 2                                             [180, 180+front)                #(front + 45 , 360 - (front + 45)]
    # side (right) = 3                                     [180+front, 360)                #(360 - (front + 45), -front + 360]
    # top = 4                                              [0, overhead]                   #[0, overhead]
    # bottom = 5                                           [180-overhead, 180]             #[180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    
    #front=70
    res[(phis > (2*np.pi - front)) & (phis <= front)] = 0
    res[(phis > front) & (phis <= (np.pi/2 + front))] = 1
    res[(phis > (np.pi/2 + front)) & (phis <= (2*np.pi - (np.pi/2 + front)))] = 2
    res[(phis > (2*np.pi - (np.pi/2 + front))) & (phis <= (2*np.pi - front))] = 3
    
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    
    # override by thetas
    """
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    """

    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path

def rgb2gray(image): #input:tensor :mask_rgb [1,3,w,w]
        image_gray = torch.sum(image.squeeze(0),dim=0)  #[w,w]
        image_gray = (image_gray > 0).float().unsqueeze(0).unsqueeze(0).to(device) #[1,1,w,w]
        return image_gray

def tensor_toImage(tensor_img): 
    ## input: tensor    value[0-1]   [1,3,w,w] or [1,1,w,w]  float()
    ## output: np.array value[0-255] [w,w,3] or [w,w,1]      np.uint8
    image = tensor_img.permute(0,2,3,1).squeeze(0).cpu().detach().numpy()
    image = (image.copy() * 255).astype(np.uint8)
    return image
    
def Image_totensor(image): ##image: [w,w,3] or [w,w,1]
    tensor_img = torch.from_numpy(np.array(image)).permute(2,0,1).unsqueeze(0).float() / 255 #[1,3,w,w]
    tensor_img = tensor_img.to(device)
    return tensor_img

def save_texture_img(path, texture_img, name=''):
    ##save mask
    colors = torch.sum(texture_img.squeeze(0),dim=0)  #[w,w]
    colors = (colors > 0).float().unsqueeze(-1).to(device) #[1,1,w,w]
    colors = colors.cpu().detach().numpy()
    colors = (colors * 255).astype(np.uint8)
    colors = Image.fromarray(colors)
    colors.save(os.path.join(path, f'{name}_texture_img.png'))

def save_colormap(tensor: torch.Tensor, path: Path):
    Image.fromarray((cm.seismic(tensor.cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def smooth_image(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
    """apply gaussian blur to an image tensor with shape [C, H, W]"""
    img = T.GaussianBlur(kernel_size=(51, 51), sigma=(sigma, sigma))(img)
    return img


def get_nonzero_region(mask:torch.Tensor):
    # Get the indices of the non-zero elements
    nz_indices = mask.nonzero()
    # Get the minimum and maximum indices along each dimension
    min_h, max_h = nz_indices[:, 0].min(), nz_indices[:, 0].max()
    min_w, max_w = nz_indices[:, 1].min(), nz_indices[:, 1].max()

    # Calculate the size of the square region
    size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
    # Calculate the upper left corner of the square region
    h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
    w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

    min_h = int(h_start)
    min_w = int(w_start)
    max_h = int(min_h + size)
    max_w = int(min_w + size)

    return min_h, min_w, max_h, max_w

def mask_thinned_edge(image,threshold1=50, threshold2=100):  ##input:image(mask) tensor[1,1,w,w]
    image_numpy = tensor_toImage(image)
    gray = image_numpy[:,:,0]
    kernel = np.ones((3, 3), np.uint8) 
    #thinned_edges = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)
    #dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    #thinned_edges = cv2.erode(dilated_edges, kernel, iterations=2)
    thinned_edges=edges
    h,w = thinned_edges.shape[0], thinned_edges.shape[1]
    thinned_edges = thinned_edges.reshape((h,w,1))
    thinned_edges = Image_totensor(thinned_edges)
    return thinned_edges

def calculate_refine_mask(update_mask, mask, kernel_size):
    kernel_dilate = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    update_mask_numpy = tensor_toImage(update_mask)
    junction_mask = cv2.morphologyEx(update_mask_numpy, cv2.MORPH_GRADIENT, kernel)
    junction_mask = cv2.dilate(junction_mask, kernel_dilate)
    junction_mask = torch.from_numpy(junction_mask).float()
    junction_mask = junction_mask.unsqueeze(0).unsqueeze(0).clamp(0,1).to(device)
    junction_mask[mask == 0] = 0
        
    refine_mask=junction_mask
        
    return refine_mask


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def gaussian_blur(image:torch.Tensor, kernel_size:int, std:int, device='cuda') -> torch.Tensor:
    gaussian_filter = gkern(kernel_size, std=std)
    gaussian_filter /= gaussian_filter.sum()

    image = F.conv2d(image, gaussian_filter.unsqueeze(0).unsqueeze(0).to(device), padding=kernel_size // 2)
    return image

def color_with_shade(color: List[float],z_normals:torch.Tensor,light_coef=0.7):
    normals_with_light = (light_coef + (1 - light_coef) * z_normals.detach())
    shaded_color = torch.tensor(color).view(1, 3, 1, 1).to(
        z_normals.device) * normals_with_light
    return shaded_color

def polar_to_xyz(theta, phi, r):
    """ assume y-axis is the up axis """
    ##elev(theta) azim(phi)
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    x = np.sin(theta) * np.sin(phi) * r
    y = np.cos(theta) * r
    z = np.sin(theta) * np.cos(phi) * r

    return [x, y, z]

def visualize_viewpoints(output_dir, dataloader: DataLoader, name):
    theta_list=[]
    phi_list=[]
    radius_list=[]
    for i, data in enumerate(dataloader): 
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        theta = round(np.rad2deg(theta))
        phi = round(np.rad2deg(phi))


        theta_list.append(theta)
        phi_list.append(phi)
        radius_list.append(radius)

    
    DIST = radius_list[0]
    xyz_list = [polar_to_xyz(theta, phi, DIST) for theta, phi in zip(theta_list, phi_list)]

    xyz_np = np.array(xyz_list)

    color_np = np.array([[0, 0, 0]]).repeat(xyz_np.shape[0], 0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    SCALE = 0.8
    ax.set_xlim((-DIST, DIST))
    ax.set_ylim((-DIST, DIST))
    ax.set_zlim((-SCALE * DIST, SCALE * DIST))

    ax.scatter(xyz_np[:, 0], xyz_np[:, 2], xyz_np[:, 1], s=100, c=color_np, depthshade=True, label="Rendering views")
    ax.scatter([0], [0], [0], c=[[1, 0, 0]], s=100, depthshade=True, label="Object center")

    # draw hemisphere
    # theta inclination angle
    # phi azimuthal angle
    n_theta = 50    # number of values for theta
    n_phi = 200     # number of values for phi
    r = DIST        #radius of sphere

    # theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
    theta, phi = np.mgrid[0.0:1*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.25, linewidth=1)

    # Make the grid
    ax.quiver(
        xyz_np[:, 0], 
        xyz_np[:, 2], 
        xyz_np[:, 1], 
        -xyz_np[:, 0], 
        -xyz_np[:, 2], 
        -xyz_np[:, 1],
        normalize=True,
        length=0.3
    )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.view_init(30, 35)
    ax.legend()

    plt.show()

    plt.savefig(os.path.join(output_dir, f"visual_viewpoints_{name}.png"))