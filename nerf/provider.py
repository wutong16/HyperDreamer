import os, sys
import cv2
import math
import random
import numpy as np
from PIL import Image
import trimesh
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline

from .utils import get_rays, safe_normalize

sys.path.insert(0, '../zero123')
from zero123.ldm.models.diffusion.ddim import DDIMSampler
from zero123.local_run import sample_model, load_model_from_config

# from lib import render

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # front = 0             [-front/2, front/2)
    # side (cam left) = 1   [front/2, 180-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [180+front/2, 360-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def pose_rotation(ori_theta=90, ori_azimuth=0, ori_z=3.2, mode=None):

    if mode is not None:
        if mode == 'above':
            x, y, z = (-90.0, 0.0, 0.0)
        elif mode == 'below':
            x, y, z = (90.0, 0.0, 0.0)
        elif mode == 'left':
            x, y, z = (0.0, -90.0, 0.0)
        elif mode == 'right':
            x, y, z = (0.0, 90.0, 0.0)
        elif mode == 'behind':
            x, y, z = (0.0, 180.0, 0.0)
        elif mode == 'sector':
            delta_az = np.random.rand() * 120 - 60
            x, y, z = (0.0, delta_az, 0.0)
        else:
            raise NotImplementedError
    # ori_T = ori_pose[:3, -1] # location
    # ori_theta, ori_azimuth, orsi_z = cartesian_to_spherical(ori_T[None, :]) # spherical location

    # d_T = xyz_to_T(x, y, z)
    # d_theta, d_azimuth, d_z = cartesian_to_spherical(d_T[None, :])

    target_theta = math.radians(ori_theta) + math.radians(x)
    target_azimuth = (math.radians(ori_azimuth) + math.radians(y)) % (2 * math.pi)
    target_z = ori_z + z
    ''' # zero123 coord 
    eyes = np.zeros((1, 3))

    eyes[:, 0] = target_z * np.sin(target_theta) * np.cos(np.pi - target_azimuth)  # x
    eyes[:, 1] = target_z * np.sin(target_theta) * np.sin(np.pi - target_azimuth)  # z
    eyes[:, 2] = target_z * np.cos(target_theta)  # y

    pose = camera_pose(eyes, -eyes, np.asarray([0,0,1]))
    '''

    pose, dirs = circle_poses('cpu', radius=torch.tensor(target_z).view(-1),
                   theta=torch.tensor(target_theta).view(-1) / np.pi * 180,
                   phi=torch.tensor(target_azimuth).view(-1) / np.pi * 180, return_dirs=True)

    return pose, dirs, x, -y, z # fixme: that's sooooo weird!! It should be y to return, but it's always opposite

def rand_poses(size, device, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size, device=device),
                torch.abs(torch.randn(size, device=device)),
                torch.randn(size, device=device),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        # phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis = (torch.rand(size, device=device)/size + torch.linspace(0,1,steps=size+1,device=device)[:-1]) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius

def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), return_dirs=False, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs    

def blend_rgba(img):
    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
    return img

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000 # infinite

        self.sr_cnt = 0 # hard coded

    def get_default_view_data(self):

        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx = H / 2
        cy = W / 2

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device)
        thetas = torch.FloatTensor(self.opt.ref_polars).to(self.device)
        phis = torch.FloatTensor(self.opt.ref_azimuths).to(self.device)
        poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        projection = torch.tensor([
            [2*focal/W, 0, 0, 0],
            [0, -2*focal/H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(radii), 1, 1)

        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, H, W, -1)

        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'pose': poses,
            'polar': self.opt.ref_polars,
            'azimuth': self.opt.ref_azimuths,
            'radius': self.opt.ref_radii,
        }

        return data

    def get_sr_view_data(self, sr_size=3):

        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx = H / 2
        cy = W / 2

        sr_data = []

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device)
        thetas = torch.rand(1, device=self.device) * (self.opt.theta_range[1] - self.opt.theta_range[0]) + self.opt.theta_range[0]
        phis_sz = (torch.linspace(0,1.0, sr_size+1) + torch.rand(1))* (self.opt.phi_range[1] - self.opt.phi_range[0]) + self.opt.phi_range[0]
        phis_sz = phis_sz[:-1]
        phis_sz[phis_sz > 360] -= 360
        phis_sz = phis_sz.to(self.device)
        fov = self.opt.default_fovy - 5 * random.random()

        for i in range(sr_size):
            phis = phis_sz[i]
            
            poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)    

            focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, cx, cy])

            projection = torch.tensor([
                [2*focal/W, 0, 0, 0],
                [0, -2*focal/H, 0, 0],
                [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                [0, 0, -1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            mvp = projection @ torch.inverse(poses) # [1, 4, 4]

            # sample a low-resolution but full image
            rays = get_rays(poses, intrinsics, H, W, -1)

            # delta polar/azimuth/radius to default view
            delta_polar = thetas - self.opt.default_polar
            delta_azimuth = phis - self.opt.default_azimuth
            delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
            delta_radius = radii - self.opt.default_radius

            data = {
                'H': H,
                'W': W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'dir': dirs,
                'mvp': mvp,
                'pose': poses,
                'polar': delta_polar,
                'azimuth': delta_azimuth,
                'radius': delta_radius,
            }

            sr_data.append(data)

        self.sr_cnt += 1
        return sr_data

    def collate(self, index):

        B = len(index)

        if self.training:
            # random pose on the fly
            poses, dirs, thetas, phis, radius = rand_poses(B, self.device, self.opt, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]

        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.default_polar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = self.opt.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'pose': poses,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader
