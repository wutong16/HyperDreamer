import os
import kaolin as kal
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from . import mesh 
from .render import Renderer
from editing.src.configs.train_config import GuideConfig


class TexturedMeshModel(nn.Module):
    def __init__(self,
                 opt: GuideConfig,
                 render_grid_size=1200,
                 texture_resolution=1200,
                 initial_texture_path=None,
                 cache_path=None,
                 device=torch.device('cuda'),
                 ):

        super().__init__()
        self.device = device
        self.opt = opt
        self.dy = self.opt.dy
        self.mesh_scale = self.opt.shape_scale
        self.texture_resolution = texture_resolution
        if initial_texture_path is not None:
            self.initial_texture_path = initial_texture_path
        else:
            self.initial_texture_path = self.opt.initial_texture
        
        self.cache_path = cache_path
        self.num_features = 3
        if self.opt.mtl_path is not None:
            self.mtl_file = self.opt.mtl_path
        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size),
                                 interpolation_mode=self.opt.texture_interpolation_mode)
        self.mesh, self.mat_name= self.init_meshes()
        self.default_color = [0.8, 0.1, 0.8]
        self.texture_img, self.mask_texture_img,self.mask_zero_texture_img = self.init_paint()
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        
        
        self.original_texture_img = None
        self.base_texture_img = None
        self.base_texture_mask = None
        self.editing_texture_img = None


        self.vt, self.ft = self.mesh.vt, self.mesh.ft
            
        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long()).detach()

        self.n_eigen_values = 20
        self._L = None
        self._eigenvalues = None
        self._eigenvectors = None
        
        self.mesh.vt = self.vt
        self.mesh.ft = self.ft.long()

 

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
        vertices -= vertices.mean(dim=0)[None, :]
        vertices /= vertices.norm(dim=1).max()
        vertices *= mesh_scale
        vertices[:, 1] += dy
        return vertices

    def init_meshes(self):
        mesh_tex, mat_name = mesh.load_mesh(self.opt.shape_path)   
        return mesh_tex, mat_name

    def zero_meta(self):
        with torch.no_grad():
            self.meta_texture_img[:] = 0
    
    def rgb2gray(self,image): #input:tensor :mask_rgb [1,3,w,w]
        image_gray = torch.sum(image.squeeze(0),dim=0)  #[w,w]
        image_gray = (image_gray > 0.2).float().unsqueeze(0).unsqueeze(0).to(self.device) #[1,1,w,w]
        return image_gray #output[1,1,w,w]
    
    def preparing_editing_texture(self):
        self.base_texture_img = self.texture_img.clone().detach() #[2048, 2048]
        self.base_texture_mask = self.mask_texture_img.clone().detach()
        self.base_texture_mask = self.rgb2gray(self.base_texture_mask)  #[1,1,w,w]
        
        self.mask_texture_img = nn.Parameter(self.rgb2gray(self.mask_texture_img.clone().detach()))
        
        base_texture_mask_colored = self.base_texture_mask * torch.Tensor(self.default_color).reshape(1, 3, 1, 1).to(self.device)
        editing_texture_img = (self.base_texture_img * torch.abs(1 - self.base_texture_mask)).to(self.device)  #editting regions is 0

        editing_texture_img_colored = editing_texture_img + base_texture_mask_colored
        self.texture_img = nn.Parameter(editing_texture_img_colored)
        self.meta_texture_img = nn.Parameter(editing_texture_img)

        #update texture_img(target editing regions setting to zero)
        #update meta_texture(equal to texture_img(modified))

    def init_paint(self):
        if self.initial_texture_path is not None:
            texture = torch.Tensor(np.array(Image.open(self.initial_texture_path).resize(
                (self.texture_resolution, self.texture_resolution)))).permute(2, 0, 1).to(self.device).unsqueeze(0) / 255.0
        else:
            texture = torch.ones(1, 3, self.texture_resolution, self.texture_resolution).to(self.device) * torch.Tensor(
                self.default_color).reshape(1, 3, 1, 1).to(self.device)
            
        mask_texture_img = torch.zeros(1, 3, self.texture_resolution, self.texture_resolution).to(self.device)
        mask_zero_texture_img = torch.zeros(1, 3, self.texture_resolution, self.texture_resolution).to(self.device)
        
        texture_img = nn.Parameter(texture)
        mask_texture_img = nn.Parameter(mask_texture_img)
        mask_zero_texture_img = nn.Parameter(mask_zero_texture_img)
        
        return texture_img, mask_texture_img, mask_zero_texture_img

    def invert_color(self, color: torch.Tensor) -> torch.Tensor:
        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2

        pinv = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).to(self.device)) @ A.T)
        if len(color) == 1 or type(color) is torch.Tensor:
            init_color_in_latent = color @ pinv.T
        else:
            init_color_in_latent = pinv @ torch.tensor(
                list(color)).float().to(A.device)
        return init_color_in_latent

    def change_default_to_median(self):
        diff = (self.texture_img - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        default_mask = (diff < 0.1).float().unsqueeze(0)
        median_color = self.texture_img[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(axis=1)
        with torch.no_grad():
            self.texture_img.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)


    def get_params(self,use_texture_mask=False, use_texture_mask_zero=False, use_texture_original=False):
        if use_texture_mask:
            return [self.mask_texture_img]
        elif use_texture_mask_zero:
            return [self.mask_zero_texture_img]
        elif use_texture_original:
            return [self.original_texture_img]
        else:
            return [self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_mesh(self, path):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        f_vn = self.mesh.face_n
        vn = self.mesh.vn
        h0, w0 = 256, 256
        ssaa, name = 1, ''

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]
        if f_vn is not None:
            f_vnp = f_vn.cpu().numpy()
        if f_vn is not None:
            v_n = vn.cpu().numpy()
        
        colors = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)
        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()  #vt1
        colors = Image.fromarray(colors)
        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)

        colors.save(os.path.join(path, f'{name}albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')
        self.mtl_file = mtl_file
        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info('writing vertices texture coords {vt_np.shape}')
            for vt in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {vt[0]} {vt[1]} \n')
                
            if f_vn is not None:
                for vni in v_n:
                    fp.write(f'vn {vni[0]} {vni[1]} {vni[2]} \n')
            
            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl {self.mat_name} \n')
            for i in range(len(f_np)):
                fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl defaultMat \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')


    
    
    
    def render(self, theta=None, phi=None, radius=None, background=None,
               use_meta_texture=False, render_cache=None, use_median=False, dims=None, 
               use_texture_mask=False,
               use_texture_mask_zero=False,
               use_texture_original=False):
        
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None

        background_type = 'none'
        use_render_back = False
        if background is not None and type(background) == str:
            background_type = background
            use_render_back = True
            
        if use_texture_mask:
            texture_img_render = self.mask_texture_img
        elif use_texture_mask_zero:
            texture_img_render = self.mask_zero_texture_img
        elif use_texture_original:
            texture_img_render = self.original_texture_img
        elif use_meta_texture:
            texture_img_render = self.meta_texture_img
        else:
            texture_img_render = self.texture_img
            


        vertices = self.mesh.vertices

        
        if use_median:
            diff = (texture_img_render - torch.tensor(self.default_color).view(1, 3, 1, 1).to(
                self.device)).abs().sum(axis=1)
            default_mask = (diff < 0.1).float().unsqueeze(0)

            
            median_color = texture_img_render[0, :].reshape(3, -1)[:, default_mask.flatten() == 0].mean(
                axis=1)
            texture_img_render = texture_img_render.clone()
            with torch.no_grad():
                texture_img_render.reshape(3, -1)[:, default_mask.flatten() == 1] = median_color.reshape(-1, 1)
        

            
        pred_features, mask, depth, normals, render_cache = self.renderer.render_single_view_texture(vertices,
                                                                                                    self.mesh.faces,
                                                                                                    self.face_attributes,
                                                                                                    texture_img_render,
                                                                                                    elev=theta,
                                                                                                    azim=phi,
                                                                                                    radius=radius,
                                                                                                    look_at_height=self.dy,
                                                                                                    render_cache=render_cache,
                                                                                                    dims=dims,
                                                                                                    background_type=background_type)
            
                                                                                                                     
        
        mask = mask.detach()
        #pred_map = pred_features

        if use_render_back:
            pred_map = pred_features
            pred_back = pred_features
        else:
            if background is None:
                background = torch.Tensor([0.8, 0.8, 0.8]).to(self.device) 
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            elif len(background.shape) == 1 :
                pred_back = torch.ones_like(pred_features) * background.reshape(1, 3, 1, 1)
            else:
                pred_back = background

            pred_map = pred_back * (1 - mask) + pred_features * mask

        if not use_meta_texture:
            pred_map = pred_map.clamp(0, 1)
            pred_features = pred_features.clamp(0, 1)
            

        return {'image': pred_map, 
                'mask': mask, 
                'foreground': pred_features, 
                'depth': depth, 
                'normals': normals, 
                'render_cache': render_cache,
                'texture_map': texture_img_render}


        
