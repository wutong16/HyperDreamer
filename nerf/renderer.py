import os
import math
import cv2
import trimesh
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

import mcubes
import raymarching

from meshutils import decimate_mesh, clean_mesh
from .utils import custom_meshgrid, safe_normalize

from lib.render import mesh, render

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class DMTet():
    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [ 1,  0,  2, -1, -1, -1],
            [ 4,  0,  3, -1, -1, -1],
            [ 1,  4,  2,  1,  3,  4],
            [ 3,  1,  5, -1, -1, -1],
            [ 2,  3,  0,  2,  5,  3],
            [ 1,  4,  0,  1,  5,  4],
            [ 4,  2,  5, -1, -1, -1],
            [ 4,  5,  2, -1, -1, -1],
            [ 4,  1,  0,  4,  5,  1],
            [ 3,  2,  0,  3,  5,  2],
            [ 1,  3,  5, -1, -1, -1],
            [ 4,  1,  2,  4,  3,  1],
            [ 3,  0,  4, -1, -1, -1],
            [ 2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device=device)
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device=device)

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            ) # indexing='ij')

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    def __call__(self, pos_nx3, sdf_n, tet_fx4, get_uv=False):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1 # two vertices of the edge is one neg and one pos
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device=self.device)
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1) # find the approximate zero sets of the edges

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        if not get_uv:
            return verts, faces

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces, uvs, uv_idx

def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):

    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))


def laplacian_uniform(verts, faces):

    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss

def laplace_regularizer_const(v_pos, t_pos_idx):
    t_pos_idx = t_pos_idx.long()
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

class Gaussian(nn.Module):
    def __init__(self, ksize, sigma):
        super(Gaussian, self).__init__()
        self.gaussian_filter = self._gaussian_2dconv(ksize, sigma)

    def forward(self, x):
        return self.gaussian_filter(x)

    @torch.no_grad()
    def _gaussian_2dconv(self, ksize=3, sigma=1):
        x = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        y = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = torch.from_numpy(kernel).cuda().float()  # [None].repeat(3,1,1)
        m = nn.Conv2d(1, 1, ksize, stride=1, padding=ksize // 2,
                      padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum(0)
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.max_level = None
        self.dmtet = opt.dmtet
        self.cuda_ray = opt.cuda_ray
        self.taichi_ray = opt.taichi_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.glctx = None

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
        
        if self.opt.dmtet:
            # load dmtet vertices
            tets = np.load('tets/{}_tets.npz'.format(self.opt.tet_grid_size))
            self.verts = - torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * 2 # covers [-1, 1]
            if tets['vertices'].mean() ** 2 > 0.1:
                self.verts += 1
            self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
            self.tet_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
            self.dmtet = DMTet('cuda')

            edges = torch.tensor([0,1, 0,2, 0,3, 1,2, 1,3, 2,3], dtype=torch.long, device="cuda") # six edges for each tetrahedron.
            all_edges = self.indices[:,edges].reshape(-1,2) # [M * 6, 2]
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

            if self.opt.h <= 2048 and self.opt.w <= 2048:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = dr.RasterizeGLContext()
        
        if self.taichi_ray:
            raise NotImplementedError

    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        
        if self.opt.density_activation == 'exp':
            g = self.opt.blob_density * torch.exp(- d / (2 * self.opt.blob_radius ** 2))
        else:
            g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g
    
    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not (self.cuda_ray or self.taichi_ray):
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128, speculars=None, semantic=False):

        if self.opt.dmtet:

            pred = self.dmtet_decoder(self.encoder(self.verts))
            sdf, deform = pred[:,0], pred[:,1:]
            deform = torch.tanh(deform) / self.opt.tet_grid_size

            vertices, triangles = self.dmtet(self.verts + deform, sdf, self.indices)

            vertices = vertices.detach().cpu().numpy()
            triangles = triangles.detach().cpu().numpy()

        else:

            if resolution is None:
                resolution = self.grid_size

            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh) \
                    if np.greater(self.mean_density, 0) else self.density_thresh
            else:
                density_thresh = self.density_thresh
            
            # TODO: use a larger thresh to extract a surface mesh from the density field, but this value is very empirical...
            if self.opt.density_activation == 'softplus':
                density_thresh = density_thresh * 25
            
            sigmas = np.zeros([resolution, resolution, resolution], dtype=np.float32)

            # query
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat([-xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                        val = self.density(pts.to(self.aabb_train.device))
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]

            print(f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
            vertices = vertices / (resolution - 1.0) * 2 - 1

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        # # try:
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.01)

        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)
        # except:
        #     print('Failed to clean mesh!!')

        v = torch.from_numpy(vertices).contiguous().float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(self.aabb_train.device)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            # v, f: torch Tensor
            device = v.device
            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4 # for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            
            if self.glctx is None:
                if h <= 2048 and w <= 2048:
                    self.glctx = dr.RasterizeCudaContext()
                else:
                    self.glctx = dr.RasterizeGLContext()

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            feats_spec = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            feats_sam = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
            feats_sg_roughness = torch.zeros(h * w, 1, device=device, dtype=torch.float32)
            feats_sg_specular = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # batched inference to avoid OOM
                all_feats = []
                all_speculars = []
                all_semantics = []
                all_sg_roughness = []
                all_sg_speculars = []
                all_normal_offsets = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())

                    if semantic or speculars is not None:
                        labels = self.density(xyzs[head:tail], predict_class=True)['pred_class'].argmax(-1)
                        if speculars is not None:
                            N = len(speculars)
                            specular_colors = torch.from_numpy((np.ones((N, 3)) * speculars.reshape((N, -1)))).float().cuda()
                            results_ = specular_colors[labels.long()]
                            all_speculars.append(results_)
                        if semantic:
                            semantic_colors = torch.from_numpy(np.asarray(matplotlib.colormaps[self.opt.colormap].colors)).float().cuda()
                            results_ = semantic_colors[labels.long()]
                            all_semantics.append(results_)
                    if self.opt.use_svbrdf:
                        sg_envmap_material = self.svbrdf_forward(xyzs[head:tail], predict_class=self.opt.material_offset)
                        if self.opt.material_offset:
                            if self.opt.soft_material:
                                weight = torch.softmax(sg_envmap_material['pred_class'], 1)
                                base_roughness = (self.RegionMaterials.roughness[None] * weight).sum(-1)
                                base_specular = (self.RegionMaterials.specular[None] * weight).sum(-1)
                            else:
                                labels = sg_envmap_material['pred_class'].argmax(-1)
                                base_roughness = self.RegionMaterials.roughness[labels]
                                base_specular = self.RegionMaterials.specular[labels]
                                # base_albedo_gray = self.RegionMaterials.albedo_gray[labels]
                            sg_envmap_material['sg_specular_reflectance'] = base_specular[..., None] + self.opt.material_offset_ratio * sg_envmap_material['sg_specular_reflectance']
                            sg_envmap_material['sg_roughness'] = base_roughness[..., None] + self.opt.material_offset_ratio * sg_envmap_material['sg_roughness']
                        all_sg_roughness.append(sg_envmap_material['sg_roughness'])
                        all_sg_speculars.append(sg_envmap_material['sg_specular_reflectance'])
                        if self.opt.normal_offset:
                            normal_offsets = sg_envmap_material['delta_normal']

                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)
                if speculars is not None:
                    feats_spec[mask] = torch.cat(all_speculars, dim=0)
                if semantic:
                    feats_sam[mask] = torch.cat(all_semantics, dim=0)
                if self.opt.use_svbrdf:
                    feats_sg_roughness[mask] = torch.cat(all_sg_roughness, dim=0)
                    feats_sg_specular[mask] = torch.cat(all_sg_speculars, dim=0)

            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as an antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

            # do ssaa after the NN search, in numpy
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

            if speculars is not None:
                feats_spec = feats_spec.view(h,w,-1)
                feats_spec = feats_spec.cpu().numpy()
                feats_spec = (feats_spec * 255).astype(np.uint8)
                feats_spec[tuple(inpaint_coords.T)] = feats_spec[tuple(search_coords[indices[:, 0]].T)]

                feats_spec = cv2.cvtColor(feats_spec, cv2.COLOR_RGB2BGR)

                # do ssaa after the NN search, in numpy
                if ssaa > 1:
                    feats_spec = cv2.resize(feats_spec, (w0, h0), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(path, f'{name}specular.png'), feats_spec)

            if semantic:
                feats_sam = feats_sam.view(h,w,-1)
                feats_sam = feats_sam.cpu().numpy()
                feats_sam = (feats_sam * 255).astype(np.uint8)
                feats_sam[tuple(inpaint_coords.T)] = feats_sam[tuple(search_coords[indices[:, 0]].T)]

                feats_sam = cv2.cvtColor(feats_sam, cv2.COLOR_RGB2BGR)

                # do ssaa after the NN search, in numpy
                if ssaa > 1:
                    feats_sam = cv2.resize(feats_sam, (w0, h0), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(path, f'{name}semantic.png'), feats_sam)

            if self.opt.use_svbrdf:
                feats_sg_specular = (feats_sg_specular.view(h,w,-1).cpu().numpy()* 255).astype(np.uint8)
                feats_sg_specular[tuple(inpaint_coords.T)] = feats_sg_specular[tuple(search_coords[indices[:, 0]].T)]
                feats_sg_specular = cv2.cvtColor(feats_sg_specular, cv2.COLOR_RGB2BGR)
                if ssaa > 1:
                    feats_sg_specular = cv2.resize(feats_sg_specular, (w0, h0), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(path, f'{name}sg_specular.png'), feats_sg_specular)

                feats_sg_roughness = (feats_sg_roughness.view(h, w, -1).cpu().numpy() * 255).astype(np.uint8)
                feats_sg_roughness[tuple(inpaint_coords.T)] = feats_sg_roughness[tuple(search_coords[indices[:, 0]].T)]
                feats_sg_roughness = cv2.cvtColor(feats_sg_roughness, cv2.COLOR_RGB2BGR)
                if ssaa > 1:
                    feats_sg_roughness = cv2.resize(feats_sg_roughness, (w0, h0), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(path, f'{name}sg_roughness.png'), feats_sg_roughness)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')
                
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')

        _export(v, f)

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        # nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # nears.unsqueeze_(-1)
        # fars.unsqueeze_(-1)
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, self.opt.num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / self.opt.num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, self.opt.num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, self.opt.num_steps, -1)

        # upsample z_vals (nerf-like)
        if self.opt.upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.opt.upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, self.opt.upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, self.opt.upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        light_d = light_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        dirs = safe_normalize(dirs)
        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d.reshape(-1, 3), ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
        if normals is not None:
            normals = normals.view(N, -1, 3)

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)

        if self.training:
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.sum(-1).mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) and normals is not None:
                normal_image = torch.sum(weights.unsqueeze(-1) * (normals + 1) / 2, dim=-2) # [N, 3], in [0, 1]
                results['normal_image'] = normal_image
        
        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum

        return results


    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]

        results = {}

        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)
            dirs = safe_normalize(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]

            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, binarize)
            
            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if (self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0) and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), normals, ts, rays, T_thresh, binarize) # original: (normals + 1) / 2
                results['normal_image'] = normal_image

            # weights normalization
            results['weights'] = weights

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                dirs = safe_normalize(dirs)
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)

                rays_alive = rays_alive[rays_alive >= 0]
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum

        return results

    @torch.no_grad()
    def init_tet(self):

        # if self.cuda_ray:
        #     density_thresh = min(self.mean_density, self.density_thresh)
        # else:
        #     density_thresh = self.density_thresh

        # if self.opt.density_activation == 'softplus':
        #     density_thresh = density_thresh * 25

        # # init scale
        # sigma = self.density(self.verts)['sigma'] # verts covers [-1, 1] now
        # mask = sigma > density_thresh
        # valid_verts = self.verts[mask]
        # self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
        # self.verts = self.verts * self.tet_scale

        # # init sigma
        # sigma = self.density(self.verts)['sigma'] # new verts
        # self.sdf.data += (sigma - density_thresh).clamp(-1, 1)

        # print(f'[INFO] init dmtet: scale = {self.tet_scale}')
        pass

    @torch.no_grad()
    def tet_to_density(self, name='', scale=1):

        N = len(self.verts)
        inside = self.sdf.data.detach().cpu().numpy() > 0
        verts = self.verts + torch.tanh(self.deform) / self.opt.tet_grid_size
        vertices = verts.detach().cpu().numpy()
        colors = np.zeros((N,4))
        colors[~inside] = np.asarray([255,0,0,0.2])
        colors[inside] = np.asarray([0,0,255,1])
        # pcd = trimesh.PointCloud(vertices, colors)

        pcd = trimesh.PointCloud(vertices[inside], colors[inside])
        pcd.export(name)

    def run_dmtet(self, rays_o, rays_d, mvp, h, w, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, predict_class=False, pose=None, **kwargs):
        device = mvp.device
        campos = rays_o[:, 0, :] # only need one ray per batch
        B = rays_o.shape[0]

        results = {}

        # get mesh
        pred = self.dmtet_decoder(self.encoder(self.verts))
        sdf, deform = pred[:,0], pred[:,1:]
        deform = torch.tanh(deform) / self.opt.tet_grid_size

        verts, faces, uvs, uv_idx = self.dmtet(self.verts + deform, sdf, self.indices, get_uv=True)

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx)
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        # prepare material
        if shading == 'svbrdf':
            material = {
                'svbrdf_func': self.svbrdf_forward,
                'predict_class': predict_class,
                'material_offset': self.opt.material_offset,
                'soft_material': self.opt.soft_material,
                'material_offset_ratio': self.opt.material_offset_ratio,
                'RegionMaterials': getattr(self, 'RegionMaterials', None),
                'num_classes': getattr(self, 'num_classes', None),
            }
            imesh.material = material
        elif shading == 'albedo':
            material = {
                'color_func': self.density,
                'predict_class': predict_class,
                'num_classes': self.num_classes if hasattr(self, 'num_classes') else None,
            }
            imesh.material = material

        # render mesh
        resolution = (h, w)
        background = torch.ones(B, h, w, 3, dtype=torch.float32, device=device)
        normal_rotate = pose.reshape((-1, 4, 4))[:, :3, :3].permute(0, 2, 1).contiguous()
        normal_rotate = torch.Tensor(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).to(device) @ normal_rotate

        buffers = render.render_mesh(self.glctx, imesh, mvp, campos, None, resolution, msaa=True, background=background,
                                  bsdf=shading, normal_rotate=normal_rotate, if_flip_the_normal=False)

        normal, alpha = buffers['normal'][...,:3], buffers['normal'][...,-1]

        results['image'] = buffers['color'].clamp(0, 1)
        results['albedo'] = buffers['albedo'].clamp(0, 1)
        if shading == 'svbrdf':
            results['spec_color'] = buffers['spec_color'].clamp(0, 1)
        results['depth'] = alpha
        results['normal_image'] = normal
        results['weights_sum'] = alpha
        if 'pred_class' in buffers:
            results['pred_class'] = buffers['pred_class']

        if self.training:
            if self.opt.lambda_mesh_normal > 0:
                results['normal_loss'] = torch.tensor([0], dtype=torch.float32, device="cuda")
            if self.opt.lambda_mesh_laplacian > 0:
                results['lap_loss'] = torch.tensor([0], dtype=torch.float32, device="cuda")
        return results

    def normal_loss(self, normal_pred, normal_gt, mask=None):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        if mask is None:
            l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
            cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1)).mean()
        else:
            l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
            cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
        return l1, cos

    def get_normal_loss(self, pred, target, pose, mask, rotate=True, convert=False, smooth=False):
        '''
        Args:
            pred: pred normal. [h, w, 3]
            target: gt normal. [h, w, 3]
            mask: [h, w]
        '''
        # pred = pred.view(-1, sample_size * sample_size, 3)
        # target = target.view(-1, sample_size * sample_size, 3)
        '''
        rot = pose[:, :3, :3].permute(0, 2, 1).contiguous()
        pred = rot @ pred.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1).contiguous()
        '''

        pose = pose.reshape((-1, 4, 4))
        rot = pose[:, :3, :3].permute(0, 2, 1).contiguous()
        if convert:
            rot = torch.Tensor(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).to(rot.device) @ rot
        if rotate:
            pred = rot @ pred[...,None]
            pred = pred.squeeze().contiguous()

        if self.training and smooth:
            pred = self.gaussian(pred[None].permute(3,0,1,2)).squeeze().permute(1,2,0)
        normal_loss_l1, normal_loss_cos = self.normal_loss(pred, target, mask)
        normal_loss_l1 = torch.nan_to_num(normal_loss_l1)
        normal_loss_cos = torch.nan_to_num(normal_loss_cos)
        return normal_loss_l1, normal_loss_cos

    def depth_loss(self, pearson, pred_depth, depth_gt, mask):
        pred_depth = pred_depth.squeeze()
        pred_depth = torch.nan_to_num(pred_depth)
        depth_gt = depth_gt.squeeze().reshape(-1)
        mask = mask.squeeze().reshape(-1)
        pred_depth = pred_depth.reshape(-1)
        mask = (mask==1)
        co = pearson(pred_depth[mask], depth_gt[mask])
        return 1 - co

    def depth_l1_loss(self, pred_depth, depth_gt, mask):
        l1_loss = nn.L1Loss()(pred_depth * mask , depth_gt * mask)
        return l1_loss

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not (self.cuda_ray or self.taichi_ray):
            return 
        
        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        if self.cuda_ray:
            self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)
        elif self.taichi_ray:
            self.packbits_taichi(self.density_grid.reshape(-1).contiguous(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')


    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        B, N = rays_o.shape[:2]
        device = rays_o.device

        if self.dmtet:
            results = self.run_dmtet(rays_o, rays_d, mvp, h, w, **kwargs)
        elif self.cuda_ray:
            results = self.run_cuda(rays_o, rays_d, **kwargs)
        elif self.taichi_ray:
            raise NotImplementedError
        else:
            if staged:
                depth = torch.empty((B, N), device=device)
                image = torch.empty((B, N, 3), device=device)
                weights_sum = torch.empty((B, N), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                        depth[b:b+1, head:tail] = results_['depth']
                        weights_sum[b:b+1, head:tail] = results_['weights_sum']
                        image[b:b+1, head:tail] = results_['image']
                        head += max_ray_batch
                
                results = {}
                results['depth'] = depth
                results['image'] = image
                results['weights_sum'] = weights_sum

            else:
                results = self.run(rays_o, rays_d, **kwargs)

        return results