# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light
import numpy as np

from lib.sg_render import render_with_sg
# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        bsdf,
        normal_rotate,
        mode,
        if_flip_the_normal,
        mask,
    ):

    perturbed_nrm = None

    B, h, w, _ = gb_pos.shape
    buffers = {}
    
    #produces a final normal used for shading  [B, 512, 512, 3]
    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)
    gb_normal1 = gb_normal
    # gb_normal1 = gb_normal @ normal_rotate[:,None,...] # We randomly rotate the normals to change the color gamut of nomral at the same angle. We find this help to deform the shape
    gb_normal1 = torch.einsum('bij,bhwj->bhwi', normal_rotate, gb_normal)
    # buffers['normal'] = gb_normal1
    buffers['normal'] = gb_normal

    if bsdf == 'normal':
        shaded_col = gb_normal1
        if if_flip_the_normal:
            shaded_col[...,0][shaded_col[...,0]>0]= shaded_col[...,0][shaded_col[...,0]>0]*(-1) # Flip the x-axis positive half-axis of Normal. We find this process helps to alleviate the Janus problem.
        shaded_col = shaded_col*0.5 + 0.5
        buffers['albedo'] = shaded_col
        buffers['color'] = shaded_col
    elif bsdf == 'albedo':
        assert material is not None

        xyzs = gb_pos.view(-1, 3)

        outputs = material['color_func'](xyzs[mask], material['predict_class'])

        shaded_col = torch.zeros_like(xyzs)
        shaded_col[mask] = outputs['albedo']
        buffers['albedo'] = shaded_col.view(B, h, w, 3)
        buffers['color'] = shaded_col.view(B, h, w, 3)

        if material['predict_class']:
            pred_class = torch.zeros((xyzs.shape[0], material['num_classes']), dtype=torch.float32).to(xyzs.device)
            pred_class[mask] = outputs['pred_class'].float()
            buffers['pred_class'] = pred_class.view(B, h, w, material['num_classes'])
    else: # 'svbrdf'
        assert material is not None
        
        xyzs = gb_pos.view(-1, 3)

        albedo = torch.zeros_like(xyzs)
        color = torch.zeros_like(xyzs)
        spec_color = torch.zeros_like(xyzs)

        sg_envmap_material = material['svbrdf_func'](xyzs[mask], material['predict_class'])

        if sg_envmap_material['sg_specular_reflectance'].shape[-1] == 1:
            sg_envmap_material['sg_specular_reflectance'] = sg_envmap_material['sg_specular_reflectance'].repeat(1,3)

        if material['predict_class']:
            pred_class = torch.zeros((xyzs.shape[0], material['num_classes']), dtype=torch.float32).to(xyzs.device)
            pred_class[mask] = sg_envmap_material['pred_class'].float() # only foreground
            buffers['pred_class'] = pred_class.view(B, h, w, material['num_classes'])

            if material['material_offset']:
                if material['soft_material']:
                    weight = torch.softmax(sg_envmap_material['pred_class'].detach(), 1)
                    base_roughness = (material['RegionMaterials'].roughness[None] * weight).sum(-1)
                    base_specular = (material['RegionMaterials'].specular[None] * weight).sum(-1)
                else:
                    labels = sg_envmap_material['pred_class'].argmax(-1) #.detach()
                    base_roughness = material['RegionMaterials'].roughness[labels]
                    base_specular = material['RegionMaterials'].specular[labels]
                sg_envmap_material['sg_specular_reflectance'] = base_specular[...,None] + material['material_offset_ratio'] * sg_envmap_material['sg_specular_reflectance']
                sg_envmap_material['sg_roughness'] = base_roughness[...,None] + material['material_offset_ratio'] * sg_envmap_material['sg_roughness']

        viewdirs = util.safe_normalize(view_pos - gb_pos).view(-1,3)[mask]

        normal_ = gb_normal.view(-1, 3)[mask]
        # if material['normal_offset']: # XXX
            # normal_ = util.safe_normalize(normal_.detach() + sg_envmap_material['delta_normal']) # fixme: should calcualte this in sph space!
        sg_ret = render_with_sg(
            lgtSGs=sg_envmap_material['sg_lgtSGs'],
            specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
            roughness=sg_envmap_material['sg_roughness'],
            diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
            normal=normal_, viewdirs=viewdirs.view(-1,3))

        albedo[mask] = sg_envmap_material['sg_diffuse_albedo'].float()
        color[mask] = sg_ret['sg_rgb'].float()
        spec_color[mask] = sg_ret['sg_specular_rgb'].float()
        buffers['spec_color'] = spec_color.view(B, h, w, 3)
        buffers['albedo'] = albedo.view(B, h, w, 3)
        buffers['color'] = color.view(B, h, w, 3)

    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        normal_rotate,
        mode,
        if_flip_the_normal,
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast    #[u,v,z,triangle_id]
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int()) 
    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :] 
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :] 
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :] 
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0)) 
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3) #[10688,3] 三角面片每个顶点的法线的索引
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())
    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents
    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################

    mask = (rast[..., 3:] > 0).view(-1).detach()
    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, lgt, mesh.material, bsdf, normal_rotate, mode, if_flip_the_normal, mask)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1, # fixed!!!
        msaa        = False,
        background  = None, 
        bsdf        = None,
        normal_rotate = None,
        mode = 'geometry_modeling',
        if_flip_the_normal = False,
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias, cat_alpha):
        accum = background
        for buffers, rast in reversed(layers):
            # alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:] # [1,512,512,1] 保留有物体的像素的alpha为1，没有物体的像素alpha为0
            # accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha) #[1,512,512,4] 最后一个通道是alpha通道，若像素有物体则为1，无物体则为0  outi=starti+weighti×(endi−starti)
            alpha = (rast[..., -1:] > 0).float()
            if cat_alpha:
                accum = torch.lerp(accum, torch.cat((buffers[key], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            else:
                accum = torch.lerp(accum, buffers[key], alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)
    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)
    v_pos_clip = v_pos_clip.cuda()
    mesh.t_pos_idx = mesh.t_pos_idx.cuda()
    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, normal_rotate, mode, if_flip_the_normal), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key != 'pred_class':
            if key == 'normal':
                accum = composite_buffer(key, layers, background, True, True)
            else:
                accum = composite_buffer(key, layers, background[...,:-1], True, False)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False, False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)
