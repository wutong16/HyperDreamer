import kaolin as kal
import torch
import copy
import os

import editing.src.utils as util

device_cuda = 'cuda'

class Mesh:
    def __init__(self, vertices=None, faces=None, vn=None, face_n=None, vt=None, ft=None, v_tng=None, t_tng_idx=None, material=None, base=None):
        # from https://github.com/threedle/text2mesh
        
        """
        顶点索引:         以f v1 v2 v3
        顶点法线索引:     以f v1//vn1 v2//vn2 v3//vn3 
        顶点纹理法线索引: 以f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 
        """
        
        self.vertices = vertices    #v
        self.vt = vt                #vt
        self.vn = vn                #vn
        
        self.faces = faces          #f_v
        self.ft = ft                #f_vt
        self.face_n = face_n        #f_vn
        self.v_tng = v_tng
        self.t_tng_idx = t_tng_idx
        self.material = material
        if base is not None:
            self.copy_none(base)
        
        if vertices is not None and faces is not None:
            self.normals, self.face_area = self.calculate_face_normals(self.vertices, self.faces)
        if self.face_n is None:
            self.face_n = self.normals
        self.face_n = self.face_n.to(torch.int64)

        if self.vn is None:
            self.vn = self.auto_normals(self.vertices, self.faces)
        if v_tng is not None:
            self.v_tng = v_tng
        if t_tng_idx is not None:
            self.t_tng_idx = t_tng_idx


        if self.vertices is not None:
            self.vertices = self.vertices.to(device_cuda)
        if self.vt is not None:
            self.vt = self.vt.to(device_cuda)
        if self.vn is not None:
            self.vn = self.vn.to(device_cuda)
        if self.faces is not None:
            self.faces = self.faces.to(device_cuda)
        if self.ft is not None:
            self.ft = self.ft.to(device_cuda)
        if self.face_n is not None:
            self.face_n = self.face_n.to(device_cuda)
        
        if self.v_tng is not None:
            self.v_tng = self.v_tng.to(device_cuda)
        if self.t_tng_idx is not None:
            self.t_tng_idx = self.t_tng_idx.to(device_cuda)
            
    def copy_none(self, other):
        if self.vertices is None:
            self.vertices = other.vertices
        if self.faces is None:
            self.faces = other.faces
        if self.vn is None:
            self.vn = other.vn
        if self.face_n is None:
            self.face_n = other.face_n
        if self.vt is None:
            self.vt = other.vt
        if self.ft is None:
            self.ft = other.ft
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material    

        
    def clone(self):
        out = Mesh(base=self)
        if out.vertices is not None:
            out.vertices = out.vertices.clone().detach()
        if out.faces is not None:
            out.faces = out.faces.clone().detach()
        if out.vn is not None:
            out.vn = out.vn.clone().detach()
        if out.face_n is not None:
            out.face_n = out.face_n.clone().detach()
        if out.vt is not None:
            out.vt = out.vt.clone().detach()
        if out.ft is not None:
            out.ft = out.ft.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out
    
    @staticmethod
    def calculate_face_normals(vertices: torch.Tensor, faces: torch.Tensor):
        """
        calculate per face normals from vertices and faces
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=-1)
        twice_area = torch.norm(n, dim=-1)
        n = n / twice_area[:, None]
        return n, twice_area / 2
    
    ######################################################################################
    # Simple smooth vertex normal computation
    ######################################################################################
    def auto_normals(self, vertices, faces):

        i0 = faces[:, 0]
        i1 = faces[:, 1]
        i2 = faces[:, 2]

        v0 = vertices[i0, :]
        v1 = vertices[i1, :]
        v2 = vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(vertices)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
        v_nrm = util.safe_normalize(v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))
        v_n = v_nrm
        return v_n

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.std(torch.norm(verts, p=2, dim=1))
        verts /= scale
        mesh.vertices = verts
        return mesh

    def normalize_mesh(self,inplace=False, target_scale=1, dy=0):
        mesh = self if inplace else copy.deepcopy(self)
        verts_raw = mesh.vertices
        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts = verts - center
        scale = torch.max(torch.norm(verts, p=2, dim=1))
        verts = verts / scale
        verts *= target_scale
        verts[:, 1] += dy
        mesh.vertices = verts
        """
        vt = mesh.vt
        vt_normalized = (vt - center[:2]) / scale
        vt_normalized *= target_scale
        mesh.vt = vt_normalized
        """
        
        return mesh


def read_obj(filename):
    vertices, vt, vn = [],[],[]
    faces, ft, face_n = [],[],[]          

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        prefix = line.split()[0].lower()
        
        if prefix == 'usemtl': # Track used materials
            mat_name = line.split()[1]
        
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(vt) for vt in line.split()[1:]]
            vt.append([val[0], val[1]])
        elif prefix == 'vn':
            vn.append([float(vn) for vn in line.split()[1:]])
            
            
        elif prefix == 'f': # Parse face
            
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            nnv = len(vv)

            if nnv == 1:
                #print('obj face type: f v1 v2 v3')
                v0 = int(vv[0]) - 1
                t0 = -1
                n0 = -1
                for i in range(nv - 2): # Triangulate polygons
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = -1
                    n1 = -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = -1
                    n2 = -1

                    faces.append([v0, v1, v2])
                    ft.append([t0, t1, t2])
                    face_n.append([n0, n1, n2])
            elif nnv == 2:
                #print('obj face type: f v1/vn1 v2/vn2 v3/vn3')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1
                n0 = - 1
                for i in range(nv - 2): # Triangulate polygons
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1
                    n1 = - 1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1
                    n2 = -1

                    faces.append([v0, v1, v2])
                    ft.append([t0, t1, t2])
                    face_n.append([n0, n1, n2])
                
            elif nnv == 3:
                #print('obj face type: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                n0 = int(vv[2]) - 1 if vv[2] != "" else -1
                for i in range(nv - 2): # Triangulate polygons
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    n2 = int(vv[2]) - 1 if vv[2] != "" else -1

                    faces.append([v0, v1, v2])
                    ft.append([t0, t1, t2])
                    face_n.append([n0, n1, n2])
                
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device_cuda)
    vt = torch.tensor(vt, dtype=torch.float32, device=device_cuda) if len(vt) > 0 else None
    vn = torch.tensor(vn, dtype=torch.float32, device=device_cuda) if len(vn) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int64, device=device_cuda)
    ft = torch.tensor(ft, dtype=torch.int64, device=device_cuda) if vt is not None else None
    face_n = torch.tensor(face_n, dtype=torch.int64, device=device_cuda) if vn is not None else None
    return vertices, vt, vn, faces, ft, face_n, mat_name




def load_mesh(obj_path):
    vertices, vt, vn, faces, ft, face_n, mat_name = read_obj(obj_path)

    return Mesh(vertices=vertices, faces=faces, vn=vn, face_n=face_n, vt=vt, ft=ft), mat_name


######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.vertices, dim=0).values, torch.max(mesh.vertices, dim=0).values



######################################################################################
# Compute unique edge list from attribute/vertex index list
######################################################################################
def compute_edges(attr_idx, return_inverse=False):
    with torch.no_grad():
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

        # Eliminate duplicates and return inverse mapping
        return torch.unique(sorted_edges, dim=0, return_inverse=return_inverse)
    
    

######################################################################################
# Compute unique edge to face mapping from attribute/vertex index list
######################################################################################
def compute_edge_to_face_mapping(attr_idx, return_inverse=False):
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

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).to(device_cuda)

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).to(device_cuda)

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge
    


######################################################################################
# Align base mesh to reference mesh:move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.vertices - (vmax + vmin) / 2 # Center mesh on origin
        v_pos = v_pos * scale                  # Rescale to unit size

        return Mesh(v_pos, base=mesh)
    
    



######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(imesh):
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    imesh.ft = imesh.ft.long()
    
    for i in range(0,3):

        pos[i] = imesh.vertices[imesh.faces[:, i]]
        tex[i] = imesh.vt[imesh.ft[:, i]]

        vn_idx[i] = imesh.face_n[:, i]
    tangents = torch.zeros_like(imesh.vn)
    tansum   = torch.zeros_like(imesh.vn)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.vn) * imesh.vn)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.face_n, base=imesh)