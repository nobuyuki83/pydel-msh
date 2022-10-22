
from .del_msh import*


def centerize_scale_3d_points(vtx_xyz):
    x0 = vtx_xyz[:,0].min()
    x1 = vtx_xyz[:,0].max()
    y0 = vtx_xyz[:,1].min()
    y1 = vtx_xyz[:,1].max()
    z0 = vtx_xyz[:,2].min()
    z1 = vtx_xyz[:,2].max()
    vtx_xyz[:,0] -= (x0+x1)*0.5
    vtx_xyz[:,1] -= (y0+y1)*0.5
    vtx_xyz[:,2] -= (z0+z1)*0.5
    scale = 1.0/max(x1-x0,y1-y0,z1-z0)
    return vtx_xyz * scale

