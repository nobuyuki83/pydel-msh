from .del_msh import*
import numpy

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


def extract_submesh(tri2vtx, tri2bool, vtx2xyz):
    assert tri2vtx.shape[0] == tri2bool.shape[0]
    tri_new2vtx_old = tri2vtx[tri2bool]
    vtx_new2vtx_old = list(set((tri_new2vtx_old[:][:]).flatten()))
    num_vtx_new = len(vtx_new2vtx_old)
    vtx_new2xyz = numpy.zeros((num_vtx_new, 3), dtype=numpy.float32)
    vtx_new2xyz[:][:] = vtx2xyz[vtx_new2vtx_old[:]][:]
    vtx_old2vtx_new = numpy.full(shape=vtx2xyz.shape[0],
                                 fill_value=numpy.iinfo(numpy.uint64).max,
                                 dtype=numpy.uint64)
    for vtx_new, vtx_old in enumerate(vtx_new2vtx_old):
        vtx_old2vtx_new[vtx_old] = vtx_new
    tri_new2vtx_new = numpy.zeros_like(tri_new2vtx_old)
    tri_new2vtx_new[:][:] = vtx_old2vtx_new[tri_new2vtx_old[:][:]]
    return tri_new2vtx_new, vtx_new2xyz, vtx_new2vtx_old