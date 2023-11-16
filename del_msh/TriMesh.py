import pathlib, typing

import numpy.typing


def edges(tri2vtx: numpy.typing.ArrayLike, num_vtx: int):
    assert len(tri2vtx.shape) == 2
    from .del_msh import edges_of_uniform_mesh
    return edges_of_uniform_mesh(tri2vtx, num_vtx)


def torus(major_radius=1.0, minor_radius=0.4, ndiv_major_radius=32, ndiv_minor_radius=32):
    from .del_msh import torus_meshtri3
    return torus_meshtri3(major_radius, minor_radius, ndiv_major_radius, ndiv_minor_radius)


def capsule(radius=1.0, height=1., ndiv_theta=32, ndiv_height=32, ndiv_longtitude=32):
    from .del_msh import capsule_meshtri3
    return capsule_meshtri3(radius, height, ndiv_theta, ndiv_longtitude, ndiv_height)


def cylinder(radius=1., height=1., ndiv_theta=32, ndiv_height=8):
    from .del_msh import cylinder_closed_end_meshtri3
    return cylinder_closed_end_meshtri3(radius, height, ndiv_theta, ndiv_height)


def sphere(radius=1., ndiv_latitude=32, ndiv_longtitude=32):
    from .del_msh import sphere_meshtri3
    return sphere_meshtri3(radius, ndiv_latitude, ndiv_longtitude)


def load_wavefront_obj(
        path_file: pathlib.Path,
        is_centerize=False,
        normalized_size: typing.Optional[float] = None):
    from .del_msh import load_wavefront_obj_as_triangle_mesh
    tri2vtx, vtx2xyz = load_wavefront_obj_as_triangle_mesh(str(path_file))
    if is_centerize:
        vtx2xyz[:] -= (vtx2xyz.max(axis=0) + vtx2xyz.min(axis=0)) * 0.5
    if type(normalized_size) == float:
        vtx2xyz *= normalized_size / (vtx2xyz.max(axis=0) - vtx2xyz.min(axis=0)).max()
    return tri2vtx, vtx2xyz


def unindexing(tri2vtx, vtx2xyz):
    from .del_msh import unidex_vertex_attribute_for_triangle_mesh
    return unidex_vertex_attribute_for_triangle_mesh(tri2vtx, vtx2xyz)


def triangle_adjacency(tri2vtx, num_vtx: int):
    from .del_msh import elsuel_uniform_mesh_polygon
    return elsuel_uniform_mesh_polygon(tri2vtx, num_vtx)


def topological_distance_of_tris(idx_tri, tri2tri):
    from .del_msh import topological_distance_on_uniform_mesh
    return topological_distance_on_uniform_mesh(idx_tri, tri2tri)


def areas(tri2vtx, vtx2xyz):
    from .del_msh import areas_of_triangles_of_mesh
    return areas_of_triangles_of_mesh(tri2vtx, vtx2xyz)


def position(
        tri2vtx: numpy.typing.NDArray, vtx2xyz: numpy.typing.NDArray,
        idx_tri: int, r0: float, r1: float):
    i0 = tri2vtx[idx_tri][0]
    i1 = tri2vtx[idx_tri][1]
    i2 = tri2vtx[idx_tri][2]
    p0 = vtx2xyz[i0]
    p1 = vtx2xyz[i1]
    p2 = vtx2xyz[i2]
    return r0 * p0 + r1 * p1 + (1. - r0 - r1) * p2

def sample(cumsum_area: numpy.typing.NDArray, r0: float, r1: float):
    from .del_msh import sample_uniformly_trimesh
    return sample_uniformly_trimesh(cumsum_area, r0, r1)


def sample_many(tri2vtx, vtx2xy, N:int):
    import random
    tri2area = areas(tri2vtx, vtx2xy)
    cumsum_area = numpy.cumsum(numpy.append(0., tri2area)).astype(numpy.float32)
    xys = []
    for i in range(N):
        smpl_i = sample(cumsum_area, random.random(), random.random())
        pos_i = position(tri2vtx, vtx2xy, *smpl_i)
        xys.append(pos_i.tolist())
    return numpy.array(xys)