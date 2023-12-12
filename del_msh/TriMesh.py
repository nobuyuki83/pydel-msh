import typing
#
import numpy
from nptyping import Shape, NDArray, UInt, Float


def edge2vtx(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        num_vtx: int) \
        -> NDArray[Shape["*, 2"], UInt]:
    """
    compute line mesh connectivity from a triangle connectivity
    :param tri2vtx: triangle mesh connectivity
    :param num_vtx: number of vertex
    :return: 2D numpy.ndarray showing the vertex index for each edge
    """
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    from .del_msh import edge2vtx_uniform_mesh
    return edge2vtx_uniform_mesh(tri2vtx, num_vtx)


def vtx2vtx(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        num_vtx: int) \
        -> (NDArray[Shape["*"], UInt], NDArray[Shape["*"], UInt]):
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    from .del_msh import vtx2vtx_trimesh
    return vtx2vtx_trimesh(tri2vtx, num_vtx)


def tri2tri(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        num_vtx: int) \
        -> NDArray[Shape["*, 3"], UInt]:
    from .del_msh import elem2elem_uniform_mesh_polygon_indexing
    return elem2elem_uniform_mesh_polygon_indexing(tri2vtx, num_vtx)


def tri2distance(
        idx_tri: int,
        tri2tri: NDArray[Shape["*, 3"], UInt]) -> NDArray[Shape["*"], UInt]:
    from .del_msh import topological_distance_on_uniform_mesh
    return topological_distance_on_uniform_mesh(idx_tri, tri2tri)


# above: topology
# ------------------------
# below: primitive

def torus(
        major_radius=1.0,
        minor_radius=0.4,
        ndiv_major_radius=32,
        ndiv_minor_radius=32) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    from .del_msh import torus_meshtri3
    return torus_meshtri3(major_radius, minor_radius, ndiv_major_radius, ndiv_minor_radius)


def capsule(
        radius: float = 1.0,
        height: float = 1.,
        ndiv_theta: int = 32,
        ndiv_height: int = 32,
        ndiv_longtitude: int = 32) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    from .del_msh import capsule_meshtri3
    return capsule_meshtri3(radius, height, ndiv_theta, ndiv_longtitude, ndiv_height)


def cylinder(
        radius: float = 1.,
        height: float = 1.,
        ndiv_theta: int = 32,
        ndiv_height: int = 8) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    from .del_msh import cylinder_closed_end_meshtri3
    return cylinder_closed_end_meshtri3(radius, height, ndiv_theta, ndiv_height)


def sphere(
        radius: float = 1.,
        ndiv_latitude: int = 32,
        ndiv_longtitude: int = 32) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    from .del_msh import sphere_meshtri3
    return sphere_meshtri3(radius, ndiv_latitude, ndiv_longtitude)


def hemisphere(
        radius: float = 1.,
        ndiv_longtitude: int = 32,
        ndiv_latitude: int = 32) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    assert ndiv_longtitude > 0
    assert ndiv_latitude > 2
    from .del_msh import trimesh3_hemisphere_zup
    return trimesh3_hemisphere_zup(radius, ndiv_longtitude, ndiv_latitude)


# above: primitive
# ------------------------------
# below: misc

def load_wavefront_obj(
        path_file: str,
        is_centerize=False,
        normalized_size: typing.Optional[float] = None):
    from .del_msh import load_wavefront_obj_as_triangle_mesh
    tri2vtx, vtx2xyz = load_wavefront_obj_as_triangle_mesh(path_file)
    if is_centerize:
        vtx2xyz[:] -= (vtx2xyz.max(axis=0) + vtx2xyz.min(axis=0)) * 0.5
    if type(normalized_size) == float:
        vtx2xyz *= normalized_size / (vtx2xyz.max(axis=0) - vtx2xyz.min(axis=0)).max()
    return tri2vtx, vtx2xyz


def unindexing(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz: NDArray[Shape["*, 3"], Float]) \
        -> NDArray[Shape["*, *, *"], Float]:
    from .del_msh import unidex_vertex_attribute_for_triangle_mesh
    return unidex_vertex_attribute_for_triangle_mesh(tri2vtx, vtx2xyz)


def tri2area(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz: NDArray[Shape["*, *"], Float]) \
        -> NDArray[Shape["*"], Float]:
    """
    Areas of the triangles in a 2D/3D mesh.
    :param tri2vtx:
    :param vtx2xyz:
    :return:
    """
    assert vtx2xyz.shape[1] == 2 or vtx2xyz.shape[2] == 3, "the dimension should be 2 or 3"
    from .del_msh import areas_of_triangles_of_mesh
    return areas_of_triangles_of_mesh(tri2vtx, vtx2xyz)


def merge(
        tri2vtx0: NDArray[Shape["*, 3"], UInt],
        vtx2xyz0: NDArray[Shape["*, 3"], Float],
        tri2vtx1: NDArray[Shape["*, 3"], UInt],
        vtx2xyz1: NDArray[Shape["*, 3"], Float]):
    num_vtx0 = vtx2xyz0.shape[0]
    tri2vtx = numpy.vstack([tri2vtx0, tri2vtx1 + num_vtx0])
    vtx2xyz = numpy.vstack([vtx2xyz0, vtx2xyz1])
    return tri2vtx, vtx2xyz


# above: property
# ---------------------
# below: search

def first_intersection_ray(
        src,
        dir,
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz: NDArray[Shape["*, 3"], Float]) -> [NDArray[Shape["3"], Float], int]:
    """
    compute first intersection of a ray against a triangle mesh
    :param src: source of ray
    :param dir: direction of ray (can be un-normalized vector)
    :param tri2vtx: triangle index
    :param vtx2xyz: vertex positions
    :return: position and triangle index
    """
    from .del_msh import first_intersection_ray_meshtri3
    return first_intersection_ray_meshtri3(src, dir, tri2vtx, vtx2xyz)


def pick_vertex(
        src,
        dir,
        tri2vtx,
        vtx2xyz):
    from .del_msh import pick_vertex_meshtri3
    return pick_vertex_meshtri3(src, dir, tri2vtx, vtx2xyz)


# above: search
# ------------------------
# below: sampling

def position(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz: numpy.ndarray,
        idx_tri: int, r0: float, r1: float) -> numpy.ndarray:
    i0 = tri2vtx[idx_tri][0]
    i1 = tri2vtx[idx_tri][1]
    i2 = tri2vtx[idx_tri][2]
    p0 = vtx2xyz[i0]
    p1 = vtx2xyz[i1]
    p2 = vtx2xyz[i2]
    return r0 * p0 + r1 * p1 + (1. - r0 - r1) * p2


def sample(cumsum_area: numpy.ndarray, r0: float, r1: float):
    from .del_msh import sample_uniformly_trimesh
    return sample_uniformly_trimesh(cumsum_area, r0, r1)


def sample_many(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xy, num_sample: int) -> numpy.ndarray:
    import random
    tri2area_ = tri2area(tri2vtx, vtx2xy)
    cumsum_area = numpy.cumsum(numpy.append(numpy.zeros(1, dtype=numpy.float32), tri2area_))
    num_dim = vtx2xy.shape[1]
    xys = numpy.zeros([num_sample, num_dim], numpy.float32)
    for i in range(num_sample):
        smpl_i = sample(cumsum_area, random.random(), random.random())
        xys[i] = position(tri2vtx, vtx2xy, *smpl_i)
    return xys


# --------------------------------------

def merge_hessian_mesh_laplacian(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz,
        row2idx,
        idx2col,
        row2val, idx2val):
    from .del_msh import merge_hessian_mesh_laplacian_on_trimesh
    merge_hessian_mesh_laplacian_on_trimesh(
        tri2vtx, vtx2xyz,
        row2idx, idx2col,
        row2val, idx2val)


# ------------------------------------

def bvh_aabb(
        tri2vtx: NDArray[Shape["*, 3"], UInt],
        vtx2xyz: NDArray[Shape["*, 3"], Float]) \
        -> (NDArray[Shape["*, 3"], UInt], NDArray[Shape["*, 3"], Float]):
    from .del_msh import build_bvh_topology
    bvhnodes = build_bvh_topology(tri2vtx, vtx2xyz)
    aabbs = numpy.zeros((bvhnodes.shape[0], 6), dtype=numpy.float32)
    from .del_msh import build_bvh_geometry_aabb
    build_bvh_geometry_aabb(aabbs, bvhnodes, tri2vtx, vtx2xyz)
    return bvhnodes, aabbs
