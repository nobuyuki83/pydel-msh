import numpy
from nptyping import NDArray, Shape, Float, UInt

def edges_of_aabb(aabb: NDArray[Shape["*, *"], Float])\
        -> NDArray[Shape["*, *, *"], Float]:
    num_dim = aabb.shape[1] // 2
    assert aabb.shape[1] == num_dim * 2
    num_aabb = aabb.shape[0]
    if num_dim == 3:
        edge2node2xyz = numpy.zeros((num_aabb, 12, 2, num_dim), dtype=aabb.dtype)
        edge2node2xyz[:, 0, 0, :] = edge2node2xyz[:, 3, 1, :] = edge2node2xyz[:, 8, 0, :] = aabb[:, [0, 1, 2]]
        edge2node2xyz[:, 0, 1, :] = edge2node2xyz[:, 1, 0, :] = edge2node2xyz[:, 9, 0, :] = aabb[:, [3, 1, 2]]
        edge2node2xyz[:, 2, 1, :] = edge2node2xyz[:, 3, 0, :] = edge2node2xyz[:, 10, 0, :] = aabb[:, [0, 4, 2]]
        edge2node2xyz[:, 1, 1, :] = edge2node2xyz[:, 2, 0, :] = edge2node2xyz[:, 11, 0, :] = aabb[:, [3, 4, 2]]
        edge2node2xyz[:, 4, 0, :] = edge2node2xyz[:, 7, 1, :] = edge2node2xyz[:, 8, 1, :] = aabb[:, [0, 1, 5]]
        edge2node2xyz[:, 4, 1, :] = edge2node2xyz[:, 5, 0, :] = edge2node2xyz[:, 9, 1, :] = aabb[:, [3, 1, 5]]
        edge2node2xyz[:, 6, 1, :] = edge2node2xyz[:, 7, 0, :] = edge2node2xyz[:, 10, 1, :] = aabb[:, [0, 4, 5]]
        edge2node2xyz[:, 5, 1, :] = edge2node2xyz[:, 6, 0, :] = edge2node2xyz[:, 11, 1, :] = aabb[:, [3, 4, 5]]
        edge2node2xyz = edge2node2xyz.reshape(num_aabb * 12, 2, num_dim)
        return edge2node2xyz


def self_intersection_trimesh3(
        tri2vtx: NDArray[Shape["*, *"], UInt],
        vtx2xyz: NDArray[Shape["*, *"], Float],
        bvhnodes: NDArray[Shape["*, *"], UInt],
        aabbs: NDArray[Shape["*, *"], Float]):
    from .del_msh import bvh3_self_intersection_trimesh3
    return bvh3_self_intersection_trimesh3(tri2vtx, vtx2xyz, bvhnodes, aabbs)
