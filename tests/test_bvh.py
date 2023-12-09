import numpy
from del_msh import TriMesh, BVH


def test_01():
    tri2vtx, vtx2xyz = TriMesh.sphere(1., 8, 4)
    from del_msh.del_msh import build_bvh_topology
    bvhnodes = build_bvh_topology(tri2vtx, vtx2xyz)
    assert bvhnodes.shape[1] == 3
    aabbs = numpy.zeros((bvhnodes.shape[0], 6), dtype=numpy.float32)
    from del_msh.del_msh import build_bvh_geometry_aabb
    build_bvh_geometry_aabb(aabbs, bvhnodes, tri2vtx, vtx2xyz)
    assert bvhnodes.shape[0] == aabbs.shape[0]
    assert aabbs.shape[1] == 6
    assert numpy.linalg.norm(aabbs[0] - numpy.array([-1., -1., -1., 1., 1., 1])) < 1.0e-5
    edge2node2xyz, edge2tri = BVH.self_intersection_trimesh3(tri2vtx, vtx2xyz, bvhnodes, aabbs)