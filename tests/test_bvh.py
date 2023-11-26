import numpy
from del_msh import TriMesh
def test_01():
    tri2vtx, vtx2xyz = TriMesh.sphere(1., 8, 4)
    from del_msh.del_msh import build_bvh_topology
    bvhnodes = build_bvh_topology(tri2vtx, vtx2xyz)
    print(bvhnodes)
    print(bvhnodes.shape)
    aabbs = numpy.zeros((bvhnodes.shape[0], 6), dtype=numpy.float32)
    from del_msh.del_msh import build_bvh_geometry_aabb
    build_bvh_geometry_aabb(aabbs, bvhnodes, tri2vtx, vtx2xyz)
    print(aabbs.shape)
    print(aabbs[0])