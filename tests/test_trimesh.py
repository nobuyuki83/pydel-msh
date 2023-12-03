import math
import numpy
from del_msh import TriMesh


def test_01():
    tri2vtx, vtx2xyz = TriMesh.sphere()
    tri2node2xyz = TriMesh.unindexing(tri2vtx, vtx2xyz)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    vtx2idx, idx2vtx = TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0])
    tri2tri = TriMesh.tri2tri(tri2vtx, vtx2xyz.shape[0])
    tri2dist = TriMesh.tri2distance(0, tri2tri)
    assert tri2dist[0] == 0
    areas = TriMesh.tri2area(tri2vtx, vtx2xyz)
    assert math.fabs(areas.sum() - 4. * math.pi) < 0.1
    cumsum_areas = numpy.cumsum(numpy.append(numpy.zeros(1, dtype=numpy.float32), areas))
    sample = TriMesh.sample(cumsum_areas, 0.5, 0.1)
    samples2xyz = TriMesh.sample_many(tri2vtx, vtx2xyz, num_sample=1000)
    row2val = numpy.zeros([vtx2xyz.shape[0]], dtype=numpy.float64)
    idx2val = numpy.zeros([idx2vtx.shape[0]], dtype=numpy.float64)
    TriMesh.merge_hessian_mesh_laplacian(
        tri2vtx, vtx2xyz.astype(numpy.float64),
        vtx2idx, idx2vtx,
        row2val, idx2val)
