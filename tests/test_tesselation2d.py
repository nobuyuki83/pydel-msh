import numpy
from del_msh import PolyLoop

def test_0():
    vtxi2xyi = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.1],
        [0, 0.1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(
        vtxi2xyi, resolution_edge=0.11, resolution_face=-1)