import numpy
import nptyping


def build_topology(
        vtx2xy: nptyping.NDArray[nptyping.Shape["*, *"], nptyping.Float]):
    if vtx2xy.shape[1] == 2:
        from del_msh.del_msh import kdtree_build_2d
        return kdtree_build_2d(vtx2xy)
    else:
        assert False


def build_edge(tree: numpy.ndarray, vtx2xy: numpy.ndarray):
    vmin = vtx2xy.min(axis=0)
    vmax = vtx2xy.max(axis=0)
    if vtx2xy.shape[1] == 2:
        from del_msh.del_msh import kdtree_edge_2d
        return kdtree_edge_2d(tree, vtx2xy, vmin, vmax)
    else:
        assert False
