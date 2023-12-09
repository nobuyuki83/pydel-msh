import numpy


def tesselation2d(vtx2xy):
    from .del_msh import tesselation2d
    return tesselation2d(vtx2xy)


def area2(vtx2xy: numpy.ndarray) -> float:
    if vtx2xy.dtype == numpy.float32:
        from .del_msh import polyloop2_area_f32
        return polyloop2_area_f32(vtx2xy)
    elif vtx2xy.dtype == numpy.float64:
        from .del_msh import polyloop2_area_f64
        return polyloop2_area_f64(vtx2xy)
