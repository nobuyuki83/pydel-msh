"""
Functions for polygon mesh (e.g., mixture of triangles, quadrilaterals, pentagons) in 2D and 3D
"""

import numpy


def triangles(pelem2pidx: numpy.ndarray, pidx2vtxxyz: numpy.ndarray) -> numpy.ndarray:
    from .del_msh import triangles_from_polygon_mesh
    return triangles_from_polygon_mesh(pelem2pidx, pidx2vtxxyz)


def extract(elem2idx: numpy.ndarray, idx2vtx: numpy.ndarray, elem2bool: numpy.ndarray):
    from .del_msh import extract_flagged_polygonal_element
    return extract_flagged_polygonal_element(elem2idx, idx2vtx, elem2bool)


def edges(elem2idx: numpy.ndarray, idx2vtx: numpy.ndarray, num_vtx: int):
    from .del_msh import edge2vtx_polygon_mesh
    return edge2vtx_polygon_mesh(elem2idx, idx2vtx, num_vtx)
