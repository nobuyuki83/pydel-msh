def triangles(pelem2pidx, pidx2vtxxyz):
    from .del_msh import triangles_from_polygon_mesh
    return triangles_from_polygon_mesh(pelem2pidx, pidx2vtxxyz)


def extract(elem2idx, idx2vtx, elem2bool):
    from .del_msh import extract_flagged_polygonal_element
    return extract_flagged_polygonal_element(elem2idx, idx2vtx, elem2bool)


def edges(elem2idx, idx2vtx, num_vtx):
    from .del_msh import edges_of_polygon_mesh
    return edges_of_polygon_mesh(elem2idx, idx2vtx, num_vtx)
