use numpy::ToPyArray;

pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(build_bvh_topology, m)?)?;
    m.add_function(wrap_pyfunction!(build_bvh_geometry_aabb, m)?)?;
    m.add_function( wrap_pyfunction!(bvh3_self_intersection_trimesh3, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn build_bvh_topology<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>)  -> &'a numpy::PyArray2<usize>
{
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let elem2elem = {
        let (vtx2idx, idx2elem)
            = del_msh::vtx2elem::from_uniform_mesh(tri2vtx, 3, vtx2xyz.len() / 3);
        let (face2jdx, jdx2node)
            = del_msh::elem2elem::face2node_of_polygon_element(3);
        del_msh::elem2elem::from_uniform_mesh_with_vtx2elem(
            tri2vtx, 3,
            &vtx2idx, &idx2elem,
            &face2jdx, &jdx2node)
    };
    let elem2center = del_msh::elem2center::from_uniform_mesh(
        tri2vtx, 3, vtx2xyz, 3);
    let bvhnodes = del_msh::bvh3::build_topology_for_uniform_mesh_with_elem2elem_elem2center(
        &elem2elem, 3, &elem2center);
    let a = numpy::PyArray1::<usize>::from_slice(_py, &bvhnodes);
    let a = a.reshape((bvhnodes.len()/3, 3)).unwrap();
    a
}

#[pyo3::pyfunction]
fn build_bvh_geometry_aabb<'a>(
    _py: pyo3::Python<'a>,
    mut aabbs: numpy::PyReadwriteArray2<'a, f32>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>)
{
    assert_eq!( bvhnodes.shape()[0], aabbs.shape()[0] );
    assert_eq!( bvhnodes.shape()[1], 3 );
    assert_eq!( aabbs.shape()[1], 6 );
    let num_noel = tri2vtx.shape()[1];
    let aabbs = aabbs.as_slice_mut().unwrap();
    let bvhnodes = bvhnodes.as_slice().unwrap();
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    del_msh::bvh3::build_geometry_aabb_for_uniform_mesh(
        aabbs,
        0, bvhnodes,
        tri2vtx,
        num_noel, vtx2xyz);
}

#[pyo3::pyfunction]
fn bvh3_self_intersection_trimesh3<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    aabbs: numpy::PyReadonlyArray2<'a, f32>)
    -> (&'a numpy::PyArray3<f32>, &'a numpy::PyArray2<usize>)
{
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let bvhnodes = bvhnodes.as_slice().unwrap();
    let aabbs = aabbs.as_slice().unwrap();
    let mut pairs = Vec::<del_msh::bvh3_intersection_self::IntersectingPair>::new();
    del_msh::bvh3_intersection_self::intersection_triangle_mesh_inside_branch(
        &mut pairs,
        tri2vtx, vtx2xyz,
        0, bvhnodes, aabbs);
    let mut edge2node2xyz = Vec::<f32>::new();
    let mut edge2tri = Vec::<usize>::new();
    for pair in pairs.iter() {
        edge2node2xyz.extend(pair.p0.iter());
        edge2node2xyz.extend(pair.p1.iter());
        edge2tri.push(pair.i_tri);
        edge2tri.push(pair.j_tri);
    }
    (
        numpy::ndarray::Array3::from_shape_vec(
            (edge2node2xyz.len()/6,2,3), edge2node2xyz).unwrap().to_pyarray(_py),
        numpy::ndarray::Array2::from_shape_vec(
            (edge2tri.len()/2,2), edge2tri).unwrap().to_pyarray(_py)
    )
}