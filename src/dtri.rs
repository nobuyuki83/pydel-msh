
pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(tesselation2d, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn tesselation2d<'a>(
    py: pyo3::Python<'a>,
    vtx2xy_in: numpy::PyReadonlyArray2<'a, f32>)
    -> (&'a numpy::PyArray2<usize>, &'a numpy::PyArray2<f32>)
{
    let num_vtx = vtx2xy_in.shape()[0];
    let loop2idx = vec!(0, num_vtx);
    let idx2vtx = Vec::<usize>::from_iter(0..num_vtx);
    type Vec2 = nalgebra::Vector2<f32>;
    let mut vtx2xy = Vec::<Vec2>::new();
    for ivtx in 0..num_vtx {
        let v0 = Vec2::new(
            *vtx2xy_in.get((ivtx, 0)).unwrap(),
            *vtx2xy_in.get((ivtx, 1)).unwrap());
        vtx2xy.push(v0);
    }
    let mut tri2pnt = Vec::<del_dtri::topology::DynamicTriangle>::new();
    let mut pnt2tri = Vec::<del_dtri::topology::DynamicVertex>::new();
    del_dtri::mesher2::meshing_single_connected_shape2(
        &mut pnt2tri, &mut vtx2xy, &mut tri2pnt,
        &loop2idx, &idx2vtx);
    let mut vtx2xy_out = Vec::<f32>::new();
    for xy in vtx2xy.iter() {
        vtx2xy_out.push(xy.x);
        vtx2xy_out.push(xy.y);
    }
    let mut tri2vtx_out = Vec::<usize>::new();
    for pnt in tri2pnt.iter() {
        tri2vtx_out.push(pnt.v[0]);
        tri2vtx_out.push(pnt.v[1]);
        tri2vtx_out.push(pnt.v[2]);
    }
    use numpy::IntoPyArray;
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx_out.len() / 3, 3), tri2vtx_out).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (vtx2xy_out.len() / 2, 2), vtx2xy_out).unwrap().into_pyarray(py),
    )
}