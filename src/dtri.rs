
pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(tesselation2d, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn tesselation2d<'a>(
    py: pyo3::Python<'a>,
    vtx2xy_in: numpy::PyReadonlyArray2<'a, f32>,
    resolution_edge: f32,
    resolution_face: f32)
    -> (&'a numpy::PyArray2<usize>, &'a numpy::PyArray2<f32>)
{
    let num_vtx = vtx2xy_in.shape()[0];
    let vtx2xy_in = vtx2xy_in.as_slice().unwrap();
    let mut loop2idx = vec!(0, num_vtx);
    let mut idx2vtx = Vec::<usize>::from_iter(0..num_vtx);
    type Vec2 = nalgebra::Vector2<f32>;
    let mut vtx2xy = vtx2xy_in.chunks(2).map(|v| Vec2::new(v[0],v[1]) ).collect();
    //
    if resolution_edge > 0. {
        del_msh::polyloop::resample_multiple_loops_remain_original_vtxs(
            &mut loop2idx, &mut idx2vtx, &mut vtx2xy, resolution_edge);
    }
    //
    let mut tri2pnt = Vec::<del_dtri::topology::DynamicTriangle>::new();
    let mut pnt2tri = Vec::<del_dtri::topology::DynamicVertex>::new();
    del_dtri::mesher2::meshing_single_connected_shape2(
        &mut pnt2tri, &mut vtx2xy, &mut tri2pnt,
        &loop2idx, &idx2vtx);
    // ----------------------------------------
    if resolution_face > 1.0e-10 {
        let nvtx = vtx2xy.len();
        let mut vtx2flag = vec!(0;nvtx);
        let mut tri2flag = vec!(0;tri2pnt.len());
        del_dtri::mesher2::meshing_inside(
            &mut pnt2tri, &mut tri2pnt, &mut vtx2xy,
            &mut vtx2flag, &mut tri2flag,
            nvtx, 0, resolution_face);
    }
    // ----------------------------------------
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