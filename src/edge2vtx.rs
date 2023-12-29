use numpy::IntoPyArray;

pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(edge2vtx_uniform_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(edge2vtx_polygon_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn edge2vtx_uniform_mesh<'a>(
    py: pyo3::Python<'a>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> &'a numpy::PyArray2<usize> {
    // TODO: make this function general to uniform mesh (currently only triangle mesh)
    let mshline = del_msh::edge2vtx::from_uniform_mesh_with_specific_edges(
        elem2vtx.as_slice().unwrap(), 3,
        &[0, 1, 1, 2, 2, 0], num_vtx);
    numpy::ndarray::Array2::from_shape_vec(
        (mshline.len() / 2, 2), mshline).unwrap().into_pyarray(py)
}

#[pyo3::pyfunction]
fn edge2vtx_polygon_mesh<'a>(
    py: pyo3::Python<'a>,
    elem2idx: numpy::PyReadonlyArray1<'a, usize>,
    idx2vtx: numpy::PyReadonlyArray1<'a, usize>,
    num_vtx: usize) -> &'a numpy::PyArray2<usize> {
    let mshline = del_msh::edge2vtx::from_polygon_mesh(
        elem2idx.as_slice().unwrap(), idx2vtx.as_slice().unwrap(), num_vtx);
    numpy::ndarray::Array2::from_shape_vec(
        (mshline.len() / 2, 2), mshline).unwrap().into_pyarray(py)
}