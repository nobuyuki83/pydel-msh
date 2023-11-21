pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(polyloop_area2, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn polyloop_area2(
    vtx2xy: numpy::PyReadonlyArray2<f32>) -> f32
{
    let vtx2xy = vtx2xy.as_slice().unwrap();
    del_msh::polyloop::area2(vtx2xy)
}