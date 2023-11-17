use numpy::{PyReadonlyArray1};
use pyo3::{types::PyModule, PyResult, Python};
pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(sample_uniformly_trimesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn sample_uniformly_trimesh(
    _py: Python,
    cumsum_tri2area: PyReadonlyArray1<f32>,
    val01_a: f32,
    val01_b: f32) -> (usize, f32, f32) {
    del_msh::sampling::sample_uniformly_trimesh(
        cumsum_tri2area.as_slice().unwrap(),
        val01_a, val01_b)
}