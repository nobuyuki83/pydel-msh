use numpy::{IntoPyArray,
            PyArray3,
            PyReadonlyArray2};
use pyo3::{types::PyModule, PyResult, Python};
pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(unidex_vertex_attribute_for_triangle_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn unidex_vertex_attribute_for_triangle_mesh<'a>(
    py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>) -> &'a PyArray3<f32> {
    let num_val = vtx2xyz.shape()[1];
    let tri2xyz = del_msh::unindex::unidex_vertex_attribute_for_triangle_mesh(
        tri2vtx.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
        num_val);
    numpy::ndarray::Array3::from_shape_vec(
        (tri2xyz.len() / (3 * num_val), 3, num_val), tri2xyz).unwrap()
        .into_pyarray(py)
}