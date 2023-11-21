use numpy::{IntoPyArray,
            PyArray1,
            PyReadonlyArray2};
use pyo3::{types::PyModule, PyResult, Python};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(topological_distance_on_uniform_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn topological_distance_on_uniform_mesh<'a>(
    py: Python<'a>,
    ielm_ker: usize,
    elsuel: PyReadonlyArray2<'a, usize>) -> &'a PyArray1<usize> {
    let num_elem = elsuel.shape()[0];
    let elem2dist = del_msh::dijkstra::topological_distance_on_uniform_mesh(
        ielm_ker,
        elsuel.as_slice().unwrap(),
        num_elem);
    assert_eq!(elem2dist.len(), num_elem);
    numpy::ndarray::Array1::from_shape_vec(
        num_elem, elem2dist).unwrap().into_pyarray(py)
}