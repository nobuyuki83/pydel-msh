use numpy::{IntoPyArray,
            PyArray2, PyArray1,
            PyReadonlyArrayDyn};
use pyo3::{types::PyModule, PyResult, Python};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(unify_two_indices_of_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(unify_two_indices_of_polygon_mesh, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn unify_two_indices_of_triangle_mesh<'a>(
    py: Python<'a>,
    tri2vtxa: PyReadonlyArrayDyn<'a, usize>,
    tri2vtxb: PyReadonlyArrayDyn<'a, usize>) -> (&'a PyArray2<usize>, &'a PyArray1<usize>, &'a PyArray1<usize>) {
    let (tri2uni, uni2vtxxyz, uni2vtxuv)
        = del_msh::unify_index::unify_two_indices_of_triangle_mesh(
        tri2vtxa.as_slice().unwrap(),
        tri2vtxb.as_slice().unwrap());
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2uni.len() / 3, 3), tri2uni).unwrap().into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxxyz).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxuv).into_pyarray(py),
    )
}

#[pyo3::pyfunction]
fn unify_two_indices_of_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArrayDyn<'a, usize>,
    idx2vtxa: PyReadonlyArrayDyn<'a, usize>,
    idx2vtxb: PyReadonlyArrayDyn<'a, usize>) -> (&'a PyArray1<usize>, &'a PyArray1<usize>, &'a PyArray1<usize>) {
    let (idx2uni, uni2vtxa, uni2vtxb)
        = del_msh::unify_index::unify_two_indices_of_polygon_mesh(
        elem2idx.as_slice().unwrap(),
        idx2vtxa.as_slice().unwrap(),
        idx2vtxb.as_slice().unwrap());
    (
        numpy::ndarray::Array1::from_vec(idx2uni).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxa).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(uni2vtxb).into_pyarray(py)
    )
}