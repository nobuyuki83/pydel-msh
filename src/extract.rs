use numpy::{IntoPyArray,
            PyArray1,
            PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{types::PyModule, PyResult, Python};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(extract_flagged_polygonal_element, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn extract<'a>(
    py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize,
    tri2tri_new: PyReadonlyArray1<'a, usize>,
    num_tri_new: usize) -> (&'a PyArray1<usize>, usize, &'a PyArray1<usize>)
{
    let (tri2vtx_new, num_vtx_new, vtx2vtx_new) = del_msh::extract::extract(
        tri2vtx.as_slice().unwrap(), num_vtx,
        tri2tri_new.as_slice().unwrap(), num_tri_new);
    (
        numpy::ndarray::Array1::from_vec(tri2vtx_new).into_pyarray(py),
        num_vtx_new,
        numpy::ndarray::Array1::from_vec(vtx2vtx_new).into_pyarray(py)
    )
}


#[pyo3::pyfunction]
fn extract_flagged_polygonal_element<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
    elem2flag: PyReadonlyArray1<'a, bool>) -> (&'a PyArray1<usize>, &'a PyArray1<usize>) {
    assert_eq!(elem2flag.len() + 1, elem2idx.len());
    let elem2flag = elem2flag.as_slice().unwrap();
    let elem2idx = elem2idx.as_slice().unwrap();
    let idx2vtx = idx2vtx.as_slice().unwrap();
    let (felem2jdx, jdx2vtx)
        = del_msh::extract::from_polygonal_mesh_array(elem2idx, idx2vtx, elem2flag);
    (
        numpy::ndarray::Array1::from_vec(felem2jdx).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(jdx2vtx).into_pyarray(py)
    )
}