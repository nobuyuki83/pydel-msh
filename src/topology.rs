use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2,
            PyArray2};
use pyo3::{Python, pyfunction, types::PyModule, PyResult, wrap_pyfunction};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edges_of_uniform_mesh, m)?)?;
    m.add_function( wrap_pyfunction!(edges_of_polygon_mesh, m)?)?;
    m.add_function( wrap_pyfunction!(triangles_from_polygon_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn edges_of_uniform_mesh<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> &'a PyArray2<usize> {
    let mshline = del_msh::line2vtx::from_sepecific_edges_of_uniform_mesh(
        &elem2vtx.as_slice().unwrap(), 3,
        &[0,1,1,2,2,0], num_vtx);
    numpy::ndarray::Array2::from_shape_vec(
        (mshline.len()/2,2), mshline).unwrap().into_pyarray(py)
}

#[pyfunction]
fn edges_of_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
    num_vtx: usize) -> &'a PyArray2<usize> {
    let mshline = del_msh::line2vtx::edge_of_polygon_mesh(
        &elem2idx.as_slice().unwrap(), &idx2vtx.as_slice().unwrap(), num_vtx);
    numpy::ndarray::Array2::from_shape_vec(
        (mshline.len()/2,2), mshline).unwrap().into_pyarray(py)
}

#[pyfunction]
fn triangles_from_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>) -> &'a PyArray2<usize> {
    let (tri2vtx,_) = del_msh::tri2vtx::from_polygon_mesh(
        &elem2idx.as_slice().unwrap(), &idx2vtx.as_slice().unwrap());
    numpy::ndarray::Array2::from_shape_vec(
        (tri2vtx.len()/3, 3), tri2vtx).unwrap().into_pyarray(py)
}

