use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2,
            PyArray1, PyArray2};
use pyo3::{Python, pyfunction, types::PyModule, PyResult, wrap_pyfunction};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edges_of_uniform_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(edges_of_polygon_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(triangles_from_polygon_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(elsuel_uniform_mesh_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(elsuel_uniform_mesh_simplex, m)?)?;
    m.add_function( wrap_pyfunction!(group_connected_element_uniform_polygon_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
fn edges_of_uniform_mesh<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> &'a PyArray2<usize> {
    let mshline = del_msh::line2vtx::from_sepecific_edges_of_uniform_mesh(
        &elem2vtx.as_slice().unwrap(), 3,
        &[0, 1, 1, 2, 2, 0], num_vtx);
    numpy::ndarray::Array2::from_shape_vec(
        (mshline.len() / 2, 2), mshline).unwrap().into_pyarray(py)
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
        (mshline.len() / 2, 2), mshline).unwrap().into_pyarray(py)
}

#[pyfunction]
fn triangles_from_polygon_mesh<'a>(
    py: Python<'a>,
    elem2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>) -> &'a PyArray2<usize> {
    let (tri2vtx, _) = del_msh::tri2vtx::from_polygon_mesh(
        &elem2idx.as_slice().unwrap(), &idx2vtx.as_slice().unwrap());
    numpy::ndarray::Array2::from_shape_vec(
        (tri2vtx.len() / 3, 3), tri2vtx).unwrap().into_pyarray(py)
}

#[pyfunction]
fn elsuel_uniform_mesh_polygon<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> &'a PyArray2<usize> {
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node) = del_msh::elem2elem::face2node_of_polygon_element(num_node);
    let elsuel = del_msh::elem2elem::from_uniform_mesh2(
        &elem2vtx.as_slice().unwrap(),
        num_node, &face2idx, &idx2node, num_vtx);
    assert_eq!(elem2vtx.len(), elsuel.len());
    numpy::ndarray::Array2::from_shape_vec(
        (elsuel.len() / num_node, num_node), elsuel).unwrap().into_pyarray(py)
}

#[pyfunction]
fn elsuel_uniform_mesh_simplex<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> &'a PyArray2<usize> {
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node)
        = del_msh::elem2elem::face2node_of_simplex_element(num_node);
    let elsuel = del_msh::elem2elem::from_uniform_mesh2(
        &elem2vtx.as_slice().unwrap(),
        num_node, &face2idx, &idx2node, num_vtx);
    assert_eq!(elem2vtx.len(), elsuel.len());
    numpy::ndarray::Array2::from_shape_vec(
        (elsuel.len() / num_node, num_node), elsuel).unwrap().into_pyarray(py)
}

#[pyfunction]
fn group_connected_element_uniform_polygon_mesh<'a>(
    py: Python<'a>,
    elem2vtx: PyReadonlyArray2<'a, usize>,
    num_vtx: usize) -> (usize, &'a PyArray1<usize>)
{
    let num_node = elem2vtx.shape()[1];
    let (face2idx, idx2node)
        = del_msh::elem2elem::face2node_of_polygon_element(num_node);
    let elem2elem_adj = del_msh::elem2elem::from_uniform_mesh2(
        &elem2vtx.as_slice().unwrap(),
        num_node, &face2idx, &idx2node, num_vtx);
    let (num_group, elem2group) = del_msh::group::make_group_elem(
        elem2vtx.as_slice().unwrap(),
        num_node,
        &elem2elem_adj);
    (
        num_group,
        numpy::ndarray::Array1::from_vec(elem2group).into_pyarray(py)
    )
}