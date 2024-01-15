
use numpy::{IntoPyArray,
            PyArray2};
use pyo3::{Python, pyfunction, types::PyModule, PyResult, wrap_pyfunction};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torus_meshtri3, m)?)?;
    m.add_function(wrap_pyfunction!(capsule_meshtri3, m)?)?;
    m.add_function( wrap_pyfunction!(cylinder_closed_end_meshtri3, m)?)?;
    m.add_function( wrap_pyfunction!(sphere_meshtri3, m)?)?;
    m.add_function( wrap_pyfunction!(trimesh3_hemisphere_zup, m)?)?;
    Ok(())
}

#[pyfunction]
fn torus_meshtri3(
    py: Python,
    radius: f64, radius_tube: f64,
    nlg: usize, nlt: usize) -> (&PyArray2<usize>, &PyArray2<f64>) {
    let (tri_vtx, vtx_xyz) = del_msh::trimesh3_primitive::torus_yup::<f64>(
        radius, radius_tube, nlg, nlt);
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri_vtx.len()/3,3), tri_vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn capsule_meshtri3(
    py: Python,
    r: f64, l: f64,
    nc: usize, nr: usize, nl: usize) -> (&PyArray2<usize>, &PyArray2<f64>) {
    let ( tri_vtx, vtx_xyz) = del_msh::trimesh3_primitive::capsule_yup::<f64>(
        r, l, nc, nr, nl);
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri_vtx.len()/3,3), tri_vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn cylinder_closed_end_meshtri3(
    py: Python,
    r: f64, l: f64,
    nr: usize, nl: usize) -> (&PyArray2<usize>, &PyArray2<f64>) {
    let (tri_vtx, vtx_xyz) = del_msh::trimesh3_primitive::cylinder_closed_end_yup::<f64>(
        r, l, nr, nl);
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx_xyz.len()/3,3), vtx_xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri_vtx.len()/3,3), tri_vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn sphere_meshtri3(
    py: Python,
    r: f32,
    nr: usize, nl: usize) -> (&PyArray2<usize>, &PyArray2<f32>) {
    let ( tri2vtx, vtx2xyz) = del_msh::trimesh3_primitive::sphere_yup(
        r, nr, nl);
    let v = numpy::ndarray::Array2::from_shape_vec(
        (vtx2xyz.len()/3, 3), vtx2xyz).unwrap();
    let f = numpy::ndarray::Array2::from_shape_vec(
        (tri2vtx.len()/3, 3), tri2vtx).unwrap();
    (f.into_pyarray(py), v.into_pyarray(py))
}

#[pyfunction]
fn trimesh3_hemisphere_zup(
    py: Python,
    r: f32,
    nr: usize, nl: usize) -> (&PyArray2<usize>, &PyArray2<f32>) {
    let ( tri2vtx, vtx2xyz) = del_msh::trimesh3_primitive::hemisphere_zup(
        r, nr, nl);
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len()/3, 3), tri2vtx).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
        (vtx2xyz.len()/3, 3), vtx2xyz).unwrap().into_pyarray(py)
    )
}
