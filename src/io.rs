use numpy::{IntoPyArray,
            PyArray2, PyArray1};
use pyo3::{pyfunction, types::PyModule, PyResult, Python, PyObject, ToPyObject, wrap_pyfunction};

pub fn add_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_wavefront_obj, m)?)?;
    m.add_function(wrap_pyfunction!(load_wavefront_obj_as_triangle_mesh, m)?)?;
    m.add_function( wrap_pyfunction!(load_nastran_as_triangle_mesh, m)?)?;
    m.add_function( wrap_pyfunction!(load_off_as_triangle_mesh, m)?)?;
    Ok(())
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn load_wavefront_obj(
    py: Python,
    path_file: String) -> (&PyArray2<f32>, &PyArray2<f32>, &PyArray2<f32>,
                           &PyArray1<usize>,
                           &PyArray1<usize>, &PyArray1<usize>, &PyArray1<usize>,
                           &PyArray1<usize>, PyObject,
                           &PyArray1<usize>, PyObject, PyObject) {
    let mut obj = del_msh::io_obj::WavefrontObj::<f32>::new();
    obj.load(&path_file);
    (
        numpy::ndarray::Array2::from_shape_vec(
            (obj.vtx2xyz.len()/3,3), obj.vtx2xyz).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (obj.vtx2uv.len()/2,2), obj.vtx2uv).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (obj.vtx2nrm.len()/3,3), obj.vtx2nrm).unwrap().into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.elem2idx).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_xyz).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_uv).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.idx2vtx_nrm).into_pyarray(py),
        numpy::ndarray::Array1::from_vec(obj.elem2group).into_pyarray(py),
        obj.group2name.to_object(py),
        numpy::ndarray::Array1::from_vec(obj.elem2mtl).into_pyarray(py),
        obj.mtl2name.to_object(py),
        obj.mtl_file_name.to_object(py)
    )
}

#[pyfunction]
pub fn load_wavefront_obj_as_triangle_mesh(
    py: Python,
    path_file: String) -> (&PyArray2<usize>,
                           &PyArray2<f32>)
{
    let (tri2vtx, vtx2xyz)
        = del_msh::io_obj::load_tri_mesh(path_file, Option::None);
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len()/3,3), tri2vtx).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len()/3,3), vtx2xyz).unwrap().into_pyarray(py)
    )
}

#[pyfunction]
pub fn load_nastran_as_triangle_mesh(
    py: Python,
    path_file: String)  -> (&PyArray2<usize>,
                            &PyArray2<f32>)
{
    let (tri2vtx, vtx2xyz) = del_msh::io_nas::load_tri_mesh(path_file);
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len()/3,3), tri2vtx).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len()/3,3), vtx2xyz).unwrap().into_pyarray(py)
    )
}


#[pyfunction]
pub fn load_off_as_triangle_mesh(
    py: Python,
    path_file: String)  -> (&PyArray2<usize>,
                            &PyArray2<f32>)
{
    let (tri2vtx, vtx2xyz) = del_msh::io_off::load_as_tri_mesh(path_file);
    (
        numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len()/3,3), tri2vtx).unwrap().into_pyarray(py),
        numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len()/3,3), vtx2xyz).unwrap().into_pyarray(py)
    )
}
