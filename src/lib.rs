
use numpy::{IntoPyArray,
            PyReadonlyArray2, PyReadonlyArrayDyn,
            PyArray3, PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod topology;
mod primitive;

/// A Python module implemented in Rust.
#[pymodule]
fn del_msh(_py: Python, m: &PyModule) -> PyResult<()> {

    let _ = topology::add_functions(_py, m);
    let _ = primitive::add_functions(_py, m);

    #[pyfn(m)]
    fn unify_triangle_indices_of_xyz_and_uv<'a>(
        py: Python<'a>,
        tri2vtx_xyz: PyReadonlyArrayDyn<'a, usize>,
        vtx2xyz: PyReadonlyArrayDyn<'a, f32>,
        tri2vtx_uv: PyReadonlyArrayDyn<'a, usize>,
        vtx2uv: PyReadonlyArrayDyn<'a, f32>) -> (&'a PyArray2<usize>, &'a PyArray2<f32>, &'a PyArray2<f32>) {
        let (uni2xyz, uni2uv, tri2uni, _, _)
            = del_msh::unify_index::unify_separate_trimesh_indexing_xyz_uv(
            &vtx2xyz.as_slice().unwrap(),
            &vtx2uv.as_slice().unwrap(),
            &tri2vtx_xyz.as_slice().unwrap(),
            &tri2vtx_uv.as_slice().unwrap());
        (
            numpy::ndarray::Array2::from_shape_vec(
                (tri2uni.len()/3, 3), tri2uni).unwrap().into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec(
                (uni2xyz.len()/3, 3), uni2xyz).unwrap().into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec(
                (uni2uv.len()/2, 2), uni2uv).unwrap().into_pyarray(py)
        )
    }

    #[pyfn(m)]
    pub fn load_wavefront_obj(
        py: Python,
        path_file: String) -> (&PyArray2<f32>, &PyArray2<f32>,
                               &PyArray1<usize>,
                               &PyArray1<usize>, &PyArray1<usize>) {
        let mut obj = del_msh::io_obj::WavefrontObj::<f32>::new();
        obj.load(path_file.as_str());
        (
            numpy::ndarray::Array2::from_shape_vec(
                (obj.vtx2xyz.len()/3,3), obj.vtx2xyz).unwrap().into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec(
                (obj.vtx2uv.len()/2,2), obj.vtx2uv).unwrap().into_pyarray(py),
            numpy::ndarray::Array1::from_vec(obj.elem2idx).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(obj.idx2vtx_xyz).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(obj.idx2vtx_uv).into_pyarray(py)
        )
    }

    #[pyfn(m)]
    pub fn load_wavefront_obj_as_triangle_mesh(
        py: Python,
        path_file: String) -> (&PyArray2<usize>,
                               &PyArray2<f32>) {
        let (tri2vtx, vtx2xyz)
            = del_msh::io_obj::load_tri_mesh(&path_file, Option::None);
        (
            numpy::ndarray::Array2::from_shape_vec(
                (tri2vtx.len()/3,3), tri2vtx).unwrap().into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec(
                (vtx2xyz.len()/3,3), vtx2xyz).unwrap().into_pyarray(py)
        )
    }

    #[pyfn(m)]
    pub fn unidex_vertex_attribute_for_triangle_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>) -> &'a PyArray3<f32> {
        let tri2xyz = del_msh::unindex::unidex_vertex_attribute_for_triangle_mesh(
            &tri2vtx.as_slice().unwrap(),
            &vtx2xyz.as_slice().unwrap());
        numpy::ndarray::Array3::from_shape_vec(
            (tri2xyz.len()/9,3,3), tri2xyz).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
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

    Ok(())
}