
use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
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
        let num_val = vtx2xyz.shape()[1];
        let tri2xyz = del_msh::unindex::unidex_vertex_attribute_for_triangle_mesh(
            &tri2vtx.as_slice().unwrap(),
            &vtx2xyz.as_slice().unwrap(),
            num_val);
        numpy::ndarray::Array3::from_shape_vec(
            (tri2xyz.len()/(3*num_val),3,num_val), tri2xyz).unwrap().into_pyarray(py)
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

    #[pyfn(m)]
    pub fn areas_of_triangles_of_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>
    ) -> &'a PyArray1<f32> {
        assert_eq!(vtx2xyz.shape()[1], 3);
        let tri2area = del_msh::sampling::areas_of_triangles_of_mesh(
            tri2vtx.as_slice().unwrap(),
            vtx2xyz.as_slice().unwrap() );
        numpy::ndarray::Array1::from_shape_vec(
            tri2vtx.shape()[0], tri2area).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    pub fn sample_uniform(
        _py: Python,
        cumsum_tri2area: PyReadonlyArray1<f32>,
        val01_a: f32,
        val01_b: f32) -> (usize, f32, f32) {
        del_msh::sampling::sample_uniform(
            cumsum_tri2area.as_slice().unwrap(),
            val01_a, val01_b)
    }

    #[pyfn(m)]
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

    Ok(())
}