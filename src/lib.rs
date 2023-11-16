use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
            PyArray3, PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod topology;
mod primitive;
mod io_obj;

/// A Python module implemented in Rust.

#[pymodule]
#[pyo3(name="del_msh")]
fn del_msh_(_py: Python, m: &PyModule) -> PyResult<()> {
    let _ = topology::add_functions(_py, m);
    let _ = primitive::add_functions(_py, m);
    let _ = io_obj::add_functions(_py, m);

    #[pyfn(m)]
    fn extract_flagged_polygonal_element<'a>(
        py: Python<'a>,
        elem2idx: PyReadonlyArray1<'a, usize>,
        idx2vtx: PyReadonlyArray1<'a, usize>,
        elem2flag: PyReadonlyArray1<'a, bool>) -> (&'a PyArray1<usize>, &'a PyArray1<usize>) {
        assert_eq!(elem2flag.len() + 1, elem2idx.len());
        let mut felem2jdx = vec!(0_usize; 1);
        let mut jdx2vtx = vec!(0_usize; 0);
        for i_elem in 0..elem2flag.len() {
            if !*elem2flag.get(i_elem).unwrap() { continue; }
            let idx0 = *elem2idx.get(i_elem).unwrap();
            let idx1 = *elem2idx.get(i_elem + 1).unwrap();
            for idx in idx0..idx1 {
                jdx2vtx.push(*idx2vtx.get(idx).unwrap());
            }
            felem2jdx.push(jdx2vtx.len());
        }
        (
            numpy::ndarray::Array1::from_vec(felem2jdx).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(jdx2vtx).into_pyarray(py)
        )
    }

    #[pyfn(m)]
    fn unify_two_indices_of_triangle_mesh<'a>(
        py: Python<'a>,
        tri2vtxa: PyReadonlyArrayDyn<'a, usize>,
        tri2vtxb: PyReadonlyArrayDyn<'a, usize>) -> (&'a PyArray2<usize>, &'a PyArray1<usize>, &'a PyArray1<usize>) {
        let (tri2uni, uni2vtxxyz, uni2vtxuv)
            = del_msh::unify_index::unify_two_indices_of_triangle_mesh(
            &tri2vtxa.as_slice().unwrap(),
            &tri2vtxb.as_slice().unwrap());
        (
            numpy::ndarray::Array2::from_shape_vec(
                (tri2uni.len() / 3, 3), tri2uni).unwrap().into_pyarray(py),
            numpy::ndarray::Array1::from_vec(uni2vtxxyz).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(uni2vtxuv).into_pyarray(py),
        )
    }

    #[pyfn(m)]
    fn unify_two_indices_of_polygon_mesh<'a>(
        py: Python<'a>,
        elem2idx: PyReadonlyArrayDyn<'a, usize>,
        idx2vtxa: PyReadonlyArrayDyn<'a, usize>,
        idx2vtxb: PyReadonlyArrayDyn<'a, usize>) -> (&'a PyArray1<usize>, &'a PyArray1<usize>, &'a PyArray1<usize>) {
        let (idx2uni, uni2vtxa, uni2vtxb)
            = del_msh::unify_index::unify_two_indices_of_polygon_mesh(
            &elem2idx.as_slice().unwrap(),
            &idx2vtxa.as_slice().unwrap(),
            &idx2vtxb.as_slice().unwrap());
        (
            numpy::ndarray::Array1::from_vec(idx2uni).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(uni2vtxa).into_pyarray(py),
            numpy::ndarray::Array1::from_vec(uni2vtxb).into_pyarray(py)
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
            (tri2xyz.len() / (3 * num_val), 3, num_val), tri2xyz).unwrap()
            .into_pyarray(py)
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
        vtx2xyz: PyReadonlyArray2<'a, f32>,
    ) -> &'a PyArray1<f32> {
        let tri2area = match vtx2xyz.shape()[1] {
            2 => {
                del_msh::trimesh::areas_of_trimesh2(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            },
            3 => {
                del_msh::trimesh::areas_of_trimesh3(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            },
            _ => { panic!(); }
        };
        numpy::ndarray::Array1::from_shape_vec(
            tri2vtx.shape()[0], tri2area).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    pub fn sample_uniformly_trimesh(
        _py: Python,
        cumsum_tri2area: PyReadonlyArray1<f32>,
        val01_a: f32,
        val01_b: f32) -> (usize, f32, f32) {
        del_msh::sampling::sample_uniformly_trimesh(
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

    #[pyfn(m)]
    pub fn extend_trimesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f64>,
        step: f64,
        niter: usize) -> &'a PyArray2<f64> {
        let tri2vtx = tri2vtx.as_slice().unwrap();
        let vtx2xyz = vtx2xyz.as_slice().unwrap();
        let vtx2nrm = del_msh::trimesh::normal(tri2vtx, vtx2xyz);
        let num_vtx = vtx2xyz.len() / 3;
        let mut a= vec!(0_f64; num_vtx*3);
        for i_vtx in 0..num_vtx {
            let mut p0 = [
                vtx2xyz[i_vtx * 3 + 0] + step * vtx2nrm[i_vtx * 3 + 0],
                vtx2xyz[i_vtx * 3 + 1] + step * vtx2nrm[i_vtx * 3 + 1],
                vtx2xyz[i_vtx * 3 + 2] + step * vtx2nrm[i_vtx * 3 + 2]];
            for _ in 1..niter {
                p0 = del_msh::trimesh::extend_avoid_intersection(
                    tri2vtx, vtx2xyz, &p0, step);
            }
            a[i_vtx*3+0] = p0[0];
            a[i_vtx*3+1] = p0[1];
            a[i_vtx*3+2] = p0[2];
        }
        numpy::ndarray::Array2::<f64>::from_shape_vec((num_vtx,3), a).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    pub fn extend_polyedge<'a>(
        py: Python<'a>,
        lpvtx2xyz: PyReadonlyArray2<'a, f64>,
        step: f64,
        niter: usize) -> (&'a PyArray2<usize>, &'a PyArray2<f64>) {
        assert_eq!(lpvtx2xyz.shape()[1], 3);
        let lpvtx2bin = del_msh::edgeloop::smooth_frame(lpvtx2xyz.as_slice().unwrap());
        let (tri2vtx, vtx2xyz) = del_msh::edgeloop::tube_mesh_avoid_intersection(
            lpvtx2xyz.as_slice().unwrap(), &lpvtx2bin, step.into(), niter);
        let v1 = numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len() / 3, 3), tri2vtx).unwrap().into_pyarray(py);
        let v2 = numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len() / 3, 3), vtx2xyz).unwrap().into_pyarray(py);
        (v1, v2)
    }

    Ok(())
}