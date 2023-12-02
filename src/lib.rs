use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2,
            PyArray2, PyArray1};
use pyo3::{types::PyModule, PyResult, Python};

mod topology;
mod primitive;
mod io_obj;
mod unify_index;
mod unindex;
mod dijkstra;
mod sampling;
mod extract;
mod trimesh3_search;
mod edge2vtx;
mod elem2elem;
mod dtri;
mod polyloop;
mod bvh;
mod kdtree;

/// A Python module implemented in Rust.

/*
#[pyo3::pyclass]
struct MyClass {
    tree: del_msh::kdtree2::KdTree2<f32>,
}

#[pyo3::pymethods]
impl MyClass {
    #[new]
    fn new(vtx2xy: PyReadonlyArray2<f32>) -> Self {
        let slice = vtx2xy.as_slice().unwrap();
        let points_ = nalgebra::Matrix2xX::<f32>::from_column_slice(slice);
        let tree = del_msh::kdtree2::KdTree2::from_matrix(&points_);
        vtx2xy.as_slice().unwrap();
        MyClass {
            tree
        }
    }

    fn edges<'a>(&self, py: Python<'a>) -> &'a PyArray3<f32> {
        let e = self.tree.edges();
        numpy::ndarray::Array3::<f32>::from_shape_vec(
            (e.len() / 4, 2, 2), e).unwrap().into_pyarray(py)
    }
}
 */


/* ------------------------ */


#[pyo3::pymodule]
#[pyo3(name = "del_msh")]
fn del_msh_(_py: Python, m: &PyModule) -> PyResult<()> {
    topology::add_functions(_py, m)?;
    edge2vtx::add_functions(_py, m)?;
    elem2elem::add_functions(_py, m)?;
    unify_index::add_functions(_py, m)?;
    unindex::add_functions(_py, m)?;
    dijkstra::add_functions(_py, m)?;
    primitive::add_functions(_py, m)?;
    io_obj::add_functions(_py, m)?;
    sampling::add_functions(_py, m)?;
    extract::add_functions(_py, m)?;
    trimesh3_search::add_functions(_py, m)?;
    dtri::add_functions(_py, m)?;
    polyloop::add_functions(_py, m)?;
    bvh::add_functions(_py, m)?;
    kdtree::add_functions(_py, m)?;

    #[pyfn(m)]
    pub fn areas_of_triangles_of_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>,
    ) -> &'a PyArray1<f32> {
        let tri2area = match vtx2xyz.shape()[1] {
            2 => {
                del_msh::trimesh2::tri2area(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            3 => {
                del_msh::trimesh3::elem2area(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            _ => { panic!(); }
        };
        numpy::ndarray::Array1::from_shape_vec(
            tri2vtx.shape()[0], tri2area).unwrap().into_pyarray(py)
    }


    #[pyfn(m)]
    #[allow(clippy::identity_op)]
    pub fn extend_trimesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f64>,
        step: f64,
        niter: usize) -> &'a PyArray2<f64> {
        let tri2vtx = tri2vtx.as_slice().unwrap();
        let vtx2xyz = vtx2xyz.as_slice().unwrap();
        let vtx2nrm = del_msh::trimesh3::vtx2normal(tri2vtx, vtx2xyz);
        let num_vtx = vtx2xyz.len() / 3;
        let mut a = vec!(0_f64; num_vtx * 3);
        for i_vtx in 0..num_vtx {
            let mut p0 = [
                vtx2xyz[i_vtx * 3 + 0] + step * vtx2nrm[i_vtx * 3 + 0],
                vtx2xyz[i_vtx * 3 + 1] + step * vtx2nrm[i_vtx * 3 + 1],
                vtx2xyz[i_vtx * 3 + 2] + step * vtx2nrm[i_vtx * 3 + 2]];
            for _ in 1..niter {
                p0 = del_msh::trimesh3::extend_avoid_intersection(
                    tri2vtx, vtx2xyz, &p0, step);
            }
            a[i_vtx * 3 + 0] = p0[0];
            a[i_vtx * 3 + 1] = p0[1];
            a[i_vtx * 3 + 2] = p0[2];
        }
        numpy::ndarray::Array2::<f64>::from_shape_vec((num_vtx, 3), a).unwrap().into_pyarray(py)
    }

    #[pyfn(m)]
    pub fn extend_polyedge<'a>(
        py: Python<'a>,
        lpvtx2xyz: PyReadonlyArray2<'a, f64>,
        step: f64,
        niter: usize) -> (&'a PyArray2<usize>, &'a PyArray2<f64>) {
        assert_eq!(lpvtx2xyz.shape()[1], 3);
        let lpvtx2bin = del_msh::polyloop3::smooth_frame(lpvtx2xyz.as_slice().unwrap());
        let (tri2vtx, vtx2xyz) = del_msh::polyloop3::tube_mesh_avoid_intersection(
            lpvtx2xyz.as_slice().unwrap(), &lpvtx2bin, step, niter);
        let v1 = numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len() / 3, 3), tri2vtx).unwrap().into_pyarray(py);
        let v2 = numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len() / 3, 3), vtx2xyz).unwrap().into_pyarray(py);
        (v1, v2)
    }

    #[pyfn(m)]
    pub fn merge_hessian_mesh_laplacian_on_trimesh<'a>(
        _py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f64>,
        row2idx: PyReadonlyArray1<'a, usize>,
        idx2col: PyReadonlyArray1<'a, usize>,
        mut row2val: numpy::PyReadwriteArray1<'a, f64>,
        mut idx2val: numpy::PyReadwriteArray1<'a, f64>)
    {
        let mut merge_buffer = vec!(0_usize;0);
        let tri2vtx = tri2vtx.as_slice().unwrap();
        let vtx2xyz = vtx2xyz.as_slice().unwrap();
        let row2idx = row2idx.as_slice().unwrap();
        let idx2col = idx2col.as_slice().unwrap();
        let row2val = row2val.as_slice_mut().unwrap();
        let idx2val = idx2val.as_slice_mut().unwrap();
        for node2vtx in tri2vtx.chunks(3) {
            let (i0,i1,i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
            let v0 = &vtx2xyz[i0*3..i0*3+3];
            let v1 = &vtx2xyz[i1*3..i1*3+3];
            let v2 = &vtx2xyz[i2*3..i2*3+3];
            let cots = del_geo::tri3::cot_(v0,v1,v2);
            let emat: [f64; 9] = [
                cots[1] + cots[2], -cots[2], -cots[1],
                -cots[2], cots[2] + cots[0], -cots[0],
                -cots[1], -cots[0], cots[0] + cots[1]];
            del_msh::merge(
                node2vtx, node2vtx, &emat,
                row2idx, idx2col,
                row2val, idx2val,
                &mut merge_buffer);
        }
    }

    Ok(())
}