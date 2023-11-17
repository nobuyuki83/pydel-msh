use numpy::{IntoPyArray,
            PyReadonlyArray1, PyReadonlyArray2,
            PyArray3, PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod topology;
mod primitive;
mod io_obj;
mod unify_index;
mod unindex;
mod dijkstra;
mod sampling;
mod extract;

/// A Python module implemented in Rust.

fn squared_dist(p0: &[f32], p1: &[f32]) -> f32 {
    (p0[0] - p1[0]) * (p0[0] - p1[0])
        + (p0[1] - p1[1]) * (p0[1] - p1[1])
        + (p0[2] - p1[2]) * (p0[2] - p1[2])
}

#[pyo3::pyfunction]
fn first_intersection_ray_meshtri3<'a>(
    py: Python<'a>,
    src: PyReadonlyArray1<'a, f32>,
    dir: PyReadonlyArray1<'a, f32>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    tri2vtx: PyReadonlyArray2<'a, usize>) -> (&'a PyArray1<f32>, i64)
{
    let res = del_msh::trimesh3_search::first_intersection_ray(
        src.as_slice().unwrap(),
        dir.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
        tri2vtx.as_slice().unwrap());
    match res {
        None => {
            let a = PyArray1::<f32>::zeros(py, 3, true);
            return (a, -1);
        }
        Some(postri) => {
            let a = PyArray1::<f32>::from_slice(py, &postri.0);
            return (a, postri.1 as i64);
        }
    }
}

#[pyo3::pyfunction]
fn pick_vertex_meshtri3<'a>(
    src: PyReadonlyArray1<'a, f32>,
    dir: PyReadonlyArray1<'a, f32>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    tri2vtx: PyReadonlyArray2<'a, usize>) -> i64
{
    let res = del_msh::trimesh3_search::first_intersection_ray(
        src.as_slice().unwrap(),
        dir.as_slice().unwrap(),
        vtx2xyz.as_slice().unwrap(),
        tri2vtx.as_slice().unwrap());
    match res {
        None => {
            return -1;
        }
        Some(postri) => {
            let pos = postri.0;
            let idx_tri = postri.1;
            let i0 = tri2vtx.get([idx_tri, 0]).unwrap();
            let i1 = tri2vtx.get([idx_tri, 1]).unwrap();
            let i2 = tri2vtx.get([idx_tri, 2]).unwrap();
            let q0 = &vtx2xyz.as_slice().unwrap()[i0 * 3..i0 * 3 + 3];
            let q1 = &vtx2xyz.as_slice().unwrap()[i1 * 3..i1 * 3 + 3];
            let q2 = &vtx2xyz.as_slice().unwrap()[i2 * 3..i2 * 3 + 3];
            let d0 = squared_dist(&pos, q0);
            let d1 = squared_dist(&pos, q1);
            let d2 = squared_dist(&pos, q2);
            if d0 <= d1 && d0 <= d2 { return *i0 as i64; }
            if d1 <= d2 && d1 <= d0 { return *i1 as i64; }
            if d2 <= d0 && d2 <= d1 { return *i2 as i64; }
            return -1;
        }
    }
}

#[pyo3::pyclass]
struct MyClass {
    tree: del_msh::kdtree2::KdTree2<f32>,
}

#[pyo3::pymethods]
impl MyClass {
    #[new]
    fn new<'a>(vtx2xy: PyReadonlyArray2<'a, f32>) -> Self {
        let slice = vtx2xy.as_slice().unwrap();
        let points_ = nalgebra::Matrix2xX::<f32>::from_column_slice(slice);
        let tree = del_msh::kdtree2::KdTree2::from_matrix(&points_);
        vtx2xy.as_slice().unwrap();
        MyClass {
            tree: tree
        }
    }

    fn edges<'a>(&self, py: Python<'a>) -> &'a PyArray3<f32> {
        let e = self.tree.edges();
        numpy::ndarray::Array3::<f32>::from_shape_vec(
            (e.len() / 4, 2, 2), e).unwrap().into_pyarray(py)
    }
}

/* ------------------------ */


#[pymodule]
#[pyo3(name = "del_msh")]
fn del_msh_(_py: Python, m: &PyModule) -> PyResult<()> {
    let _ = topology::add_functions(_py, m);
    let _ = primitive::add_functions(_py, m);
    let _ = io_obj::add_functions(_py, m);
    let _ = unify_index::add_functions(_py, m);
    let _ = unindex::add_functions(_py, m);
    let _ = dijkstra::add_functions(_py, m);
    let _ = sampling::add_functions(_py, m);
    let _ = extract::add_functions(_py, m);

    #[pyfn(m)]
    pub fn areas_of_triangles_of_mesh<'a>(
        py: Python<'a>,
        tri2vtx: PyReadonlyArray2<'a, usize>,
        vtx2xyz: PyReadonlyArray2<'a, f32>,
    ) -> &'a PyArray1<f32> {
        let tri2area = match vtx2xyz.shape()[1] {
            2 => {
                del_msh::trimesh2::areas(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            3 => {
                del_msh::trimesh3::areas(
                    tri2vtx.as_slice().unwrap(),
                    vtx2xyz.as_slice().unwrap())
            }
            _ => { panic!(); }
        };
        numpy::ndarray::Array1::from_shape_vec(
            tri2vtx.shape()[0], tri2area).unwrap().into_pyarray(py)
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
        let vtx2nrm = del_msh::trimesh3::normal(tri2vtx, vtx2xyz);
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
        let lpvtx2bin = del_msh::edgeloop::smooth_frame(lpvtx2xyz.as_slice().unwrap());
        let (tri2vtx, vtx2xyz) = del_msh::edgeloop::tube_mesh_avoid_intersection(
            lpvtx2xyz.as_slice().unwrap(), &lpvtx2bin, step.into(), niter);
        let v1 = numpy::ndarray::Array2::from_shape_vec(
            (tri2vtx.len() / 3, 3), tri2vtx).unwrap().into_pyarray(py);
        let v2 = numpy::ndarray::Array2::from_shape_vec(
            (vtx2xyz.len() / 3, 3), vtx2xyz).unwrap().into_pyarray(py);
        (v1, v2)
    }

    m.add_function(pyo3::wrap_pyfunction!(first_intersection_ray_meshtri3, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(pick_vertex_meshtri3, m)?)?;
    m.add_class::<MyClass>()?;

    Ok(())
}