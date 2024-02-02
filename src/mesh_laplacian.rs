use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::Python;

pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(merge_hessian_mesh_laplacian_on_trimesh3, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_rotation_for_vertex, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn merge_hessian_mesh_laplacian_on_trimesh3<'a>(
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
        let emat = del_geo::tri3::emat_cotangent_laplacian(v0,v1,v2);
        del_msh::merge(
            node2vtx, node2vtx, &emat,
            row2idx, idx2col,
            row2val, idx2val,
            &mut merge_buffer);
    }
}

#[pyo3::pyfunction]
pub fn optimal_rotation_for_vertex<'a>(
    _py: Python<'a>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f64>,
    vtx2xyz_def: PyReadonlyArray2<'a, f64>,
    vtx2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
    idx2val: PyReadonlyArray1<'a, f64>,
    mut vtx2rot: numpy::PyReadwriteArray3<'a, f64>) {
    assert_eq!(vtx2xyz_ini.shape(), vtx2xyz_def.shape());
    assert_eq!(vtx2xyz_ini.shape().len(),2);
    assert_eq!(vtx2xyz_ini.shape()[1],3);
    let num_vtx = vtx2xyz_ini.shape()[0];
    let vtx2xyz_ini = vtx2xyz_ini.as_slice().unwrap();
    let vtx2xyz_def = vtx2xyz_def.as_slice().unwrap();
    let vtx2idx = vtx2idx.as_slice().unwrap();
    let idx2col = idx2vtx.as_slice().unwrap();
    let idx2val = idx2val.as_slice().unwrap();
    let vtx2rot = vtx2rot.as_slice_mut().unwrap();
    for i_vtx in 0..num_vtx {
        let adj2vtx = &idx2col[vtx2idx[i_vtx]..vtx2idx[i_vtx+1]];
        let adj2weight = &idx2val[vtx2idx[i_vtx]..vtx2idx[i_vtx+1]];
        let rot = del_msh::trimesh3::optimal_rotation_for_arap(
            i_vtx,
            adj2vtx,vtx2xyz_ini, vtx2xyz_def,
            adj2weight, -1.);
        rot.iter().enumerate().for_each(|(i,&v)| vtx2rot[i_vtx*9+i] = v );
    }
}