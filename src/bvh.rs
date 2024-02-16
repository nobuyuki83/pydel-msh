
pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(build_bvh_topology_topdown, m)?)?;
    m.add_function(wrap_pyfunction!(build_bvh_topology_morton, m)?)?;
    m.add_function(wrap_pyfunction!(shift_bvhnodes,m)?)?;
    // geometry
    m.add_function(wrap_pyfunction!(build_bvh_geometry_aabb_uniformmesh_f32, m)?)?;
    m.add_function(wrap_pyfunction!(build_bvh_geometry_aabb_uniformmesh_f64, m)?)?;
    Ok(())
}

// TODO make this function for uniforma mesh
#[pyo3::pyfunction]
fn build_bvh_topology_topdown<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>) -> &'a numpy::PyArray2<usize>
{
    assert!(tri2vtx.is_c_contiguous());
    assert!(vtx2xyz.is_c_contiguous());
    assert_eq!(vtx2xyz.shape()[1],3);
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    // change this to uniform mesh
    let bvhnodes = del_msh::bvh3_topology_topdown::from_triangle_mesh(tri2vtx, vtx2xyz);
    let bvhnodes = numpy::PyArray1::<usize>::from_slice(_py, &bvhnodes);
    bvhnodes.reshape((bvhnodes.len() / 3, 3)).unwrap()
}

#[pyo3::pyfunction]
fn build_bvh_topology_morton<'a>(
    _py: pyo3::Python<'a>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>) -> &'a numpy::PyArray2<usize>
{
    let num_vtx = vtx2xyz.shape()[0];
    assert!(vtx2xyz.is_c_contiguous());
    assert_eq!(vtx2xyz.shape(),[num_vtx,3]);
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let mut idx2vtx = vec!(0usize; num_vtx);
    let mut idx2morton = vec!(0u32; num_vtx);
    let mut vtx2morton = vec!(0u32; num_vtx);
    del_msh::bvh3_topology_morton::sorted_morten_code(
        &mut idx2vtx, &mut idx2morton, &mut vtx2morton,
        vtx2xyz);
    let bvhnodes = numpy::PyArray2::<usize>::zeros(
        _py, (num_vtx * 2 - 1, 3), false);
    {
        let bvhnodes_slice = unsafe { bvhnodes.as_slice_mut().unwrap() };
        del_msh::bvh3_topology_morton::bvhnodes_morton(
            bvhnodes_slice, &idx2vtx, &idx2morton);
    }
    bvhnodes
}

#[pyo3::pyfunction]
#[allow(clippy::identity_op)]
fn shift_bvhnodes<'a>(
    _py: pyo3::Python<'a>,
    mut bvhnodes: numpy::PyReadwriteArray2<'a, usize>,
    node_offset: usize,
    idx_offset: usize)
{
    assert!(bvhnodes.is_c_contiguous());
    let num_bvhnode = bvhnodes.shape()[0];
    assert_eq!(bvhnodes.shape()[1], 3);
    let bvhnodes = bvhnodes.as_slice_mut().unwrap();
    for i in 0..num_bvhnode {
        if bvhnodes[i * 3 + 0] != usize::MAX {
            bvhnodes[i * 3 + 0] += node_offset;
        }
        if bvhnodes[i * 3 + 2] == usize::MAX {
            bvhnodes[i * 3 + 1] += idx_offset;
            continue;
        } else {
            bvhnodes[i * 3 + 1] += node_offset;
            bvhnodes[i * 3 + 2] += node_offset;
        }
    }
}

fn build_bvh_geometry_aabb_uniformmesh<'a, T>(
    _py: pyo3::Python<'a>,
    mut aabbs: numpy::PyReadwriteArray2<'a, T>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, T>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, T>)
    where T: numpy::Element + num_traits::Float
{
    assert!(aabbs.is_c_contiguous());
    assert!(bvhnodes.is_c_contiguous());
    assert!(elem2vtx.is_c_contiguous());
    assert!(vtx2xyz0.is_c_contiguous());
    assert!(vtx2xyz1.is_c_contiguous());
    assert_eq!(bvhnodes.shape()[0], aabbs.shape()[0]);
    assert_eq!(bvhnodes.shape()[1], 3);
    assert_eq!(aabbs.shape()[1], 6);
    let num_noel = elem2vtx.shape()[1];
    let aabbs = aabbs.as_slice_mut().unwrap();
    let bvhnodes = bvhnodes.as_slice().unwrap();
    let elem2vtx = elem2vtx.as_slice().unwrap();
    let vtx2xyz0 = vtx2xyz0.as_slice().unwrap();
    let vtx2xyz1 = vtx2xyz1.as_slice().unwrap();
    del_msh::bvh3::build_geometry_aabb_for_uniform_mesh(
        aabbs,
        i_bvhnode_root, bvhnodes,
        elem2vtx,
        num_noel, vtx2xyz0, vtx2xyz1);
}

// 2D and 3D
#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_uniformmesh_f32<'a>(
    _py: pyo3::Python<'a>,
    aabbs: numpy::PyReadwriteArray2<'a, f32>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f32>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f32>)
{
    build_bvh_geometry_aabb_uniformmesh::<f32>(
        _py, aabbs, bvhnodes, tri2vtx, vtx2xyz0, i_bvhnode_root, vtx2xyz1);
}

#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_uniformmesh_f64<'a>(
    _py: pyo3::Python<'a>,
    aabbs: numpy::PyReadwriteArray2<'a, f64>,
    bvhnodes: numpy::PyReadonlyArray2<'a, usize>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f64>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f64>)
{
    build_bvh_geometry_aabb_uniformmesh::<f64>(
        _py, aabbs, bvhnodes, tri2vtx, vtx2xyz0, i_bvhnode_root, vtx2xyz1);
}

