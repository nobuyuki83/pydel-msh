# del-msh

Python static 3D mesh utility library for prototyping in computer graphics research.


## Install 

```shell
pip install del-msh
```

## Generate Primitive Mesh


```python
import del_msh

# generate primitive mesh
V,F = del_msh.torus_meshtri3(0.6, 0.3, 32, 32) # torus
V,F = del_msh.capsule_meshtri3(0.1, 0.6, 32, 32, 32) # capsule
V,F = del_msh.cylinder_closed_end_meshtri3(0.1, 0.8, 32, 32) # cylinder
V,F = del_msh.sphere_meshtri3(1., 32, 32) # sphere
print("V is vertex coordinates: type",type(V),", dtype: <",V.dtype,">, shape:",V.shape)
print(type(F),F.shape)
```

    V is vertex coordinates: type <class 'numpy.ndarray'> , dtype: < float32 >, shape: (994, 3)
    <class 'numpy.ndarray'> (1984, 3)


---
## Importing Wavefront Obj file


```python
from pathlib import Path
newpath = Path('.') / 'asset' / 'HorseSwap.obj'

vtx_xyz, elem_vtx_index, elem_vtx_xyz = del_msh.load_wavefront_obj(str(newpath))    
print("vtx_xyz",type(vtx_xyz),vtx_xyz.shape)
print("elem_vtx_index",type(elem_vtx_index),elem_vtx_index.shape)
print("elem_vtx_xyz",type(elem_vtx_xyz),elem_vtx_xyz.shape)
```

    vtx_xyz <class 'numpy.ndarray'> (2738, 3)
    elem_vtx_index <class 'numpy.ndarray'> (2777,)
    elem_vtx_xyz <class 'numpy.ndarray'> (10996,)

