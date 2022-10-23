# del-msh

Python static 3D mesh utility library for prototyping in computer graphics research.


```python
import del_msh

# generate primitive mesh
V,F = del_msh.torus_meshtri3(0.6, 0.3, 32, 32) # torus
V,F = del_msh.capsule_meshtri3(0.1, 0.6, 32, 32, 32) # capsule
V,F = del_msh.cylinder_closed_end_meshtri3(0.1, 0.8, 32, 32) # cylinder
V,F = del_msh.sphere_meshtri3(1., 32, 32) # sphere
print(type(V),type(F), V.shape, F.shape)
```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'> (994, 3) (1984, 3)

