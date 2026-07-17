<img src='https://julianpanetta.com/meshfem_logo.jpg' width='500px' />

# MeshFEMSparse
This repository contains the sparse matrix data structures, fast Hessian assembly routines, and sparse direct solver wrappers developed for the [MeshFEM](https://github.com/MeshFEM/MeshFEM) library.
Although most users will access these components through `MeshFEM`, they are implemented within a standalone library here to simplify integration into other codebases.

For instance, the following should suffice to import `MeshFEMSparse` routines into another project using `cmake` and [`CPM`](https://github.com/cpm-cmake/CPM.cmake):
```
include(CPM)
CPMAddPackage(
    NAME MeshFEMSparse
    GIT_REPOSITORY https://github.com/MeshFEM/MeshFEMSparse.git
    GIT_TAG main # or ideally a specific commit hash for reproducibility
)
target_link_libraries(my_target PUBLIC MeshFEMSparse)
```

For a detailed description of the underlying data structures and algorithms, see the accompanying [SIGGRAPH 2026 paper](https://doi.org/10.1145/3811386):
> Haleh Mohammadian, Xinzhuo Hu, Roi Poranne, and Julian Panetta.
> **MeshFEM: A Block-accelerated Solver for Nonlinear Finite Elements.**
> *ACM Transactions on Graphics (Proceedings of SIGGRAPH 2026)*, 45(4), Article 121, 2026.
> DOI: 10.1145/3811386
