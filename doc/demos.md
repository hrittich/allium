@page demos Demo Applications

Allium contains demo applications to illustrate the usage of the library.
The following demo applications are available.

- @subpage poisson
- @subpage poisson2
- @subpage poisson2_cuda
- @subpage heat_periodic2
- @subpage fisher2
- @subpage phase_field2

@page poisson Poisson Equation
This example demonstrates the usage of the sparse matrix interface. It solves
a discretization of the Poisson equation in 1D.

@include poisson.cpp

@page poisson2 Poisson Equation in 2D
@include poisson2.cpp

@page poisson2_cuda Poisson Equation in 2D with CUDA
@include poisson2_cuda.cpp

@page heat_periodic2 Heat Equation in 2D
@include heat_periodic2.cpp

@page fisher2 Fisher Equation
@include fisher2.cpp

@page phase_field2 Phase-Field Equation
@include phase_field2.cpp
