# A 2D Incompressible Navier–Stokes Solver with a Real-Time Viewer

Staggered MAC grid, six selectable pressure solvers, and an interactive OpenGL 3.3 front-end.

The solver is plain C11 with OpenMP; the viewer is C++17 (GLFW + glad + Dear ImGui). The solver has
no graphics dependency: the viewer reads simulation fields straight out of engine memory, with no
file exchange between the two. Everything runs interactively at grid sizes around $256^2$.

## Governing equations

The solver integrates the incompressible Navier–Stokes equations for a Newtonian fluid with
constant density and viscosity:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u}
= -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0
$$

with $\mu$ the dynamic viscosity, which is what the solver takes as input, and $\nu = \mu/\rho$ the
kinematic viscosity. Smoke is carried as a passive scalar $s$:

$$
\frac{\partial s}{\partial t} + (\mathbf{u} \cdot \nabla)s = 0
$$

The Reynolds number reported in the UI is $\mathrm{Re} = \rho U L / \mu$, where $U$ and $L$ are the
characteristic velocity and length scale supplied by the active scenario.

## Discretisation

### Staggered (MAC) grid

Pressure, divergence and smoke live at cell centres; velocity components live on the cell faces they
are normal to. This keeps the pressure-velocity coupling tight and avoids the odd-even decoupling a
collocated grid suffers from.

```
        +--------v[i][j+1]--------+
        |                         |
        |         p[i][j]         |
     u[i][j]    smoke[i][j]   u[i+1][j]
        |        div[i][j]        |
        |                         |
        +---------v[i][j]---------+
```

Both face velocities on an axis are stored with the same sign convention, so the divergence is the
difference of the outgoing and incoming face values.

Array extents: $p, s \in \mathbb{R}^{M \times N}$, $u \in \mathbb{R}^{(M+1) \times N}$,
$v \in \mathbb{R}^{M \times (N+1)}$. Indexing macros `IX`, `IX_U`, `IX_V` live in
[src/utilities.h](src/utilities.h).

Geometry is a per-cell `uint8_t` solid mask: arbitrary static obstacles, stair-stepped rather than
cut-cell.

### Time integration

One step is Chorin's projection method with first-order operator splitting
([`fluid_step`](src/core.c)):

$$
\mathbf{u}^n
\xrightarrow{\text{advect}} \mathbf{u}^a
\xrightarrow{\text{diffuse}} \mathbf{u}^*
\xrightarrow{\text{project}} \mathbf{u}^{n+1}
$$

Boundary conditions are re-applied after every stage. The passive scalar is advected after the
projection, so smoke is carried by the divergence-free field rather than the intermediate one. The
scheme is first-order accurate in time and the splitting error is $O(\Delta t)$.

## Numerical methods per stage

### Advection, semi-Lagrangian

Each face value is found by tracing the characteristic backwards one step and interpolating:

$$
\mathbf{u}^a(\mathbf{x}) = \mathbf{u}^n\!\left(\mathbf{x} - \Delta t\, \mathbf{u}^n(\mathbf{x})\right)
$$

with bilinear interpolation at the departure point and clamping to keep the sample inside the
domain. The off-component velocity at each face is the four-point average of its neighbours.

The scheme is unconditionally stable, with no CFL limit, and first order in space and time.

### Diffusion

The viscous term is applied with a 5-point Laplacian. The scheme is chosen from the diffusion number

$$
a = \frac{\nu \, \Delta t}{\Delta x^2}
$$

| Condition | Scheme |
| --- | --- |
| $a < 1/4$ | Explicit (forward Euler), one pass |
| $a \geq 1/4$ | Implicit (backward Euler), 20 Gauss–Seidel sweeps |

$a = 1/4$ is the stability limit of the explicit 2D scheme, so the switch keeps diffusion stable at
any $\Delta t$. The same explicit/implicit pair exists for the passive scalar, but the step does not
call it: smoke is transported by advection alone.

### Projection

The divergence of the intermediate field becomes the right-hand side of a Poisson problem for
pressure:

$$
\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
$$

discretised with the standard 5-point stencil. Solid neighbours are handled by substituting the
centre value, which imposes a homogeneous Neumann condition $\partial p / \partial n = 0$ at walls.
The correction is then

$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p
$$

Pressure is warm-started from the previous step rather than zeroed. This cuts the iteration count
substantially, and it is also why an under-converged solve does not stay contained: the leftover
divergence is carried into the next step.

## Pressure solvers

Three solver routines are selectable at runtime. Together with the relaxation factor and the
preconditioner they cover six classical methods on the same discrete system:

| Configuration | Routine | Setting |
| --- | --- | --- |
| Gauss–Seidel | SOR | $\omega = 1$ |
| SOR | SOR | $\omega = \omega_{\text{opt}}$ |
| Red-black Gauss–Seidel | RBGS | $\omega = 1$ |
| Red-black SOR | RBGS | $\omega = \omega_{\text{opt}}$ |
| CG | PCG | Identity preconditioner |
| PCG | PCG | Jacobi preconditioner |

SOR is lexicographic and sequential by construction, which makes it the serial reference. The
red-black colouring splits the same sweep into two passes with no intra-pass dependency, so it
parallelises; PCG parallelises throughout. The relaxation factor defaults to the theoretical
optimum for a square grid,

$$
\omega_{\text{opt}} = \frac{2}{1 + \sin(\pi / M)}
$$

clamped to $[1.0,\ 1.99]$. PCG ignores it, and the UI disables the control when PCG is selected.

All three routines stop on the maximum change between iterations,
$\max_{ij} |p^{k+1}_{ij} - p^k_{ij}| < \varepsilon$. In RBGS the change is measured over the black
pass only, which runs second and therefore already sees the red updates.

### Preconditioners

| Type | Definition |
| --- | --- |
| Identity | No preconditioning, reducing PCG to plain CG |
| Jacobi | $M^{-1} = \mathrm{diag}(A)^{-1}$, built from the live solid-neighbour count of each cell |
| Multigrid | Not implemented yet |

## Boundary conditions

Composed per scenario from primitives in [src/boundaries.h](src/boundaries.h):

- **No-slip** ($\mathbf{u} = 0$) on solid cells
- **Free-slip** (zero normal velocity) on the horizontal walls
- **Inlet**: prescribed velocity on the left edge with an optional smoke band
- **Outlet**: zero-gradient on the right edge

## Scenarios

| Scenario | Setup |
| --- | --- |
| **Lid-driven cavity** | Closed box, tangential velocity on the lid. The standard benchmark case |
| **Kármán vortex street** | Inlet-outlet channel with a circular obstacle; position and radius are adjustable |
| **NACA 2412 airfoil** | Analytic section rasterised into the solid mask, at zero angle of attack |
| **Urban city** | Rectangular building blocks in a channel |

## Building

Requires CMake 3.20 or newer, a C11/C++17 compiler, and OpenGL 3.3. GLFW and Dear ImGui are fetched
automatically by CMake; glad is vendored.

```bash
cmake -B build
cmake --build build -j
./build/fluid-graphics
```

Run the binary from the repository root. Shader paths are resolved relative to the working
directory, so starting it from anywhere else gives a black window and a shader compile error on
stderr.

The build defaults to `Release`. `Debug` is roughly an order of magnitude slower on the solver hot
path. On Linux, GLFW is configured for X11 and Wayland is disabled.

OpenMP is used when found: the solver core compiles and runs serially without it. The viewer
currently does not, since it queries the thread count unconditionally.

## Controls

The ImGui panel is grouped by what a control actually affects, with frame time, the live Reynolds
number and the OpenMP thread count reported at the top.

- **Visualisation**: smoke, pressure, velocity magnitude, vector field, or field plus vectors
- **Scenario**: switching reloads the scenario defaults
- **Solver**: pressure solver and preconditioner, both switchable mid-run
- **Kármán obstacle**: centre and radius, with *Rebuild Solids* to re-rasterise the mask
- **Reset**: restarts with your current parameters instead of the scenario defaults, so you can
  re-run the same configuration from $t = 0$

Parameters:

| Control | Effect |
| --- | --- |
| Inlet velocity | Characteristic velocity $U$ of the scenario; also feeds the reported Reynolds number |
| Viscosity | Dynamic viscosity $\mu$, on a logarithmic slider |
| $\Delta t$ | Time step. Advection is unconditionally stable, so this trades accuracy for speed rather than stability |
| $\omega$ | Relaxation factor, automatic by default, disabled under PCG |
| Poisson iteration cap | Upper bound on solver iterations. A safety cap rather than a tuning knob: the convergence threshold is the real stopping control, and a low cap leaves the projection under-solved |
| Substeps per frame | Simulation steps taken per rendered frame |
| Arrow stride | Sampling interval of the vector overlay |
| Arrow scale | Length scaling of the vector overlay |

## Layout

```
src/
  core.c/.h            solver: advection, diffusion, projection, lifecycle
  boundaries.c/.h      boundary condition primitives
  scenarios.c/.h       scenario definitions and geometry
  preconditioners.c/.h CG preconditioners
  utilities.h          indexing macros, stencils, diagnostics
  types.h              FluidContext and scenario parameters
  graphics.cpp/.hpp    OpenGL field and arrow rendering
  main.cpp             window, frame loop, ImGui panel
assets/shaders/        GLSL 3.30 field and arrow shaders
external/glad/         vendored GL loader
CMakeLists.txt         build: fetches GLFW and imgui, links OpenMP when found
LICENSE
```

## Limitations

- No turbulence model. Results hold while the flow stays laminar; the reported Reynolds number is
  the thing to watch.
- Convergence is judged by how much the pressure still changes, not by the residual. A solve that
  has merely slowed down looks like one that has converged.
- Divergence left after the projection is the direct measure of how well it worked. The panel does
  not show it.
- In a closed domain like the cavity, pressure is fixed only up to a constant: differences mean
  something, absolute values do not.
- No scenario has a closed-form solution, so accuracy can only be judged against reference data,
  never against exact error.
- The cavity has published reference profiles; this engine has not been checked against them.

## References

- Harlow, F. H. & Welch, J. E. (1965). Numerical calculation of time-dependent viscous
  incompressible flow. *Physics of Fluids*, 8(12), 2182–2189. The MAC grid.
- Chorin, A. J. (1968). Numerical solution of the Navier–Stokes equations. *Mathematics of
  Computation*, 22(104), 745–762. The projection method.
- Stam, J. (1999). Stable fluids. *ACM SIGGRAPH Proc.*, 121–128. Semi-Lagrangian advection.
- Barrett, R. et al. (1994). *Templates for the Solution of Linear Systems*. SIAM. The iterative
  solvers used in the projection stage.
- Ghia, U., Ghia, K. N. & Shin, C. T. (1982). High-Re solutions for incompressible flow using
  Navier-Stokes equations and multigrid method. *J. Comput. Phys.*, 48, 387–411. The cavity
  benchmark.
- Abbott, I. H. & Von Doenhoff, A. E. (1959). *Theory of Wing Sections*. Dover Publications. The
  NACA section geometry.
- Bridson, R. (2008). *Fluid Simulation for Computer Graphics*. A K Peters/CRC Press. Practical
  treatment of the MAC grid, semi-Lagrangian advection and pressure projection together.
- Anderson, J. D. (1995). *Computational Fluid Dynamics: The Basics with Applications*.
  McGraw-Hill Education. General background on discretising the governing equations.

## Influences

- [FluidX3D](https://github.com/ProjectPhysX/FluidX3D), a GPU lattice Boltzmann solver. What pulled me towards CFD in the first place.
- Müller, M. [Ten Minute Physics]. (2022, 3 Dec). *How to write an Eulerian fluid simulator with
  200 lines of code* [Video]. YouTube. https://www.youtube.com/watch?v=iKAVRgIrUOU
- Lague, S. [Sebastian Lague]. (2025, 11 Oct). *Coding adventure: Simulating smoke* [Video].
  YouTube. https://youtu.be/Q78wvrQ9xsU

## License

MIT. See [LICENSE](LICENSE).
