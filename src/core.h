#ifndef CORE_H
#define CORE_H

#include <stddef.h>

#include "types.h"
#include "preconditioners.h"
#include "scenarios.h"

// Compute the maximum absolute residual across the entire grid for convergence check.
float calculate_max_residual(const FluidContext* ctx, const float* p, const float* div);

// Largest right-hand side term of the pressure Poisson equation, max |div * cp|.
float calculate_max_rhs(const FluidContext* ctx, const float* div);

// Residual level at which a pressure solve counts as converged, max(rtol * rhs, atol).
float calculate_residual_target(const FluidContext* ctx, float rhs);

// Explicit diffusion of velocity field.
void diffuse_velocity_explicit(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src, float a);

// Implicit diffusion of velocity field.
void diffuse_velocity_implicit(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src, float a);

// Compute the diffusion coefficient and choose explicit or implicit velocity diffusion.
void diffuse_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);

// Explicit diffusion of scalar field.
void diffuse_scalar_explicit(FluidContext* ctx, float* dest, const float* src, float a);

// Implicit diffusion of scalar field.
void diffuse_scalar_implicit(FluidContext* ctx, float* dest, const float* src, float a);

// Compute the diffusion coefficient and choose explicit or implicit scalar diffusion.
void diffuse_scalar(FluidContext* ctx, float* dest, const float* src);

// Advect the scalar field with given velocities using semi-Lagrangian advection.
void advect_scalar(FluidContext* ctx, float* dest, const float* src, float* u, float* v);

// Advects the velocity field with self-advection using semi-Lagrangian advection.
void advect_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);

// Computes the divergence of the velocity field.
void compute_divergence(FluidContext* ctx, float* u, float* v);

// Subtracts the pressure gradient from the velocity field to enforce incompressibility.
void subtract_gradient(FluidContext* ctx, float* u, float* v, float* p);

// Measures the mass the last step left unbalanced in each fluid cell, |net face flux| / U,
// into ctx->continuity_rms and ctx->continuity_max.
void measure_continuity(FluidContext* ctx, ScenarioParams p);

// Stores the outcome of the pressure solve that just finished in ctx->stats and prints it in debug builds.
void record_solve_stats(FluidContext* ctx, const char* tag, SolveStats stats, double start_time);

// Solves the pressure Poisson equation using Red-Black Gauss-Seidel.
void solve_pressure_rbgs(FluidContext* ctx, float* p, const float* div);

// Solves the pressure Poisson equation using successive over-relaxation.
void solve_pressure_sor(FluidContext* ctx, float* p, const float* div);

// Solves the pressure Poisson equation using Preconditioned Conjugate Gradient method.
void solve_pressure_pcg(FluidContext* ctx, float* p, const float* div);

// Computes the cell-centered velocity magnitude for every cell into dest (num_cells floats).
void fluid_velocity_magnitude(const FluidContext* ctx, float* dest);

// Interpolates the staggered velocity field onto cell centers, writing num_cells
// floats into each of u_center and v_center.
void fluid_velocity_centered(const FluidContext* ctx, float* u_center, float* v_center);

// Start the simulation by creating a context with given parameters.
FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, SolveTolerance tol);

// Zeroes every simulation field while keeping the allocation and the physics settings intact.
void fluid_reset_context(FluidContext* ctx);

// Returns the theoretically optimal over-relaxation factor for the grid, clamped to [1.0, 1.99].
float fluid_optimal_omega(const FluidContext* ctx);

// Returns a lower-case name for a solve status, for display.
const char* fluid_solve_status_name(SolveStatus status);

// Returns the Reynolds number from the current viscosity, density and scenario parameters.
float fluid_reynolds(const FluidContext* ctx, ScenarioParams p);

// Points the context at the preconditioner implementation matching the given type.
void fluid_set_preconditioner(FluidContext* ctx, PrecondType precondition);

/**
 * @brief Configures the solver side of the context: the pressure solver, the preconditioner and the optimal over-relaxation factor.
 * @param ctx Pointer to the fluid context to configure.
 * @param pressure_solver Function pointer to the pressure solver to use (e.g., solve_pressure_pcg, solve_pressure_rbgs, solve_pressure_sor).
 * @param precondition Preconditioner type to use for the Conjugate Gradient solver (
 * e.g., PRECOND_IDENTITY, PRECOND_JACOBI, PRECOND_MULTIGRID).
 */
void fluid_setup_physics(FluidContext* ctx, PressureSolver pressure_solver, PrecondType precondition);

// Clean up the context and free memory.
void fluid_destroy_context(FluidContext* ctx);

// Executes a single step of the fluid simulation.
void fluid_step(FluidContext* ctx, ScenarioParams p, Scenario s);

#endif // CORE_H