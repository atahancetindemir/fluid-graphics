#ifndef CORE_H
#define CORE_H

#include <stddef.h>

#include "types.h"
#include "scenarios.h"

// Compute the maximum absolute residual across the entire grid for convergence check.
float calculate_max_residual(const FluidContext* ctx, const float* p, const float* div);

// Diffuses the velocity field using Gauss-Seidel iteration.
void diffuse_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);

// Diffuses the scalar field with Gauss-Seidel iteration.
void diffuse_scalar(FluidContext* ctx, float* dest, const float* src);

// Advect the scalar field with given velocities using semi-Lagrangian advection.
void advect_scalar(FluidContext* ctx, float* dest, const float* src, float* u, float* v);

// Advects the velocity field with self-advection using semi-Lagrangian advection.
void advect_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);

// Computes the divergence of the velocity field.
void compute_divergence(FluidContext* ctx, float* u, float* v);

// Subtracts the pressure gradient from the velocity field to enforce incompressibility.
void subtract_gradient(FluidContext* ctx, float* u, float* v, float* p);

// Solves the pressure Poisson equation using Red-Black Gauss-Seidel.
void solve_pressure_rbgs(FluidContext* ctx, float* p, const float* div);

// Solves the pressure Poisson equation using successive over-relaxation.
void solve_pressure_sor(FluidContext* ctx, float* p, const float* div);

// Start the simulation by creating a context with given parameters.
FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, int iters, float threshold);

// Calculate physicsal parameters based on the scenario and context settings.
void fluid_setup_physics(FluidContext* ctx, ScenarioParams p, PressureSolver pressure_solver);

// Clean up the context and free memory.
void fluid_destroy_context(FluidContext* ctx);

// Executes a single step of the fluid simulation.
void fluid_step(FluidContext* ctx, ScenarioParams p, Scenario s);

#endif // CORE_H