#ifndef CORE_H
#define CORE_H

#include "types.h"

void diffuse_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);
void diffuse_scalar(FluidContext* ctx, float* dest, const float* src);
void advect_scalar(FluidContext* ctx, float* dest, const float* src, float* u, float* v);
void advect_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src);
void compute_divergence(FluidContext* ctx, float* u, float* v);
void solve_pressure_rbgs(FluidContext* ctx, float* p, float* div);
void solve_pressure_sor(FluidContext* ctx, float* p, float* div);
void subtract_gradient(FluidContext* ctx, float* u, float* v, float* p);
FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, int iters);
void fluid_setup_physics(FluidContext* ctx, ScenarioParams p);
void fluid_destroy_context(FluidContext* ctx);
void fluid_step(FluidContext* ctx, ScenarioParams p);

#endif // CORE_H