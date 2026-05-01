#include "core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

#include "types.h"
#include "utilities.h"
#include "scenarios.h"

/*
MAC GRID (STAGGERED GRID) MEMORY & SPATIAL LAYOUT

Grid Size (Simulation): M   X N
Grid Size (U):          M+1 X N
Grid Size (V):          M   X N+1

Y
^
|
|
|
|
+-----------> X

+--------------------------+--------------------------+--------------------------+
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |         p[i][j+1]        |                          |
|                          |        smoke[i][j+1]     |                          |
|                          |         div[i][j+1]      |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |         (i, j+0.5)       |                          |
+--------------------------+---------v[i][j+1]--------+--------------------------+
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |          (i, j)          |                          |
|        p[i-1][j]      u[i][j]       p[i][j]      u[i+1][j]     p[i+1][j]       |
|       smoke[i-1][j]  (i-0.5, j)    smoke[i][j]   (i+0.5, j)   smoke[i+1][j]    |
|        div[i-1][j]       |          div[i][j]       |          div[i+1][j]     |
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
+--------------------------+---------v[i][j]----------+--------------------------+
|                          |        (i, j-0.5)        |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |         p[i][j-1]        |                          |
|                          |        smoke[i][j-1]     |                          |
|                          |         div[i][j-1]      |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
|                          |                          |                          |
+--------------------------+--------------------------+--------------------------+
*/

// Fetch neighbors with Neumann BC handling for pressure solver
static inline void fetch_neumann_neighbors(const FluidContext* ctx, const float* p, size_t i, size_t j, float* p_left, float* p_right, float* p_bottom, float* p_top) {
    float p_center = p[IX(ctx, i, j)];

    *p_left   = ctx->solid[IX(ctx, i - 1, j)] ? p_center : p[IX(ctx, i - 1, j)];
    *p_right  = ctx->solid[IX(ctx, i + 1, j)] ? p_center : p[IX(ctx, i + 1, j)];
    *p_bottom = ctx->solid[IX(ctx, i, j - 1)] ? p_center : p[IX(ctx, i, j - 1)];
    *p_top    = ctx->solid[IX(ctx, i, j + 1)] ? p_center : p[IX(ctx, i, j + 1)];
}

// 5-point stencil application for Poisson equation with Neumann BCs
static inline float compute_poisson_update(const FluidContext* ctx, const float* p, const float* div, float cp, size_t i, size_t j) {
    float p_left, p_right, p_bottom, p_top;
    fetch_neumann_neighbors(ctx, p, i, j, &p_left, &p_right, &p_bottom, &p_top);
    return (p_left + p_right + p_bottom + p_top - div[IX(ctx, i, j)] * cp) * 0.25f;
}

// Calculate algebraic residual (b - Ax) for a single cell.
static inline float compute_cell_residual(const FluidContext* ctx, const float* p, const float* div, float cp, size_t i, size_t j) {
    float p_left, p_right, p_bottom, p_top;
    fetch_neumann_neighbors(ctx, p, i, j, &p_left, &p_right, &p_bottom, &p_top);
    return (p_left + p_right + p_bottom + p_top - 4.0f * p[IX(ctx, i, j)]) - div[IX(ctx, i, j)] * cp;
}

float calculate_max_residual(const FluidContext* ctx, const float* p, const float* div) {
    float max_res = 0.0f;
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

    #pragma omp parallel for reduction(max:max_res)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i, j)]) continue;

            float residual = fabsf(compute_cell_residual(ctx, p, div, cp, i, j));
            max_res = MAX(max_res, residual);
        }
    }
    return max_res;
}

void diffuse_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src) {

    mat_cpy(u_dest, (float*)u_src, ctx->x + 1, ctx->y);
    mat_cpy(v_dest, (float*)v_src, ctx->x, ctx->y + 1);

    float kinematic_visc = ctx->visc / ctx->dens;

    // Friction coefficient
    float a = ctx->dt * kinematic_visc / (ctx->dx * ctx->dx);

    // Gauss-Seidel iteration for diffusion
    for (size_t iter = 0; iter < ctx->iter_count; iter++) {
        // u (horizontal) velocity diffusion
        for (size_t i = 1; i < ctx->x; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                // Skip solid boundaries
                // Dirichlet boundary condition at solids: u = 0. This means that the velocity at the solid boundary is zero,
                // and we can use this to compute the diffusion for the neighboring fluid cells.
                if (ctx->solid[IX(ctx, i - 1, j)] || ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                // Fetch neighboring velocities (with no-slip condition at solids)
                float u_left = u_dest[IX_U(ctx, i - 1, j)];
                float u_right = u_dest[IX_U(ctx, i + 1, j)];
                float u_bottom = u_dest[IX_U(ctx, i, j - 1)];
                float u_top = u_dest[IX_U(ctx, i, j + 1)];

                // Update u using the diffusion formula derived from the discretized diffusion equation.
                u_dest[IX_U(ctx, i, j)] = (u_src[IX_U(ctx, i, j)] + a * (u_left + u_right + u_bottom + u_top)) / (1.0f + 4.0f * a);
            }
        }

        // v (vertical) velocity diffusion
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y; j++) {
                if (ctx->solid[IX(ctx, i, j - 1)] || ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                
                float v_bottom = v_dest[IX_V(ctx, i, j - 1)];
                float v_top = v_dest[IX_V(ctx, i, j + 1)];
                float v_left = v_dest[IX_V(ctx, i - 1, j)];
                float v_right = v_dest[IX_V(ctx, i + 1, j)];

                v_dest[IX_V(ctx, i, j)] = (v_src[IX_V(ctx, i, j)] + a * (v_bottom + v_top + v_left + v_right)) / (1.0f + 4.0f * a);
            }
        }
    }
}

void diffuse_scalar(FluidContext* ctx, float* dest, const float* src) {

    mat_cpy(dest, (float*)src, ctx->x, ctx->y);

    float kinematic_visc = ctx->visc / ctx->dens;
    
    // Friction coefficient
    float a = ctx->dt * kinematic_visc / (ctx->dx * ctx->dx);

    // Gauss-Seidel iteration for diffusion
    for (size_t iter = 0; iter < ctx->iter_count; iter++) {
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                // Skip solid boundaries
                if (ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                // Fetch neighboring velocities (with no-slip condition at solids)
                float left = dest[IX(ctx, i - 1, j)];
                float right = dest[IX(ctx, i + 1, j)];
                float bottom = dest[IX(ctx, i, j - 1)];
                float top = dest[IX(ctx, i, j + 1)];

                // Update u using the diffusion formula derived from the discretized diffusion equation.
                dest[IX(ctx, i, j)] = (src[IX(ctx, i, j)] + a * (left + right + bottom + top)) / (1.0f + 4.0f * a);
            }
        }
    }
}

void advect_scalar(FluidContext* ctx, float* dest, const float* src, float* u, float* v) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {

            // Calculate the average velocity (i, j) at the center of the cell.
            // Since the u (horizontal) and v (vertical) velocities are at the edges, their average will be at the center.
            float u_avg = (u[IX_U(ctx, i, j)] + u[IX_U(ctx, i + 1, j)]) * 0.5f;
            float v_avg = (v[IX_V(ctx, i, j)] + v[IX_V(ctx, i, j + 1)]) * 0.5f;

            // Backtracing
            // We use indices like coordinates.
            float src_x = (float)i - ctx->dt * u_avg / ctx->dx;
            float src_y = (float)j - ctx->dt * v_avg / ctx->dx;

            // Clamp to boundary
            if (src_x < 0.5f)
                src_x = 0.5f;
            if (src_x > ctx->x - 1.5f)
                src_x = ctx->x - 1.5f;
            if (src_y < 0.5f)
                src_y = 0.5f;
            if (src_y > ctx->y - 1.5f)
                src_y = ctx->y - 1.5f;

            // Identify the surrounding 4 cells for bilinear interpolation.
            int i0 = (int)src_x;
            int i1 = i0 + 1;
            int j0 = (int)src_y;
            int j1 = j0 + 1;

            // Interpolation weights
            float sx1 = src_x - (float)i0;
            float sx0 = 1.0f - sx1;
            float sy1 = src_y - (float)j0;
            float sy0 = 1.0f - sy1;

            // Weighted average of 4 cells
            dest[IX(ctx, i, j)] = sx0 * (sy0 * src[IX(ctx, i0, j0)] + sy1 * src[IX(ctx, i0, j1)]) + sx1 * (sy0 * src[IX(ctx, i1, j0)] + sy1 * src[IX(ctx, i1, j1)]);
        }
    }
}

void advect_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src) {

    // u (horizontal) velocity update
    // u vectors are located on the vertical edges. Position: (i, j + 0.5)
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {

            float u = u_src[IX_U(ctx, i, j)];
            // To find the velocity v at point u, take the average of the 4 v's around it.
            float v = (v_src[IX_V(ctx, i - 1, j)] + v_src[IX_V(ctx, i, j)] + v_src[IX_V(ctx, i - 1, j + 1)] + v_src[IX_V(ctx, i, j + 1)]) * 0.25f;

            // Backtracking
            float src_x = (float)i - ctx->dt * u / ctx->dx;
            float src_y = ((float)j + 0.5f) - ctx->dt * v / ctx->dx;

            // Since the u-grid is shifted by 0.5 on the Y-axis, we subtract this from the index calculation.
            float idx_x = src_x;
            float idx_y = src_y - 0.5f;

            // Clamp for u
            if (idx_x < 0.5f)
                idx_x = 0.5f;
            if (idx_x > ctx->x - 0.5f)
                idx_x = ctx->x - 0.5f;
            if (idx_y < 0.5f)
                idx_y = 0.5f;
            if (idx_y > ctx->y - 1.5f)
                idx_y = ctx->y - 1.5f;

            int i0 = (int)idx_x;
            int i1 = i0 + 1;
            int j0 = (int)idx_y;
            int j1 = j0 + 1;

            float sx1 = idx_x - (float)i0;
            float sx0 = 1.0f - sx1;
            float sy1 = idx_y - (float)j0;
            float sy0 = 1.0f - sy1;

            // Calculate the new value of u using bilinear interpolation.
            u_dest[IX_U(ctx, i, j)] = sx0 * (sy0 * u_src[IX_U(ctx, i0, j0)] + sy1 * u_src[IX_U(ctx, i0, j1)]) + sx1 * (sy0 * u_src[IX_U(ctx, i1, j0)] + sy1 * u_src[IX_U(ctx, i1, j1)]);
        }
    }

    // v (vertical) velocity update
    // v vectors are located on the horizontal edges. Position: (i + 0.5, j)
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y; j++) {

            // To find the velocity u at point v, take the average of the 4 u's around it.
            float u = (u_src[IX_U(ctx, i, j - 1)] + u_src[IX_U(ctx, i + 1, j - 1)] + u_src[IX_U(ctx, i, j)] + u_src[IX_U(ctx, i + 1, j)]) * 0.25f;
            float v = v_src[IX_V(ctx, i, j)];

            // Backtracking
            float src_x = ((float)i + 0.5f) - ctx->dt * u / ctx->dx;
            float src_y = (float)j - ctx->dt * v / ctx->dx;

            // Since the v-grid is shifted by 0.5 on the X-axis, we subtract this from the index calculation.
            float idx_x = src_x - 0.5f;
            float idx_y = src_y;

            // Clamp for v
            if (idx_x < 0.5f)
                idx_x = 0.5f;
            if (idx_x > ctx->x - 1.5f)
                idx_x = ctx->x - 1.5f;
            if (idx_y < 0.5f)
                idx_y = 0.5f;
            if (idx_y > ctx->y - 0.5f)
                idx_y = ctx->y - 0.5f;

            int i0 = (int)idx_x;
            int i1 = i0 + 1;
            int j0 = (int)idx_y;
            int j1 = j0 + 1;

            float sx1 = idx_x - (float)i0;
            float sx0 = 1.0f - sx1;
            float sy1 = idx_y - (float)j0;
            float sy0 = 1.0f - sy1;

            // Calculate the new value of v using bilinear interpolation.
            v_dest[IX_V(ctx, i, j)] = sx0 * (sy0 * v_src[IX_V(ctx, i0, j0)] + sy1 * v_src[IX_V(ctx, i0, j1)]) + sx1 * (sy0 * v_src[IX_V(ctx, i1, j0)] + sy1 * v_src[IX_V(ctx, i1, j1)]);
        }
    }
}

void compute_divergence(FluidContext* ctx, float* u, float* v) {
    float inv_dx = 1.0f / ctx->dx;

    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i ,j)]) {
                ctx->div[IX(ctx, i, j)] = 0.0f;
                continue;
            }
            // Divergence = Right - Left + Top - Bottom
            float divergence = (u[IX_U(ctx, i + 1, j)] - u[IX_U(ctx, i, j)] + v[IX_V(ctx, i, j + 1)] - v[IX_V(ctx, i, j)]) * inv_dx;
            ctx->div[IX(ctx, i, j)] = divergence;
            ctx->p[IX(ctx, i, j)] = 0.0f;
        }
    }
}

void subtract_gradient(FluidContext* ctx, float* u, float* v, float* p) {
    float scale = ctx->dt / (ctx->dens * ctx->dx);

    // Horizontal velocity correction
    // i=1 (Left) to i=X-1 (Right)
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i - 1, j)] || ctx->solid[IX(ctx, i, j)])
                continue;
            u[IX_U(ctx, i, j)] -= scale * (p[IX(ctx, i, j)] - p[IX(ctx, i - 1, j)]);
        }
    }

    // Vertical velocity correction
    // j=1 (Bottom) to j=Y-1 (Top)
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j - 1)] || ctx->solid[IX(ctx, i, j)])
                continue;
            v[IX_V(ctx, i, j)] -= scale * (p[IX(ctx, i, j)] - p[IX(ctx, i, j - 1)]);
        }
    }
}

// Stride-2 access wastes 50% of every cache line
// The stride-1 + branch version was considered but performs the same since
// compute_poisson_update is a gather (4 neighbor reads), GCC won't vectorize
// either way without manual AVX2 intrinsics.
// If contigous red/black arrays ever get implemented, revisit this alongside SIMD.

void solve_pressure_rbgs(FluidContext* ctx, float* p, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

#ifdef VALIDATE
    double start_time = GET_TIME_SEC();
#endif // VALIDATE

    for (size_t iter = 0; iter < ctx->iter_count; iter++) {
        // Red pass - no error tracking needed here
        #pragma omp parallel for schedule(static)
        for (size_t i = 1; i < ctx->x - 1; i++) {
            size_t j_start = (i % 2 != 0) ? 1 : 2;
            for (size_t j = j_start; j < ctx->y - 1; j += 2) {
                if (ctx->solid[IX(ctx, i, j)]) continue;

                float p_new = compute_poisson_update(ctx, p, div, cp, i, j);
                p[IX(ctx, i, j)] = p[IX(ctx, i, j)] * (1.0f - ctx->omega) + p_new * ctx->omega;
            }
        }

        // Black pass - track error with OMP reduction
        float max_error = 0.0f;
        #pragma omp parallel for schedule(static) reduction(max:max_error)
        for (size_t i = 1; i < ctx->x - 1; i++) {
            size_t j_start = (i % 2 != 0) ? 2 : 1;
            for (size_t j = j_start; j < ctx->y - 1; j += 2) {
                if (ctx->solid[IX(ctx, i, j)]) continue;

                float p_old = p[IX(ctx, i, j)];
                float p_new = compute_poisson_update(ctx, p, div, cp, i, j);
                float p_updated = p_old * (1.0f - ctx->omega) + p_new * ctx->omega;

                p[IX(ctx, i, j)] = p_updated;
                max_error = MAX(max_error, fabsf(p_updated - p_old));
            }
        }

        if (max_error < ctx->threshold) {
#ifdef VALIDATE
            double end_time = GET_TIME_SEC();

            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[RBGS] Converged in %zu iters | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", iter + 1, max_error, true_residual, elapsed_ms);
#endif // VALIDATE
            break; // Convergence check
        }
#ifdef VALIDATE
        // Max Iteration Check
        if (iter == ctx->iter_count - 1) {
            double end_time = GET_TIME_SEC();
            
            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[RBGS] Max iters (%zu) reached | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", ctx->iter_count, max_error, true_residual, elapsed_ms);
        }
#endif // VALIDATE
    }
}

void solve_pressure_sor(FluidContext* ctx, float* p, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

#ifdef VALIDATE
    double start_time = GET_TIME_SEC();
#endif // VALIDATE

    for (size_t iter = 0; iter < ctx->iter_count; iter++) {
        float max_error = 0.0f;

        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                if (ctx->solid[IX(ctx, i, j)]) continue;

                float p_new = compute_poisson_update(ctx, p, div, cp, i, j);
                float p_old = p[IX(ctx, i, j)];

                p[IX(ctx, i, j)] = (1.0f - ctx->omega) * p_old + ctx->omega * p_new;
                max_error = MAX(max_error, fabsf(p[IX(ctx, i, j)] - p_old));
            }
        }

        if (max_error < ctx->threshold) {
#ifdef VALIDATE
            double end_time = GET_TIME_SEC();
            
            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[SOR] Converged in %zu iters | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", iter + 1, max_error, true_residual, elapsed_ms);
#endif // VALIDATE
            break; // Convergence check
        }
#ifdef VALIDATE
        // Max Iteration Check
        if (iter == ctx->iter_count - 1) {
            double end_time = GET_TIME_SEC();
            
            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[SOR] Max iters (%zu) reached | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", ctx->iter_count, max_error, true_residual, elapsed_ms);
        }
#endif
    }
}

FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, int iters, float threshold) {
    FluidContext* ctx = (FluidContext*)malloc(sizeof(FluidContext));
    
    ctx->x = res_x;
    ctx->y = res_y;
    ctx->num_cells = res_x * res_y;
    
    ctx->dt = dt;
    ctx->dx = dx;
    ctx->dens= dens;
    ctx->visc = visc;
    ctx->iter_count = iters;
    ctx->threshold = threshold;

    ctx->u = (float*)calloc((res_x + 1) * res_y, sizeof(float));
    ctx->v = (float*)calloc(res_x * (res_y + 1), sizeof(float));
    ctx->p = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->div = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->smoke = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->solid = (uint8_t*)calloc(ctx->num_cells, sizeof(uint8_t));

    ctx->u_prev = (float*)calloc((res_x + 1) * res_y, sizeof(float));
    ctx->v_prev = (float*)calloc(res_x * (res_y + 1), sizeof(float));
    ctx->smoke_prev = (float*)calloc(ctx->num_cells, sizeof(float));

    size_t N = ctx->x * ctx->y;
    ctx->cg_r = aligned_alloc(64, N * sizeof(float));
    ctx->cg_d = aligned_alloc(64, N * sizeof(float));
    ctx->cg_q = aligned_alloc(64, N * sizeof(float));

    return ctx;
}

void fluid_setup_physics(FluidContext* ctx, ScenarioParams p, PressureSolver pressure_solver) {
    ctx->pressure_solver = pressure_solver;

    // Calculate reynolds
    if (ctx->visc > 0.0f) { // avoid zero-divison
        ctx->reynolds = (ctx->dens * p.inlet_velocity * p.length_scale ) / ctx->visc;
    } else { // clamp
        ctx->reynolds = 0.0f;
    }

    // Find optimum omega for different scenarios
    if (p.target_omega <= 0.0f) {
        ctx->omega = 2.0f / (1.0f + sinf(PI / (float)ctx->x));
    } else {
        ctx->omega = p.target_omega;
    }

    if (ctx->omega >= 1.99f) {
        ctx->omega = 1.99f; // Clamp for upperbound
    } else if (ctx->omega < 1.0f) {
        ctx->omega = 1.0f;  // Back to gauss-seidel
    }
}

void fluid_destroy_context(FluidContext* ctx) {
    if (!ctx) return;
    free(ctx->u);
    free(ctx->v);
    free(ctx->p);
    free(ctx->div);
    free(ctx->smoke);
    free(ctx->solid);
    free(ctx->u_prev);
    free(ctx->v_prev);
    free(ctx->smoke_prev);
    free(ctx);

    free(ctx->cg_r);
    free(ctx->cg_d);
    free(ctx->cg_q);
}

void fluid_step(FluidContext* ctx, ScenarioParams p, Scenario s) {

    // Sources
    s.apply_sources(ctx, p);
    s.apply_boundaries(ctx, p);

    // Velocity Advection
    advect_velocity(ctx, ctx->u_prev, ctx->v_prev, ctx->u, ctx->v);
    
    // Swap pointers
    SWAP_PTR(ctx->u, ctx->u_prev);
    SWAP_PTR(ctx->v, ctx->v_prev);
    
    s.apply_boundaries(ctx, p);

    // Velocity Diffusion
    memcpy(ctx->u_prev, ctx->u, (ctx->x + 1) * ctx->y * sizeof(float));
    memcpy(ctx->v_prev, ctx->v, ctx->x * (ctx->y + 1) * sizeof(float));
    
    diffuse_velocity(ctx, ctx->u, ctx->v, ctx->u_prev, ctx->v_prev);
    
    s.apply_boundaries(ctx, p);

    // Projection

    // Divergence
    compute_divergence(ctx, ctx->u, ctx->v);
    
    // Pressure Solve

    ctx->pressure_solver(ctx, ctx->p, ctx->div);
    
    // Gradient Subtraction
    subtract_gradient(ctx, ctx->u, ctx->v, ctx->p);
    
    s.apply_boundaries(ctx, p);

    // Scalar Advection
    advect_scalar(ctx, ctx->smoke_prev, ctx->smoke, ctx->u, ctx->v);
    
    SWAP_PTR(ctx->smoke, ctx->smoke_prev);
    
    s.apply_boundaries(ctx, p);
}