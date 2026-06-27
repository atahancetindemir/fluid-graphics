#include "core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>

#include "types.h"
#include "preconditioners.h"
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

void diffuse_velocity_explicit(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src, float a) {

    // u diffusion
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i-1, j)] || ctx->solid[IX(ctx, i, j)]) {
                u_dest[IX_U(ctx, i, j)] = 0.0f;
                continue;
            }
            u_dest[IX_U(ctx, i, j)] = compute_explicit_u_update(ctx, u_src, a, i, j);
        }
    }

    // v diffusion
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j-1)] || ctx->solid[IX(ctx, i, j)]) {
                v_dest[IX_V(ctx, i, j)] = 0.0f;
                continue;
            }
            v_dest[IX_V(ctx, i, j)] = compute_explicit_v_update(ctx, v_src, a, i, j);
        }
    }
}

void diffuse_velocity_implicit(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src, float a) {

    mat_cpy(u_dest, (float*)u_src, ctx->x + 1, ctx->y);
    mat_cpy(v_dest, (float*)v_src, ctx->x, ctx->y + 1);

    for (size_t iter = 0; iter < ctx->diffuse_iter; iter++) {

        // u (horizontal) velocity diffusion
        for (size_t i = 1; i < ctx->x; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                if (ctx->solid[IX(ctx, i - 1, j)] || ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                u_dest[IX_U(ctx, i, j)] = compute_implicit_u_update(ctx, u_dest, u_src, a, i, j);
            }
        }

        // v (vertical) velocity diffusion
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y; j++) {
                if (ctx->solid[IX(ctx, i, j - 1)] || ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                v_dest[IX_V(ctx, i, j)] = compute_implicit_v_update(ctx, v_dest, v_src, a, i, j);
            }
        }
    }
}

void diffuse_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src) {
    float kinematic_visc = ctx->visc / ctx->dens;
    float a = ctx->dt * kinematic_visc / (ctx->dx * ctx->dx);

    a < 0.25 ? diffuse_velocity_explicit(ctx, u_dest, v_dest, u_src, v_src, a) : diffuse_velocity_implicit(ctx, u_dest, v_dest, u_src, v_src, a);
}

void diffuse_scalar_explicit(FluidContext* ctx, float* dest, const float* src, float a) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i, j)]) {
                dest[IX(ctx, i, j)] = 0.0f;
                continue;
            }
            dest[IX(ctx, i, j)] = compute_explicit_scalar_update(ctx, src, a, i, j);
        }
    }
}

void diffuse_scalar_implicit(FluidContext* ctx, float* dest, const float* src, float a) {

    mat_cpy(dest, (float*)src, ctx->x, ctx->y);

    for (size_t iter = 0; iter < ctx->diffuse_iter; iter++) {
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                if (ctx->solid[IX(ctx, i, j)]) {
                    continue;
                }
                dest[IX(ctx, i, j)] = compute_implicit_scalar_update(ctx, dest, src, a, i, j);
            }
        }
    }
}

void diffuse_scalar(FluidContext* ctx, float* dest, const float* src) {
    float kinematic_visc = ctx->visc / ctx->dens;
    float a = ctx->dt * kinematic_visc / (ctx->dx * ctx->dx);

    a < 0.25 ? diffuse_scalar_explicit(ctx, dest, src, a) : diffuse_scalar_implicit(ctx, dest, src, a);
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
            dest[IX(ctx, i, j)] = bilinear_interp(sx0, sx1, sy0, sy1, src[IX(ctx, i0, j0)], src[IX(ctx, i0, j1)], src[IX(ctx, i1, j0)], src[IX(ctx, i1, j1)]);
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
            u_dest[IX_U(ctx, i, j)] = bilinear_interp(sx0, sx1, sy0, sy1, u_src[IX_U(ctx, i0, j0)], u_src[IX_U(ctx, i0, j1)], u_src[IX_U(ctx, i1, j0)], u_src[IX_U(ctx, i1, j1)]);
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
            v_dest[IX_V(ctx, i, j)] = bilinear_interp(sx0, sx1, sy0, sy1, v_src[IX_V(ctx, i0, j0)], v_src[IX_V(ctx, i0, j1)], v_src[IX_V(ctx, i1, j0)], v_src[IX_V(ctx, i1, j1)]);
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

    for (size_t iter = 0; iter < ctx->poisson_iter; iter++) {
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
        if (iter == ctx->poisson_iter - 1) {
            double end_time = GET_TIME_SEC();
            
            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[RBGS] Max iters (%zu) reached | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", ctx->poisson_iter, max_error, true_residual, elapsed_ms);
        }
#endif // VALIDATE
    }
}

void solve_pressure_sor(FluidContext* ctx, float* p, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

#ifdef VALIDATE
    double start_time = GET_TIME_SEC();
#endif // VALIDATE

    for (size_t iter = 0; iter < ctx->poisson_iter; iter++) {
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
        if (iter == ctx->poisson_iter - 1) {
            double end_time = GET_TIME_SEC();
            
            float true_residual = calculate_max_residual(ctx, p, div);
            double elapsed_ms = (end_time - start_time) * 1000.0;
            
            printf("[SOR] Max iters (%zu) reached | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", ctx->poisson_iter, max_error, true_residual, elapsed_ms);
        }
#endif
    }
}

void solve_pressure_pcg(FluidContext* ctx, float* p, const float* div) {
    size_t N = ctx->num_cells;
    float *r = ctx->cg_r;
    float *d = ctx->cg_d;
    float *q = ctx->cg_q;
    float *z = ctx->cg_z;

    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

#ifdef VALIDATE
    double start_time = GET_TIME_SEC();
#endif

    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            size_t idx = IX(ctx, i, j);
            if (ctx->solid[idx]) {
                r[idx] = 0.0f;
                d[idx] = 0.0f;
                continue;
            }
            float Ap = apply_scalar_laplacian(ctx, p, i, j);
            r[idx] = div[idx] * cp - Ap;
        }
    }

    ctx->precondition(ctx, z, r);
    memcpy(d, z, N * sizeof(float));

    float delta_new = dot(r, z, N);

    for (size_t iter = 0; iter < ctx->poisson_iter; iter++) {

        #pragma omp parallel for schedule(static)
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                size_t idx = IX(ctx, i, j);
                if (ctx->solid[idx]) {
                    q[idx] = 0.0f;
                    continue;
                }
                q[idx] = apply_scalar_laplacian(ctx, d, i, j);
            }
        }

        float alpha = delta_new / dot(d, q, N);

        float max_change = 0.0f;
        #pragma omp parallel for simd schedule(static) reduction(max:max_change)
        for (size_t k = 0; k < N; k++) {
            float change = alpha * d[k];
            p[k] += change;
            r[k] -= alpha * q[k];
            max_change = MAX(max_change, fabsf(change));
        }

        if (max_change < ctx->threshold) {
#ifdef VALIDATE
            double end_time = GET_TIME_SEC();
            float true_residual = calculate_max_residual(ctx, p, div);
            printf("[PCG] Converged in %zu iters | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", iter + 1, max_change, true_residual, (end_time - start_time) * 1000.0);
#endif
            break;
        }
#ifdef VALIDATE
        if (iter == ctx->poisson_iter - 1) {
            double end_time = GET_TIME_SEC();
            float true_residual = calculate_max_residual(ctx, p, div);
            printf("[PCG] Max iters (%zu) reached | Delta P: %.6e | True Res: %.6e | Time: %.2f ms\n", ctx->poisson_iter, max_change, true_residual, (end_time - start_time) * 1000.0);
        }
#endif

        ctx->precondition(ctx, z, r);        

        float delta_old = delta_new;
        delta_new = dot(r, z, N);
        float beta = delta_new / delta_old;

        #pragma omp parallel for simd schedule(static)
        for (size_t k = 0; k < N; k++)
            d[k] = z[k] + beta * d[k];
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
    ctx->poisson_iter = iters;
    ctx->diffuse_iter = 20;
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

    ctx->cg_r = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->cg_d = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->cg_q = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->cg_z = (float*)calloc(ctx->num_cells, sizeof(float));

    return ctx;
}

void fluid_setup_physics(FluidContext* ctx, ScenarioParams p, PressureSolver pressure_solver, PrecondType precondition) {
    ctx->pressure_solver = pressure_solver;

    switch (precondition) {
        case PRECOND_IDENTITY:
            ctx->precondition = precondition_identity;
            break;
        case PRECOND_JACOBI:
            ctx->precondition = precondition_jacobi;
            break;
        case PRECOND_MULTIGRID:
            ctx->precondition = precondition_multigrid;
            break;
        default: // fallback to identity
            ctx->precondition = precondition_identity;
            break;
    }

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

    free(ctx->cg_r);
    free(ctx->cg_d);
    free(ctx->cg_q);
    free(ctx->cg_z);

    free(ctx);
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
    mat_zero_float(ctx->div, ctx->x, ctx->y);
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