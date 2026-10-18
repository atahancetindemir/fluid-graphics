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

    for (size_t iter = 0; iter < DIFFUSE_ITER; iter++) {

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

    for (size_t iter = 0; iter < DIFFUSE_ITER; iter++) {
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

            // Solid samples take the cell's own value
            const float own = src[IX(ctx, i, j)];
            const float s00 = ctx->solid[IX(ctx, i0, j0)] ? own : src[IX(ctx, i0, j0)];
            const float s01 = ctx->solid[IX(ctx, i0, j1)] ? own : src[IX(ctx, i0, j1)];
            const float s10 = ctx->solid[IX(ctx, i1, j0)] ? own : src[IX(ctx, i1, j0)];
            const float s11 = ctx->solid[IX(ctx, i1, j1)] ? own : src[IX(ctx, i1, j1)];

            // Weighted average of 4 cells
            dest[IX(ctx, i, j)] = bilinear_interp(sx0, sx1, sy0, sy1, s00, s01, s10, s11);
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

float calculate_max_rhs(const FluidContext* ctx, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;
    float max_div = 0.0f;
    #pragma omp parallel for reduction(max:max_div)
    for (size_t k = 0; k < ctx->num_cells; k++)
        max_div = MAX(max_div, fabsf(div[k]));
    return max_div * cp;
}

float calculate_residual_target(const FluidContext* ctx, float rhs) {
    return MAX(ctx->tol.rtol * rhs, ctx->tol.atol);
}

void record_solve_stats(FluidContext* ctx, const char* tag, SolveStats stats, double start_time) {
    stats.solve_ms = (GET_TIME_SEC() - start_time) * 1000.0;
    ctx->stats = stats;

#ifdef DEBUG
    solver_report(tag, fluid_solve_status_name(stats.status), stats.iterations, stats.delta_p, stats.residual,
                  stats.rhs > 0.0f ? stats.residual / stats.rhs : 0.0f, stats.solve_ms);
#else
    UNUSED(tag);
#endif // DEBUG
}

// Stride-2 access wastes 50% of every cache line
// The stride-1 + branch version was considered but performs the same since
// compute_poisson_update is a gather (4 neighbor reads), GCC won't vectorize
// either way without manual AVX2 intrinsics.
// If contigous red/black arrays ever get implemented, revisit this alongside SIMD.

void solve_pressure_rbgs(FluidContext* ctx, float* p, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

    double start_time = GET_TIME_SEC();
    float rhs = calculate_max_rhs(ctx, div);
    float target = calculate_residual_target(ctx, rhs);
    float initial_residual = calculate_max_residual(ctx, p, div);
    float residual = initial_residual;
    size_t iterations = 0;
    float max_error = 0.0f;

    while (residual > target && iterations < ctx->tol.max_iter) {
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
        max_error = 0.0f;
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

        iterations++;
        if (iterations % RESIDUAL_CHECK_INTERVAL == 0 || iterations == ctx->tol.max_iter)
            residual = calculate_max_residual(ctx, p, div);
    }

    record_solve_stats(ctx, "RBGS", (SolveStats){
        .status = residual <= target ? SOLVE_CONVERGED : SOLVE_CAPPED,
        .iterations = iterations,
        .delta_p = max_error,
        .initial_residual = initial_residual,
        .residual = residual,
        .rhs = rhs,
    }, start_time);
}

void solve_pressure_sor(FluidContext* ctx, float* p, const float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

    double start_time = GET_TIME_SEC();
    float rhs = calculate_max_rhs(ctx, div);
    float target = calculate_residual_target(ctx, rhs);
    float initial_residual = calculate_max_residual(ctx, p, div);
    float residual = initial_residual;
    size_t iterations = 0;
    float max_error = 0.0f;

    while (residual > target && iterations < ctx->tol.max_iter) {
        max_error = 0.0f;

        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                if (ctx->solid[IX(ctx, i, j)]) continue;

                float p_new = compute_poisson_update(ctx, p, div, cp, i, j);
                float p_old = p[IX(ctx, i, j)];

                p[IX(ctx, i, j)] = (1.0f - ctx->omega) * p_old + ctx->omega * p_new;
                max_error = MAX(max_error, fabsf(p[IX(ctx, i, j)] - p_old));
            }
        }

        iterations++;
        if (iterations % RESIDUAL_CHECK_INTERVAL == 0 || iterations == ctx->tol.max_iter)
            residual = calculate_max_residual(ctx, p, div);
    }

    record_solve_stats(ctx, "SOR", (SolveStats){
        .status = residual <= target ? SOLVE_CONVERGED : SOLVE_CAPPED,
        .iterations = iterations,
        .delta_p = max_error,
        .initial_residual = initial_residual,
        .residual = residual,
        .rhs = rhs,
    }, start_time);
}

void solve_pressure_pcg(FluidContext* ctx, float* p, const float* div) {
    size_t N = ctx->num_cells;
    float* r = ctx->cg.r;
    float* d = ctx->cg.d;
    float* q = ctx->cg.q;
    float* z = ctx->cg.z;

    double start_time = GET_TIME_SEC();
    float rhs = calculate_max_rhs(ctx, div);
    float target = calculate_residual_target(ctx, rhs);
    size_t iterations = 0;
    size_t replacements = 0;
    float max_change = 0.0f;
    int broke_down = 0;

    float delta_new = 0.0f;
    float initial_residual = pcg_reset(ctx, p, div, &delta_new);
    float recurrence_residual = initial_residual;
    float residual = initial_residual;

    for (;;) {
        while (recurrence_residual > target && iterations < ctx->tol.max_iter) {

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
            if (!isfinite(alpha)) {
                broke_down = 1;
                break;
            }

            max_change = 0.0f;
            recurrence_residual = 0.0f;
            #pragma omp parallel for simd schedule(static) reduction(max:max_change, recurrence_residual)
            for (size_t k = 0; k < N; k++) {
                float change = alpha * d[k];
                p[k] += change;
                r[k] -= alpha * q[k];
                max_change = MAX(max_change, fabsf(change));
                recurrence_residual = MAX(recurrence_residual, fabsf(r[k]));
            }

            iterations++;
            if (recurrence_residual <= target)
                break;

            ctx->precondition(ctx, z, r);

            float delta_old = delta_new;
            delta_new = dot(r, z, N);
            float beta = delta_new / delta_old;
            if (!isfinite(beta)) {
                broke_down = 1;
                break;
            }

            #pragma omp parallel for simd schedule(static)
            for (size_t k = 0; k < N; k++)
                d[k] = z[k] + beta * d[k];
        }

        residual = calculate_max_residual(ctx, p, div);
        if (residual <= target || broke_down || iterations >= ctx->tol.max_iter ||
            replacements == PCG_MAX_REPLACEMENTS)
            break;

        // True residual missed the target, restart from it
        recurrence_residual = pcg_reset(ctx, p, div, &delta_new);
        replacements++;
    }

    SolveStatus status = SOLVE_CONVERGED;
    if (residual > target)
        status = (broke_down || iterations < ctx->tol.max_iter) ? SOLVE_STALLED : SOLVE_CAPPED;

    record_solve_stats(ctx, "PCG", (SolveStats){
        .status = status,
        .iterations = iterations,
        .replacements = replacements,
        .delta_p = max_change,
        .initial_residual = initial_residual,
        .residual = residual,
        .rhs = rhs,
    }, start_time);
}

float pcg_reset(FluidContext* ctx, const float* p, const float* div, float* delta) {
    size_t N = ctx->num_cells;
    float* r = ctx->cg.r;
    float* d = ctx->cg.d;
    float* z = ctx->cg.z;
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

    float max_r = 0.0f;
    #pragma omp parallel for schedule(static) reduction(max:max_r)
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
            max_r = MAX(max_r, fabsf(r[idx]));
        }
    }

    ctx->precondition(ctx, z, r);
    memcpy(d, z, N * sizeof(float));

    *delta = dot(r, z, N);
    return max_r;
}

FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, SolveTolerance tol) {
    FluidContext* ctx = (FluidContext*)malloc(sizeof(FluidContext));
    
    ctx->x = res_x;
    ctx->y = res_y;
    ctx->num_cells = res_x * res_y;
    
    ctx->dt = dt;
    ctx->dx = dx;
    ctx->dens= dens;
    ctx->visc = visc;
    ctx->tol = tol;

    ctx->u = (float*)malloc((res_x + 1) * res_y * sizeof(float));
    ctx->v = (float*)malloc(res_x * (res_y + 1) * sizeof(float));
    ctx->p = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->div = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->smoke = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->solid = (uint8_t*)malloc(ctx->num_cells * sizeof(uint8_t));

    ctx->u_prev = (float*)malloc((res_x + 1) * res_y * sizeof(float));
    ctx->v_prev = (float*)malloc(res_x * (res_y + 1) * sizeof(float));
    ctx->smoke_prev = (float*)malloc(ctx->num_cells * sizeof(float));

    ctx->cg.r = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->cg.d = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->cg.q = (float*)malloc(ctx->num_cells * sizeof(float));
    ctx->cg.z = (float*)malloc(ctx->num_cells * sizeof(float));

    // Which fields get zeroed lives in exactly one place: fluid_reset_context.
    fluid_reset_context(ctx);

    return ctx;
}

void fluid_reset_context(FluidContext* ctx) {
    if (!ctx) return;

    size_t u_count = (ctx->x + 1) * ctx->y;
    size_t v_count = ctx->x * (ctx->y + 1);

    memset(ctx->u, 0, u_count * sizeof(float));
    memset(ctx->v, 0, v_count * sizeof(float));
    memset(ctx->p, 0, ctx->num_cells * sizeof(float));
    memset(ctx->div, 0, ctx->num_cells * sizeof(float));
    memset(ctx->smoke, 0, ctx->num_cells * sizeof(float));
    memset(ctx->solid, 0, ctx->num_cells * sizeof(uint8_t));
    ctx->step = 0;
    ctx->time = 0.0;

    memset(ctx->u_prev, 0, u_count * sizeof(float));
    memset(ctx->v_prev, 0, v_count * sizeof(float));
    memset(ctx->smoke_prev, 0, ctx->num_cells * sizeof(float));

    // PCG only writes interior cells, so the boundary entries must start at zero.
    memset(ctx->cg.r, 0, ctx->num_cells * sizeof(float));
    memset(ctx->cg.d, 0, ctx->num_cells * sizeof(float));
    memset(ctx->cg.q, 0, ctx->num_cells * sizeof(float));
    memset(ctx->cg.z, 0, ctx->num_cells * sizeof(float));

    memset(&ctx->stats, 0, sizeof(ctx->stats));
    ctx->continuity_rms = 0.0f;
    ctx->continuity_max = 0.0f;
    memset(&ctx->mass, 0, sizeof(ctx->mass));
}

const char* fluid_solve_status_name(SolveStatus status) {
    switch (status) {
        case SOLVE_CONVERGED: return "converged";
        case SOLVE_CAPPED:    return "capped";
        case SOLVE_STALLED:   return "stalled";
    }
    return "unknown";
}

float fluid_optimal_omega(const FluidContext* ctx) {
    // Jacobi spectral radius of the 5-point Laplacian on an x by y grid
    float rho = 0.5f * (cosf(PI / (float)ctx->x) + cosf(PI / (float)ctx->y));
    float omega = 2.0f / (1.0f + sqrtf(1.0f - rho * rho));

    if (omega >= 1.99f) {
        return 1.99f;   // Clamp for upperbound
    } else if (omega < 1.0f) {
        return 1.0f;    // Back to gauss-seidel
    }
    return omega;
}

float fluid_reynolds(const FluidContext* ctx, ScenarioParams p) {
    if (ctx->visc > 0.0f) { // avoid zero-divison
        return (ctx->dens * p.inlet_velocity * p.length_scale) / ctx->visc;
    }
    return 0.0f; // clamp
}

void fluid_set_preconditioner(FluidContext* ctx, PrecondType precondition) {
    switch (precondition) {
        case PRECOND_JACOBI:
            ctx->precondition = precondition_jacobi;
            break;
        case PRECOND_MULTIGRID:
            ctx->precondition = precondition_multigrid;
            break;
        case PRECOND_IDENTITY:
        default: // fallback to identity
            ctx->precondition = precondition_identity;
            break;
    }
}

void fluid_velocity_magnitude(const FluidContext* ctx, float* dest) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            float u_center = sample_u_centered(ctx, ctx->u, i, j);
            float v_center = sample_v_centered(ctx, ctx->v, i, j);
            dest[IX(ctx, i, j)] = sqrtf(u_center * u_center + v_center * v_center);
        }
    }
}

void fluid_velocity_centered(const FluidContext* ctx, float* u_center, float* v_center) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            u_center[IX(ctx, i, j)] = sample_u_centered(ctx, ctx->u, i, j);
            v_center[IX(ctx, i, j)] = sample_v_centered(ctx, ctx->v, i, j);
        }
    }
}

void fluid_setup_physics(FluidContext* ctx, PressureSolver pressure_solver, PrecondType precondition) {
    ctx->pressure_solver = pressure_solver;

    fluid_set_preconditioner(ctx, precondition);
    ctx->omega = fluid_optimal_omega(ctx);
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

    free(ctx->cg.r);
    free(ctx->cg.d);
    free(ctx->cg.q);
    free(ctx->cg.z);

    free(ctx);
}

void measure_continuity(FluidContext* ctx, ScenarioParams p) {
    // Without a characteristic velocity the flux is reported unscaled.
    float inv_u = p.inlet_velocity > 0.0f ? 1.0f / p.inlet_velocity : 1.0f;
    double sum_sq = 0.0;
    size_t fluid_cells = 0;
    float max_err = 0.0f;

    #pragma omp parallel for schedule(static) reduction(+:sum_sq, fluid_cells) reduction(max:max_err)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i, j)]) continue;

            float flux = ctx->u[IX_U(ctx, i + 1, j)] - ctx->u[IX_U(ctx, i, j)]
                       + ctx->v[IX_V(ctx, i, j + 1)] - ctx->v[IX_V(ctx, i, j)];
            float err = fabsf(flux) * inv_u;
            sum_sq += (double)err * err;
            fluid_cells++;
            max_err = MAX(max_err, err);
        }
    }

    ctx->continuity_rms = fluid_cells ? (float)sqrt(sum_sq / (double)fluid_cells) : 0.0f;
    ctx->continuity_max = max_err;
}

void measure_mass_balance(FluidContext* ctx) {
    double inflow = 0.0, outflow = 0.0;
    double left = 0.0, right = 0.0, bottom = 0.0, top = 0.0;

    // Positive u/v enters on the left/bottom and leaves on the right/top
    for (size_t j = 1; j < ctx->y - 1; j++) {
        double in_left = ctx->u[IX_U(ctx, 1, j)];
        double out_right = ctx->u[IX_U(ctx, ctx->x - 1, j)];
        left -= in_left;
        right += out_right;
        inflow += MAX(in_left, 0.0) + MAX(-out_right, 0.0);
        outflow += MAX(-in_left, 0.0) + MAX(out_right, 0.0);
    }
    for (size_t i = 1; i < ctx->x - 1; i++) {
        double in_bottom = ctx->v[IX_V(ctx, i, 1)];
        double out_top = ctx->v[IX_V(ctx, i, ctx->y - 1)];
        bottom -= in_bottom;
        top += out_top;
        inflow += MAX(in_bottom, 0.0) + MAX(-out_top, 0.0);
        outflow += MAX(-in_bottom, 0.0) + MAX(out_top, 0.0);
    }

    double dx = ctx->dx;
    ctx->mass = (MassBalance){
        .inflow = (float)(inflow * dx),
        .outflow = (float)(outflow * dx),
        .left = (float)(left * dx),
        .right = (float)(right * dx),
        .bottom = (float)(bottom * dx),
        .top = (float)(top * dx),
    };
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
    memcpy(ctx->smoke_prev, ctx->smoke, ctx->num_cells * sizeof(float));
    advect_scalar(ctx, ctx->smoke_prev, ctx->smoke, ctx->u, ctx->v);
    
    SWAP_PTR(ctx->smoke, ctx->smoke_prev);

    s.apply_boundaries(ctx, p);

    measure_continuity(ctx, p);
    measure_mass_balance(ctx);

    ctx->step++;
    ctx->time += ctx->dt;
}