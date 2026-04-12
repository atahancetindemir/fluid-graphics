#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

#define SCENARIO_KARMAN

#ifdef SCENARIO_LID_DRIVEN
    #define INIT_SCENARIO()       init_lid_driven(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_lid_driven(ctx, p)
    #define APPLY_SOURCES()       apply_sources_lid_driven(ctx)
#elif defined(SCENARIO_KARMAN)
    #define INIT_SCENARIO()       init_karman_vortex(ctx, p)
    #define APPLY_BOUNDARIES()    apply_boundaries_karman_vortex(ctx, p)
    #define APPLY_SOURCES()       apply_sources_karman_vortex(ctx)
#elif defined(SCENARIO_AIRFOIL)
    #define INIT_SCENARIO()       init_airfoil(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_airfoil(ctx)
    #define APPLY_SOURCES()       apply_sources_airfoil(ctx)
#elif defined(SCENARIO_WIND_OVER_CITY)
    #define INIT_SCENARIO()       init_wind_over_city(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_wind_over_city(ctx)
    #define APPLY_SOURCES()       apply_sources_wind_over_city(ctx)
#endif

// typedef void (*PressureSolver)(FluidContext* ctx, float* p, float* div);

// Fluid Simulation Context
typedef struct {

    // Domain Dimensions
    size_t x;               // Grid width in pixels
    size_t y;               // Grid height in pixels
    size_t num_cells;       // grid height * grid width in pixels

    // Physics Properties
    float dt;               // Time step in seconds
    float dx;               // Grid size in meters
    float dens;             // Density of the fluid
    float visc;             // Dynamic viscosity of the fluid
    float reynolds;         // Reynolds number
    float omega;            // Over-relaxation factor for pressure solver
    int iter_count;         // Iteration count for solver
    // PressureSolver pressure_solver;

    // Vector Fields
    float* u;               // Horizontal velocity
    float* v;               // Vertical velocity
    
    // Solver Fields
    float* p;               // Pressure
    float* div;             // Divergence
    
    // Transport Fields
    float* smoke;           // Smoke
    
    // Geometry
    uint8_t* solid;         // Boundary Mask

    // Previous State
    float* u_prev;          // Previous horizontal velocity
    float* v_prev;          // Previous vertical velocity
    float* smoke_prev;      // Previous smoke

} FluidContext;

typedef struct {
    float inlet_velocity;   // Characteristic velocity for the scenario
    float length_scale;     // Characteristic length scale for the scenario
    float target_omega;     // Target over-relaxation factor for the pressure solver

    float obstacle_x;       // X-coordinate of the center of the obstacle
    float obstacle_y;       // Y-coordinate of the center of the obstacle
    size_t obstacle_radius; // Radius of the obstacle
} ScenarioParams;

#define IX(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_U(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_V(ctx, i, j) ((size_t)(i) * ((ctx)->y + 1) + (size_t)(j))

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define SWAP_PTR(x0, x) do { float* tmp = x0; x0 = x; x = tmp; } while(0)

// // Layout
// #define X 128
// #define Y 128

// // Parameters
// #define DT 0.016f
// #define DX 0.1f
// #define DENSITY 1.0f
// #define ITER 20
#define SEED 54
// #define VISCOSITY 0.01f

// // Grid Cell Type
// unsigned char solid[X][Y];

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

// ################################### Utils ###################################

// Initializes rng with a seed
void rng_init_seed(uint32_t seed) {
    srand(seed);
}

// Initializes rng with current time
void rng_init_time(void) {
    srand((uint32_t)time(NULL));
}

// Generates unbiased random float in [0, 1)
float rng_urand01(void) {
    float inv_rand_max_plus_one = 1.0f / ((float)RAND_MAX + 1.0f);
    return (float)rand() * inv_rand_max_plus_one;
}

// Generates unbiased random float in [0, 1]
float rng_urand01_closed(void) {
    float inv_rand_max = 1.0f / (float)RAND_MAX;
    return (float)rand() * inv_rand_max;
}

// Generates unbiased random float in [min, max]
float rng_urand_range(float min, float max) {
    float u = rng_urand01_closed();
    return min + u * (max - min);
}

// Display matrix
void mat_display(float* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%.3f\t", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Copy src to dest
void mat_cpy(float* dest, float* src, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dest[i * cols + j] = src[i * cols + j];
        }
    }
}

// Zero out the matrix
void mat_zero_float(float* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = 0.0f;
        }
    }
}

// Zero out the matrix
void mat_zero_uint(uint8_t* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = 0;
        }
    }
}

// Fill the matrix with specific value
void mat_value(float* mat, size_t rows, size_t cols, float val) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = val;
        }
    }
}

// Fill the matrix with random values in given range.
void mat_random(float* mat, size_t rows, size_t cols, float min, float max) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = rng_urand_range(min, max);
        }
    }
}

// Save matrix to a file
void mat_save(float* mat, const char* filename, int rows, int cols, int stride) {
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        return;
    }
        
    // We save the matrix in column-major order to match the way we visualize it later.
    for (int j = cols - 1; j >= 0; j--) {
        for (int i = 0; i < rows; i++) {
            fprintf(f, "%f ", mat[i * stride + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// ################################### Utils ###################################

// ################################# Scenarios #################################

void init_lid_driven(FluidContext* ctx) {
    // Solid boundaries.
    for (size_t i = 0; i < ctx->x; i++) {
        ctx->solid[IX(ctx, i, 0)] = 1;          // Top
        ctx->solid[IX(ctx, i, ctx->y - 1)] = 1; // Bottom
    }
    for (size_t j = 0; j < ctx->y; j++) {
        ctx->solid[IX(ctx, 0, j)] = 1;          // Left
        ctx->solid[IX(ctx, ctx->x - 1, j)] = 1; // Right
    }

    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j)]) {
                ctx->smoke[IX(ctx, i, j)] = 0.0f;
            }
        }
    }
}

void apply_sources_lid_driven(FluidContext* ctx) {
    for (size_t i = 3; i < ctx->x - 3; i++) {
        ctx->smoke[IX(ctx, i, ctx->y - 3)] = 1.0f;
    }
}

void apply_boundaries_lid_driven(FluidContext* ctx, ScenarioParams p) {
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j)]) {
                ctx->u[IX_U(ctx, i, j)] = 0.0f;
                ctx->u[IX_U(ctx, i + 1, j)] = 0.0f;
                ctx->v[IX_V(ctx, i, j)] = 0.0f;
                ctx->v[IX_V(ctx, i, j + 1)] = 0.0f;

                // Top
                if (j == ctx->y - 1) {
                    ctx->u[IX_U(ctx, i, j)] = p.inlet_velocity;
                    ctx->u[IX_U(ctx, i + 1, j)] = p.inlet_velocity;
                }
            }
        }
    }
}

void init_karman_vortex(FluidContext* ctx, ScenarioParams p) {
    for (size_t i = 0; i < ctx->x; i++) {
        ctx->solid[IX(ctx, i, 0)] = 1;          // Bottom
        ctx->solid[IX(ctx, i, ctx->y - 1)] = 1; // Top
    }

    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if ((i - p.obstacle_x) * (i - p.obstacle_x) + (j - p.obstacle_y) * (j - p.obstacle_y) <= p.obstacle_radius * p. obstacle_radius) {
                ctx->solid[IX(ctx, i, j)] = 1;
            }
        }
    }

    // Start wind
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (!(ctx->solid[IX(ctx, i, j)]) && !(ctx->solid[IX(ctx, i - 1, j)])) {
                ctx->u[IX_U(ctx, i, j)] = p.inlet_velocity;
            }
        }
    }
}

void apply_sources_karman_vortex(FluidContext* ctx) {
    for (size_t j = 1; j < ctx->y - 1; j++) {
        if (j > (ctx->y / 2 - 2) && j < (ctx->y / 2 + 2)) {
            ctx->smoke[IX(ctx, 2, j)] = 1.0f;
        }
    }
}

void apply_boundaries_karman_vortex(FluidContext* ctx, ScenarioParams p) {
    // Zero out velocities at solid boundaries (No-Slip condition)
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j)]) {
                ctx->u[IX_U(ctx, i, j)] = 0.0f;
                ctx->u[IX_U(ctx, i + 1, j)] = 0.0f;
                ctx->v[IX_V(ctx, i, j)] = 0.0f;
                ctx->v[IX_V(ctx, i, j + 1)] = 0.0f;
            }
        }
    }

    // Inlet (Left Wall)
    for (size_t j = 1; j < ctx->y - 1; j++) {
        ctx->u[IX_U(ctx, 1, j)] = p.inlet_velocity;
    }    

    // Outlet
    for (size_t j = 1; j < ctx->y - 1; j++) {
        ctx->u[IX_U(ctx, ctx->x, j)] = ctx->u[IX_U(ctx, ctx->x - 1, j)]; 
    }

}

void init_airfoil(void) {

}

void apply_sources_airfoil(void) {

}

void apply_boundaries_airfoil(void) {

}

void init_wind_over_city(void) {

}

void apply_sources_wind_over_city(void) {

}

void apply_boundaries_wind_over_city(void) {

}

// ################################# Scenarios #################################

// Diffuses the velocity field using Gauss-Seidel iteration.
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

// Diffuses the scalar field with Gauss-Seidel iteration.
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
                // 
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

// Advect the scalar field with given velocities using semi-Lagrangian advection.
void advect_scalar(FluidContext* ctx, float* dest, const float* src, float* u, float* v) {

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

// Advects the velocity field with self-advection using semi-Lagrangian advection.
void advect_velocity(FluidContext* ctx, float* u_dest, float* v_dest, const float* u_src, const float* v_src) {

    // u (horizontal) velocity update
    // u vectors are located on the vertical edges. Position: (i, j + 0.5)
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

// Computes the divergence of the velocity field.
void compute_divergence(FluidContext* ctx, float* u, float* v) {
    float inv_dx = 1.0f / ctx->dx;

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

// Solves the pressure Poisson equation using Red-Black Gauss-Seidel.
void solve_pressure_rbgs(FluidContext* ctx, float* p, float* div) {

}

// Solves the pressure Poisson equation using successive over-relaxation.
void solve_pressure_sor(FluidContext* ctx, float* p, float* div) {
    float cp = (ctx->dens * ctx->dx * ctx->dx) / ctx->dt;

    for (size_t iter = 0; iter < ctx->iter_count; iter++) {
#ifdef VALIDATE
        float max_error = 0.0f;
#endif // VALIDATE
        for (size_t i = 1; i < ctx->x - 1; i++) {
            for (size_t j = 1; j < ctx->y - 1; j++) {
                if (ctx->solid[IX(ctx, i, j)]) continue;

                // P = (Left + Right + Bottom + Top - Divergence * cp) / 4
                // If the neighbor is solid, we use the current cell's pressure for that neighbor (Neumann boundary condition).

                float p_left = ctx->solid[IX(ctx, i - 1, j)] ? p[IX(ctx, i, j)] : p[IX(ctx, i - 1, j)];
                float p_right = ctx->solid[IX(ctx, i + 1, j)] ? p[IX(ctx, i, j)] : p[IX(ctx, i + 1, j)];
                float p_bottom = ctx->solid[IX(ctx, i, j - 1)] ? p[IX(ctx, i, j)] : p[IX(ctx, i, j - 1)];
                float p_top = ctx->solid[IX(ctx, i, j + 1)] ? p[IX(ctx, i, j)] : p[IX(ctx, i, j + 1)];

                float p_new = (p_left + p_right + p_bottom + p_top - div[IX(ctx, i, j)] * cp) * 0.25f;
                float p_old = p[IX(ctx, i, j)];

                p[IX(ctx, i, j)] = (1.0f - ctx->omega) * p_old + ctx->omega * p_new; // SOR update
#ifdef VALIDATE
                max_error = MAX(max_error, fabsf(p[IX(ctx, i, j)] - p_old));
#endif // VALIDATE
                
            }
        }
#ifdef VALIDATE
        if (max_error < 1e-4f) {
            printf("Pressure solver converged in %zu iterations with max error %.6f\n", iter + 1, max_error);
            break; // Convergence check
        }
        if (iter == ctx->iter_count - 1) {
            printf("Pressure solver reached max iterations with max error %.6f\n", max_error);
        }
#endif // VALIDATE
    }
}

// Subtracts the pressure gradient from the velocity field to enforce incompressibility.
void subtract_gradient(FluidContext* ctx, float* u, float* v, float* p) {
    float scale = ctx->dt / (ctx->dens * ctx->dx);

    // Horizontal velocity correction
    // i=1 (Left) to i=X-1 (Right)
    for (size_t i = 1; i < ctx->x; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            if (ctx->solid[IX(ctx, i - 1, j)] || ctx->solid[IX(ctx, i, j)])
                continue;
            u[IX_U(ctx, i, j)] -= scale * (p[IX(ctx, i, j)] - p[IX(ctx, i - 1, j)]);
        }
    }

    // Vertical velocity correction
    // j=1 (Bottom) to j=Y-1 (Top)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j - 1)] || ctx->solid[IX(ctx, i, j)])
                continue;
            v[IX_V(ctx, i, j)] -= scale * (p[IX(ctx, i, j)] - p[IX(ctx, i, j - 1)]);
        }
    }
}

// Start the simulation by creating a context with given parameters.
FluidContext* fluid_create_context(size_t res_x, size_t res_y, float dt, float dx, float dens, float visc, int iters) {
    FluidContext* ctx = (FluidContext*)malloc(sizeof(FluidContext));
    
    ctx->x = res_x;
    ctx->y = res_y;
    ctx->num_cells = res_x * res_y;
    
    ctx->dt = dt;
    ctx->dx = dx;
    ctx->dens= dens;
    ctx->visc = visc;
    ctx->iter_count = iters;

    ctx->u = (float*)calloc((res_x + 1) * res_y, sizeof(float));
    ctx->v = (float*)calloc(res_x * (res_y + 1), sizeof(float));
    ctx->p = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->div = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->smoke = (float*)calloc(ctx->num_cells, sizeof(float));
    ctx->solid = (uint8_t*)calloc(ctx->num_cells, sizeof(uint8_t));

    ctx->u_prev = (float*)calloc((res_x + 1) * res_y, sizeof(float));
    ctx->v_prev = (float*)calloc(res_x * (res_y + 1), sizeof(float));
    ctx->smoke_prev = (float*)calloc(ctx->num_cells, sizeof(float));

    return ctx;
}

// Calculate necessary physics parameters based on the scenario and context settings.
void fluid_setup_physics(FluidContext* ctx, ScenarioParams p) {
    // Calculate reynolds
    if (ctx->visc > 0.0f) { // avoid zero-divison
        ctx->reynolds = (ctx->dens * p.inlet_velocity * p.length_scale ) / ctx->visc;
    } else { // clamp
        ctx->reynolds = 0.0f;
    }

    // Find optimum omega for different scenarios
    if (p.target_omega <= 0.0f) {
        ctx->omega = 2.0f / (1.0f + sinf(M_PI / (float)ctx->x)); 
    } else {
        ctx->omega = p.target_omega;
    }

    if (ctx->omega >= 2.0f) {
        ctx->omega = 1.85f; // Clamp for upperbound
    } else if (ctx->omega < 1.0f) {
        ctx->omega = 1.0f;  // Back to gauss-seidel
    }
}

// Clean up the context and free memory.
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
}

// Executes a single step of the fluid simulation.
void fluid_step(FluidContext* ctx, ScenarioParams p) {

    // Sources
    APPLY_SOURCES();
    APPLY_BOUNDARIES();

    // Velocity Advection
    advect_velocity(ctx, ctx->u_prev, ctx->v_prev, ctx->u, ctx->v);
    
    // Swap pointers
    SWAP_PTR(ctx->u, ctx->u_prev);
    SWAP_PTR(ctx->v, ctx->v_prev);
    
    APPLY_BOUNDARIES();

    // Velocity Diffusion
    memcpy(ctx->u_prev, ctx->u, (ctx->x + 1) * ctx->y * sizeof(float));
    memcpy(ctx->v_prev, ctx->v, ctx->x * (ctx->y + 1) * sizeof(float));
    
    diffuse_velocity(ctx, ctx->u, ctx->v, ctx->u_prev, ctx->v_prev);
    
    APPLY_BOUNDARIES();

    // Projection

    // Divergence
    compute_divergence(ctx, ctx->u, ctx->v);
    
    // Pressure Solve

    solve_pressure_sor(ctx, ctx->p, ctx->div);
    
    // Gradient Subtraction
    subtract_gradient(ctx, ctx->u, ctx->v, ctx->p);
    
    APPLY_BOUNDARIES();

    // Scalar Advection
    advect_scalar(ctx, ctx->smoke_prev, ctx->smoke, ctx->u, ctx->v);
    
    SWAP_PTR(ctx->smoke, ctx->smoke_prev);
    
    APPLY_BOUNDARIES();
}

// TODO:
// header file system & detailed function explanations
// red-black gauss-seidel & parallelism & simd implemenation
// real-time rendering
// different scenarios (airfoil, wind over city)
// consider multigrid for large scale simulations

int main(void) {
    FluidContext* ctx = fluid_create_context(256, 256, 0.016f, 0.1f, 1.0f, 0.001f, 20);
    ScenarioParams p;

    p.inlet_velocity = 1.0f;
    p.obstacle_x = ctx->x / 4;
    p.obstacle_y = ctx->y / 2;
    p.obstacle_radius = 10;
    p.length_scale = 2.0f * p.obstacle_radius * ctx->dx; // karman
    // p.length_scale = (float)ctx->x * ctx->dx; // lid-driven
    p.target_omega = 0.0f; // 0 means auto-calculate

    fluid_setup_physics(ctx, p);

#ifdef VALIDATE
    printf("Reynolds Number: %.2f\n", ctx->reynolds);
    printf("Omega: %.4f\n", ctx->omega);
#endif // VALIDATE

#ifdef DBG
    clock_t start_time = clock();
#endif // DBG

    INIT_SCENARIO();

    int steps_per_frame = 20;
    int num_frames = 400;

    for (int frame = 0; frame < num_frames; frame++) {
        for (int step = 0; step < steps_per_frame; step++) {
            fluid_step(ctx, p);
        }
        // Write To File.
#ifndef DBG
        char filename[22];

        sprintf(filename, "frames/u_%04d.txt", frame);
        mat_save(ctx->u, filename, ctx->x, ctx->y, ctx->y);

        sprintf(filename, "frames/v_%04d.txt", frame);
        mat_save(ctx->v, filename, ctx->x, ctx->y, ctx->y + 1);

        sprintf(filename, "frames/p_%04d.txt", frame);
        mat_save(ctx->p, filename, ctx->x, ctx->y, ctx->y);

        sprintf(filename, "frames/div_%04d.txt", frame);
        mat_save(ctx->div, filename, ctx->x, ctx->y, ctx->y);

        sprintf(filename, "frames/smoke_%04d.txt", frame);
        mat_save(ctx->smoke, filename, ctx->x, ctx->y, ctx->y);
#endif // DBG
    }


#ifdef DBG
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    int total_frame = num_frames * steps_per_frame;
    double fps = (double)total_frame / time_spent;
    printf("Simulated %d frames in %.2f seconds (%.2f FPS)\n", total_frame, time_spent, fps);
#endif // DBG

    fluid_destroy_context(ctx);

    return 0;
}