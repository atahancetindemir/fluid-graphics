#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

// Layout
#define X 129
#define Y 129

// Parameters
#define DT 0.016f
#define DX 0.1f
#define DENSITY 1.0f
#define ITER 20
#define SEED 54

// Grid Cell Type
unsigned char solid[X][Y];

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

// ############### Utils ###############

// Initializes rng with a seed
void rng_init_seed(unsigned int seed) {
    srand(seed);
}

// Initializes rng with current time
void rng_init_time(void) {
    srand((unsigned int)time(NULL));
}

// Generates unbiased random float in [0, 1)
float rng_urand01(void) {
    float inv_rand_max_plus_one = 1.0 / ((float)RAND_MAX + 1.0);
    return (float)rand() * inv_rand_max_plus_one;
}

// Generates unbiased random float in [0, 1]
float rng_urand01_closed(void) {
    float inv_rand_max = 1.0 / (float)RAND_MAX;
    return (float)rand() * inv_rand_max;
}

// Generates unbiased random float in [min, max]
float rng_urand_range(float min, float max) {
    float u = rng_urand01_closed();
    return min * (1.0 - u) + max * u;
}

// Display matrix
void mat_display(unsigned int row, unsigned int column, float mat[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            printf("%.3f\t", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Copy src to dest
void mat_cpy(unsigned int row, unsigned int column, float src[row][column], float dest[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

// Zero out the matrix
void mat_zero(unsigned int row, unsigned int column, float mat[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] = 0.0f;
        }
    }
}

// Zero out the matrix
void mat_zero_char(unsigned int row, unsigned int column, unsigned char mat[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] = 0;
        }
    }
}

// Fill the matrix with specific value
void mat_value(unsigned int row, unsigned int column, float value, float mat[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] = value;
        }
    }
}

// Fill the matrix with random values in given range.
void mat_random(unsigned int row, unsigned int column, float min, float max, float mat[row][column]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] = rng_urand_range(min, max);
        }
    }
}

// Save matrix to a file
void save_matrix(int row, int column, float mat[row][column], const char* filename) {
    FILE* f = fopen(filename, "w");
    if (f == NULL)
        return;

    // We save the matrix in column-major order to match the way we visualize it later.
    for (int j = column - 1; j >= 0; j--) {
        for (int i = 0; i < row; i++) {
            fprintf(f, "%f ", mat[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// ############### Utils ###############

void boundaries_lid_driven(float u[X + 1][Y], float v[X][Y + 1]) {
    float lid_velocity = 1.0f;

    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            if (solid[i][j]) {
                u[i][j] = 0.0f;
                u[i + 1][j] = 0.0f;
                v[i][j] = 0.0f;
                v[i][j + 1] = 0.0f;
            }
        }
    }
    // Temporary no-slip hack
    for (int i = 2; i < X - 1; i++) {
        u[i][Y - 2] = lid_velocity;
    }
}

// Computes the divergence of the velocity field.
void compute_divergence(float u[X + 1][Y], float v[X][Y + 1], float div[X][Y], float p[X][Y], float dx) {
    float half_inv_dx = 1.0f / dx;

    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y - 1; j++) {
            if (solid[i][j]) {
                div[i][j] = 0.0f;
                continue;
            }
            // Divergence = Right - Left + Top - Bottom
            float divergence = (u[i + 1][j] - u[i][j] + v[i][j + 1] - v[i][j]) * half_inv_dx;
            div[i][j] = divergence;
            p[i][j] = 0.0f;
        }
    }
}

// Solves the pressure Poisson equation using Jacobi iteration.
void solve_pressure(float p[X][Y], float div[X][Y], float dx, float dt, float density, int iter_count) {
    float cp = (density * dx * dx) / dt;

    for (int iter = 0; iter < iter_count; iter++) {
        for (int i = 1; i < X - 1; i++) {
            for (int j = 1; j < Y - 1; j++) {
                // P = (Left + Right + Bottom + Top - Divergence * cp) / 4
                float p_left = solid[i - 1][j] ? p[i][j] : p[i - 1][j];
                float p_right = solid[i + 1][j] ? p[i][j] : p[i + 1][j];
                float p_bottom = solid[i][j - 1] ? p[i][j] : p[i][j - 1];
                float p_top = solid[i][j + 1] ? p[i][j] : p[i][j + 1];
                p[i][j] = (p_left + p_right + p_bottom + p_top - div[i][j] * cp) * 0.25f;
            }
        }
    }
}

// Subtracts the pressure gradient from the velocity field to enforce incompressibility.
void subtract_gradient(float u[X + 1][Y], float v[X][Y + 1], float p[X][Y], float dx, float dt, float density) {
    float scale = dt / (density * dx);

    // Horizontal velocity correction
    // i=1 (Left) to i=X-1 (Right)
    for (int i = 1; i < X; i++) {
        for (int j = 1; j < Y - 1; j++) {
            if (solid[i - 1][j] || solid[i][j])
                continue;
            u[i][j] -= scale * (p[i][j] - p[i - 1][j]);
        }
    }

    // Vertical velocity correction
    // j=1 (Bottom) to j=Y-1 (Top)
    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y; j++) {
            if (solid[i - 1][j] || solid[i][j])
                continue;
            v[i][j] -= scale * (p[i][j] - p[i][j - 1]);
        }
    }
}

// Advects the smoke density field using semi-Lagrangian advection.
void advect_smoke(float smoke[X][Y], float u[X + 1][Y], float v[X][Y + 1], float dt, float dx) {

    static float temp_smoke[X][Y] = {0};

    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y - 1; j++) {

            // Calculate the average velocity (i, j) at the center of the cell.
            // Since the u (horizontal) and v (vertical) velocities are at the edges, their average will be at the center.
            float u_avg = (u[i][j] + u[i + 1][j]) * 0.5f;
            float v_avg = (v[i][j] + v[i][j + 1]) * 0.5f;

            // Backtracing
            // We use indices like coordinates.
            float src_x = (float)i - dt * u_avg / dx;
            float src_y = (float)j - dt * v_avg / dx;

            // Clamp to boundary
            if (src_x < 0.5f)
                src_x = 0.5f;
            if (src_x > X - 1.5f)
                src_x = X - 1.5f;
            if (src_y < 0.5f)
                src_y = 0.5f;
            if (src_y > Y - 1.5f)
                src_y = Y - 1.5f;

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
            temp_smoke[i][j] = sx0 * (sy0 * smoke[i0][j0] + sy1 * smoke[i0][j1]) + sx1 * (sy0 * smoke[i1][j0] + sy1 * smoke[i1][j1]);
        }
    }

    // Transfer new velocities
    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y - 1; j++) {
            smoke[i][j] = temp_smoke[i][j];
        }
    }
}

// Advects the velocity field using semi-Lagrangian advection.
void advect_velocity(float u[X + 1][Y], float v[X][Y + 1], float dt, float dx) {

    // temp velocity vectors
    static float temp_u[X + 1][Y] = {0};
    static float temp_v[X][Y + 1] = {0};

    // u (horizontal) velocity update
    // u vectors are located on the vertical edges. Position: (i, j + 0.5)
    for (int i = 1; i < X; i++) {
        for (int j = 1; j < Y - 1; j++) {

            float u_vel = u[i][j];
            // To find the velocity v at point u, take the average of the 4 v's around it.
            float v_vel = (v[i - 1][j] + v[i][j] + v[i - 1][j + 1] + v[i][j + 1]) * 0.25f;

            // Backtracking
            float src_x = (float)i - dt * u_vel / dx;
            float src_y = ((float)j + 0.5f) - dt * v_vel / dx;

            // Since the u-grid is shifted by 0.5 on the Y-axis, we subtract this from the index calculation.
            float idx_x = src_x;
            float idx_y = src_y - 0.5f;

            // Clamp for u
            if (idx_x < 0.5f)
                idx_x = 0.5f;
            if (idx_x > X - 0.5f)
                idx_x = X - 0.5f;
            if (idx_y < 0.5f)
                idx_y = 0.5f;
            if (idx_y > Y - 1.5f)
                idx_y = Y - 1.5f;

            int i0 = (int)idx_x;
            int i1 = i0 + 1;
            int j0 = (int)idx_y;
            int j1 = j0 + 1;

            float sx1 = idx_x - (float)i0;
            float sx0 = 1.0f - sx1;
            float sy1 = idx_y - (float)j0;
            float sy0 = 1.0f - sy1;

            // Calculate the new value of u using bilinear interpolation.
            temp_u[i][j] = sx0 * (sy0 * u[i0][j0] + sy1 * u[i0][j1]) + sx1 * (sy0 * u[i1][j0] + sy1 * u[i1][j1]);
        }
    }

    // v (vertical) velocity update
    // v vectors are located on the horizontal edges. Position: (i + 0.5, j)
    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y; j++) {

            // To find the velocity u at point v, take the average of the 4 u's around it.
            float u_vel = (u[i][j - 1] + u[i + 1][j - 1] + u[i][j] + u[i + 1][j]) * 0.25f;
            float v_vel = v[i][j];

            // Backtracking
            float src_x = ((float)i + 0.5f) - dt * u_vel / dx;
            float src_y = (float)j - dt * v_vel / dx;

            // Since the v-grid is shifted by 0.5 on the X-axis, we subtract this from the index calculation.
            float idx_x = src_x - 0.5f;
            float idx_y = src_y;

            // Clamp for v
            if (idx_x < 0.5f)
                idx_x = 0.5f;
            if (idx_x > X - 1.5f)
                idx_x = X - 1.5f;
            if (idx_y < 0.5f)
                idx_y = 0.5f;
            if (idx_y > Y - 0.5f)
                idx_y = Y - 0.5f;

            int i0 = (int)idx_x;
            int i1 = i0 + 1;
            int j0 = (int)idx_y;
            int j1 = j0 + 1;

            float sx1 = idx_x - (float)i0;
            float sx0 = 1.0f - sx1;
            float sy1 = idx_y - (float)j0;
            float sy0 = 1.0f - sy1;

            // Calculate the new value of v using bilinear interpolation.
            temp_v[i][j] = sx0 * (sy0 * v[i0][j0] + sy1 * v[i0][j1]) + sx1 * (sy0 * v[i1][j0] + sy1 * v[i1][j1]);
        }
    }

    // Transfer new velocities
    for (int i = 1; i < X; i++) {
        for (int j = 1; j < Y - 1; j++) {
            u[i][j] = temp_u[i][j];
        }
    }
    for (int i = 1; i < X - 1; i++) {
        for (int j = 1; j < Y; j++) {
            v[i][j] = temp_v[i][j];
        }
    }
}

/*
Further Improvements:
* Engine:
    - Add diffuse_velocity
* Scenarios:
    - Lid-driven cavity flow
    - Water tank + external force
    - Karman vortex street
    - City Planning (?)
* Structure:
    - Get rid of VLA approach
    - Add Heap based matrix system
* Optimization:
    - Algorithmic speedup
    - Row-major & cache-friendly operations
    - Vector operations
    - CPU-Based Parallelism
*/

int main() {
    float u[X + 1][Y];
    float v[X][Y + 1];
    float p[X][Y];
    float div[X][Y];
    float smoke[X][Y];

    int cx = X / 2;
    int cy = Y / 2;

    rng_init_seed(SEED);

    mat_zero(X + 1, Y, u);
    mat_zero(X, Y + 1, v);
    mat_zero(X, Y, p);
    mat_zero(X, Y, smoke);

    mat_zero_char(X, Y, solid); // Set all grids to fluid.

    // Determine solid boundaries.
    for (int i = 0; i < X; i++) {
        solid[i][0] = 1;     // Top
        solid[i][Y - 1] = 1; // Bottom
    }
    for (int j = 0; j < Y; j++) {
        solid[0][j] = 1;     // Left
        solid[X - 1][j] = 1; // Right
    }

    mat_value(X, Y, 1.0f, smoke);

    for (int i = 0; i < X; i++) {
        for (int j = 0; j < Y; j++) {
            if (solid[i][j]) {
                smoke[i][j] = 0.0f;
            }
        }
    }

    // for(int i = X/4; i < 3*X/4; i++) {
    //     smoke[i][Y-3] = 1.0f;
    //     smoke[i][Y-4] = 1.0f;
    // }

    // // Simple wind tunnel
    // for(int i=1; i<X; i++) {
    //     for(int j=1; j<Y-1; j++) {
    //         if (!solid[i][j] && !solid[i-1][j]) {
    //             u[i][j] = 1.0f;
    //         }
    //     }
    // }

    int steps_per_frame = 30;
    int num_frames = 800;
    for (int frame = 0; frame < num_frames; frame++) {

        for (int step = 0; step < steps_per_frame; step++) {
            boundaries_lid_driven(u, v);

            // for(int i=1; i<X-1; i++) {
            //     for(int j=1; j<Y-1; j++) {
            //         if (smoke[i][j] > 0.5f) {
            //             u[i][j] += 0.01f;
            //         }
            //     }
            // }

            // for(int i=1; i<X-1; i++) {
            //     for(int j=1; j<Y-1; j++) {
            //         if (smoke[i][j] > 0.1f) {
            //             v[i][j] -= 0.01f;
            //         }
            //     }
            // }

            // Advection
            advect_velocity(u, v, DT, DX);
            advect_smoke(smoke, u, v, DT, DX);

            boundaries_lid_driven(u, v);

            // Divergence
            compute_divergence(u, v, div, p, DX);

            // Pressure
            solve_pressure(p, div, DX, DT, DENSITY, ITER);

            // Projection
            subtract_gradient(u, v, p, DX, DT, DENSITY);

            boundaries_lid_driven(u, v);
        }
        // Write To File.

        char filename[32];

        sprintf(filename, "frames/u_%04d.txt", frame);
        save_matrix(X + 1, Y, u, filename);

        sprintf(filename, "frames/v_%04d.txt", frame);
        save_matrix(X, Y + 1, v, filename);

        sprintf(filename, "frames/p_%04d.txt", frame);
        save_matrix(X, Y, p, filename);

        sprintf(filename, "frames/div_%04d.txt", frame);
        save_matrix(X, Y, div, filename);

        sprintf(filename, "frames/smoke_%04d.txt", frame);
        save_matrix(X, Y, smoke, filename);
    }

    return 0;
}
