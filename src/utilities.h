#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#ifndef INLINE
    #if defined(__GNUC__) || defined(__clang__)
        // GCC & Clang
        #define INLINE static inline __attribute__((always_inline))
    #elif defined(_MSC_VER)
        // MSVC
        #define INLINE __forceinline
    #else
        // Fallback to C99
        #define INLINE static inline
    #endif
#endif

#ifdef _OPENMP
    #include <omp.h>
    #define GET_TIME_SEC() (omp_get_wtime())
#else
    #include <time.h>
    #define GET_TIME_SEC() ((double)clock() / CLOCKS_PER_SEC)
#endif // _OPENMP

#define PI 3.14159265358979323846f

#define IX(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_U(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_V(ctx, i, j) ((size_t)(i) * ((ctx)->y + 1) + (size_t)(j))

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define SWAP_PTR(x0, x) do { float* tmp = x0; x0 = x; x = tmp; } while(0)

#define BUILD_BLOCK(start_x_pct, start_y_pct, w_pct, h_pct) \
    do { \
        size_t sx = (size_t)(ctx->x * (start_x_pct)); \
        size_t sy = (size_t)(ctx->y * (start_y_pct)); \
        size_t bw = (size_t)(ctx->x * (w_pct)); \
        size_t bh = (size_t)(ctx->y * (h_pct)); \
        for (size_t i = sx; i < sx + bw; i++) { \
            for (size_t j = sy; j < sy + bh; j++) { \
                if (i < ctx->x && j < ctx->y) { \
                    ctx->solid[IX(ctx, i, j)] = 1; \
                } \
            } \
        } \
    } while(0)

// Fetch neighbors with Neumann BC handling for cell-centered scalar fields.
static inline float apply_scalar_laplacian(const FluidContext* ctx, const float* src, size_t i, size_t j) {
    float center = src[IX(ctx, i, j)];
    float left = ctx->solid[IX(ctx, i - 1, j)] ? center : src[IX(ctx, i - 1, j)];
    float right = ctx->solid[IX(ctx, i + 1, j)] ? center : src[IX(ctx, i + 1, j)];
    float bottom = ctx->solid[IX(ctx, i, j - 1)] ? center : src[IX(ctx, i, j - 1)];
    float top = ctx->solid[IX(ctx, i, j + 1)] ? center : src[IX(ctx, i, j + 1)];
    return left + right + bottom + top - 4.0f * center;
}

// 5-point Laplacian for u-field (no Neumann BC, Dirichlet at solids).
static inline float apply_u_laplacian(const FluidContext* ctx, const float* u, size_t i, size_t j) {
    float center = u[IX_U(ctx, i, j)];
    float left = u[IX_U(ctx, i - 1, j)];
    float right = u[IX_U(ctx, i + 1, j)];
    float bottom = u[IX_U(ctx, i, j - 1)];
    float top = u[IX_U(ctx, i, j + 1)];
    return left + right + bottom + top - 4.0f * center;
}

// 5-point Laplacian for v-field (no Neumann BC, Dirichlet at solids).
static inline float apply_v_laplacian(const FluidContext* ctx, const float* v, size_t i, size_t j) {
    float center = v[IX_V(ctx, i, j)];
    float left = v[IX_V(ctx, i - 1, j)];
    float right = v[IX_V(ctx, i + 1, j)];
    float bottom = v[IX_V(ctx, i, j - 1)];
    float top = v[IX_V(ctx, i, j + 1)];
    return left + right + bottom + top - 4.0f * center;
}

// One explicit diffusion step for a scalar field: u_new = u + a * Laplacian(u).
static inline float compute_explicit_scalar_update(const FluidContext* ctx, const float* src, float a, size_t i, size_t j) {
    return src[IX(ctx, i, j)] + a * apply_scalar_laplacian(ctx, src, i, j);
}

// One explicit diffusion step for u-field: u_new = u + a * Laplacian(u).
static inline float compute_explicit_u_update(const FluidContext* ctx, const float* u_src, float a, size_t i, size_t j) {
    return u_src[IX_U(ctx, i, j)] + a * apply_u_laplacian(ctx, u_src, i, j);
}

// One explicit diffusion step for v-field: v_new = v + a * Laplacian(v).
static inline float compute_explicit_v_update(const FluidContext* ctx, const float* v_src, float a, size_t i, size_t j) {
    return v_src[IX_V(ctx, i, j)] + a * apply_v_laplacian(ctx, v_src, i, j);
}

// One implicit Gauss-Seidel diffusion step for a scalar field.
static inline float compute_implicit_scalar_update(const FluidContext* ctx, const float* dest, const float* src, float a, size_t i, size_t j) {
    float neighbors = apply_scalar_laplacian(ctx, dest, i, j) + 4.0f * dest[IX(ctx, i, j)];
    return (src[IX(ctx, i, j)] + a * neighbors) / (1.0f + 4.0f * a);
}

// One implicit Gauss-Seidel diffusion step for u-field.
static inline float compute_implicit_u_update(const FluidContext* ctx, const float* u_dest, const float* u_src, float a, size_t i, size_t j) {
    float neighbors = apply_u_laplacian(ctx, u_dest, i, j) + 4.0f * u_dest[IX_U(ctx, i, j)];
    return (u_src[IX_U(ctx, i, j)] + a * neighbors) / (1.0f + 4.0f * a);
}

// One implicit Gauss-Seidel diffusion step for v-field.
static inline float compute_implicit_v_update(const FluidContext* ctx, const float* v_dest, const float* v_src, float a, size_t i, size_t j) {
    float neighbors = apply_v_laplacian(ctx, v_dest, i, j) + 4.0f * v_dest[IX_V(ctx, i, j)];
    return (v_src[IX_V(ctx, i, j)] + a * neighbors) / (1.0f + 4.0f * a);
}

// Dot product of two vectors.
static inline float dot(const float* a, const float* b, size_t n) {
    double acc = 0.0;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (size_t k = 0; k < n; k++) acc += (double)(a[k] * b[k]);
    return (float)acc;
}

// Gauss-Seidel Poisson update: returns new pressure from neighbors and divergence rhs.
static inline float compute_poisson_update(const FluidContext* ctx, const float* p, const float* div, float cp, size_t i, size_t j) {
    float center = p[IX(ctx, i, j)];
    float left   = ctx->solid[IX(ctx, i - 1, j)] ? center : p[IX(ctx, i - 1, j)];
    float right  = ctx->solid[IX(ctx, i + 1, j)] ? center : p[IX(ctx, i + 1, j)];
    float bottom = ctx->solid[IX(ctx, i, j - 1)] ? center : p[IX(ctx, i, j - 1)];
    float top    = ctx->solid[IX(ctx, i, j + 1)] ? center : p[IX(ctx, i, j + 1)];
    return (left + right + bottom + top - div[IX(ctx, i, j)] * cp) * 0.25f;
}

// Pointwise residual of the Poisson equation: r = (Lap(p) - div*cp) at cell (i,j).
static inline float compute_cell_residual(const FluidContext* ctx, const float* p, const float* div, float cp, size_t i, size_t j) {
    float center = p[IX(ctx, i, j)];
    float left   = ctx->solid[IX(ctx, i - 1, j)] ? center : p[IX(ctx, i - 1, j)];
    float right  = ctx->solid[IX(ctx, i + 1, j)] ? center : p[IX(ctx, i + 1, j)];
    float bottom = ctx->solid[IX(ctx, i, j - 1)] ? center : p[IX(ctx, i, j - 1)];
    float top    = ctx->solid[IX(ctx, i, j + 1)] ? center : p[IX(ctx, i, j + 1)];
    return (left + right + bottom + top - 4.0f * center) - div[IX(ctx, i, j)] * cp;
}

static inline float bilinear_interp(float sx0, float sx1, float sy0, float sy1, float f00, float f01, float f10, float f11) {
    return sx0 * (sy0 * f00 + sy1 * f01) + sx1 * (sy0 * f10 + sy1 * f11);
}

// Initializes rng with a seed
static inline void rng_init_seed(uint32_t seed) {
    srand(seed);
}

// Initializes rng with current time
static inline void rng_init_time(void) {
    srand((uint32_t)time(NULL));
}

// Generates unbiased random float in [0, 1)
static inline float rng_urand01(void) {
    float inv_rand_max_plus_one = 1.0f / ((float)RAND_MAX + 1.0f);
    return (float)rand() * inv_rand_max_plus_one;
}

// Generates unbiased random float in [0, 1]
static inline float rng_urand01_closed(void) {
    float inv_rand_max = 1.0f / (float)RAND_MAX;
    return (float)rand() * inv_rand_max;
}

// Generates unbiased random float in [min, max]
static inline float rng_urand_range(float min, float max) {
    float u = rng_urand01_closed();
    return min + u * (max - min);
}

// Display matrix
static inline void mat_display(float* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%.3f\t", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Copy src to dest
static inline void mat_cpy(float* dest, const float* src, size_t rows, size_t cols) {
    memcpy(dest, src, rows * cols * sizeof(float));
}

// Zero out the matrix
static inline void mat_zero_float(float* mat, size_t rows, size_t cols) {
    memset(mat, 0, rows * cols * sizeof(float));
}

// Zero out the matrix
static inline void mat_zero_uint(uint8_t* mat, size_t rows, size_t cols) {
    memset(mat, 0, rows * cols);
}

// Fill the matrix with specific value
static inline void mat_value(float* mat, size_t rows, size_t cols, float val) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = val;
        }
    }
}

// Fill the matrix with random values in given range.
static inline void mat_random(float* mat, size_t rows, size_t cols, float min, float max) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = rng_urand_range(min, max);
        }
    }
}

// Save matrix to a file
static inline void mat_save(float* mat, const char* filename, int rows, int cols, int stride) {
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

#endif // UTILITIES_H