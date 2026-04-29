#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

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
static inline void mat_cpy(float* dest, float* src, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dest[i * cols + j] = src[i * cols + j];
        }
    }
}

// Zero out the matrix
static inline void mat_zero_float(float* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = 0.0f;
        }
    }
}

// Zero out the matrix
static inline void mat_zero_uint(uint8_t* mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mat[i * cols + j] = 0;
        }
    }
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