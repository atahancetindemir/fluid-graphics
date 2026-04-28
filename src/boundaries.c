#include "boundaries.h"

#include <stddef.h>

#include "types.h"
#include "utilities.h"

void bound_apply_no_slip(FluidContext* ctx) {
    // Zero out velocities around all solid cells (No-Slip condition)
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
}

void bound_apply_slip_horizontal(FluidContext* ctx) {
    // Slip condition for wind tunnel walls (Top and Bottom)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        ctx->v[IX_V(ctx, i, 1)] = 0.0f;
        ctx->v[IX_V(ctx, i, ctx->y - 1)] = 0.0f;
    }
}

void bound_apply_inlet_left(FluidContext* ctx, float velocity, float smoke_density, size_t smoke_start_j, size_t smoke_end_j) {
    for (size_t j = 1; j < ctx->y - 1; j++) {
        // Apply wind velocity
        ctx->u[IX_U(ctx, 1, j)] = velocity;
        ctx->v[IX_V(ctx, 0, j)] = 0.0f;
        
        // Inject smoke only within the specified band
        if (j >= smoke_start_j && j <= smoke_end_j) {
            ctx->smoke[IX(ctx, 1, j)] = smoke_density; 
        } else {
            ctx->smoke[IX(ctx, 1, j)] = 0.0f; 
        }
    }
}

void bound_apply_outlet_right(FluidContext* ctx) {
    // Zero-gradient for seamless outflow
    for (size_t j = 1; j < ctx->y - 1; j++) {
        ctx->u[IX_U(ctx, ctx->x - 1, j)] = ctx->u[IX_U(ctx, ctx->x - 2, j)];
        ctx->smoke[IX(ctx, ctx->x - 1, j)] = ctx->smoke[IX(ctx, ctx->x - 2, j)];
    }
}

void bound_build_outer_walls(FluidContext* ctx, int top, int bottom, int left, int right) {
    if (top) {
        for (size_t i = 0; i < ctx->x; i++) ctx->solid[IX(ctx, i, ctx->y - 1)] = 1;
    }
    if (bottom) {
        for (size_t i = 0; i < ctx->x; i++) ctx->solid[IX(ctx, i, 0)] = 1;
    }
    if (left) {
        for (size_t j = 0; j < ctx->y; j++) ctx->solid[IX(ctx, 0, j)] = 1;
    }
    if (right) {
        for (size_t j = 0; j < ctx->y; j++) ctx->solid[IX(ctx, ctx->x - 1, j)] = 1;
    }
}