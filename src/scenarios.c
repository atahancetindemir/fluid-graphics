#include "types.h"
#include "defines.h"

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