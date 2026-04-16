#include "defines.h"
#include "types.h"
#include "utilities.h"
#include "scenarios.h"
#include "core.h"

// TODO:
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