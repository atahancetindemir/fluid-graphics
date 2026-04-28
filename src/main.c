#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "defines.h"
#include "types.h"
#include "utilities.h"
#include "scenarios.h"
#include "core.h"

// TODO:
// red-black gauss-seidel & parallelism & simd implemenation
// real-time rendering & velocity visualizer for urban city and airfoil
// scenario dispatcher with dynamic parameters

int main(void) {
    FluidContext* ctx = fluid_create_context(256, 256, 0.016f, 0.1f, 1.0f, 0.001f, 20);
    ScenarioParams p;

    p.inlet_velocity = 1.0f;
    p.obstacle_x = ctx->x / 4;
    p.obstacle_y = ctx->y / 2;
    p.obstacle_radius = 10;

    // Karman Vortex Street: Length Scale = Cylinder Diameter
    // p.length_scale = 2.0f * p.obstacle_radius * ctx->dx;

    // Lid-Driven Cavity: Length Scale = Domain Size
    // p.length_scale = (float)ctx->x * ctx->dx;

    // Airfoil: Length Scale = Chord Length
    // p.chord_length = 20.0f;
    // p.angle_of_attack = 10.0f;
    // p.length_scale = p.chord_length;

    // Urban City: Length Scale = Characteristic Building Size
    p.length_scale = (float)ctx->x * 0.08f * ctx->dx;

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

    size_t steps_per_frame = 20;
    size_t num_frames = 200;

    for (size_t frame = 0; frame < num_frames; frame++) {
        for (size_t step = 0; step < steps_per_frame; step++) {
            fluid_step(ctx, p);
        }
        // Write To File.
#ifdef DBG
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