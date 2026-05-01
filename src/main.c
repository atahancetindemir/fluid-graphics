#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "utilities.h"
#include "scenarios.h"
#include "core.h"

// Further optimizations
// More efficient pressure solvers (e.g. conjugate gradient with multigrid (V-cycle)) for faster convergence.

// TODO:
// real-time rendering & velocity visualizer for urban city and airfoil & mark solids for visualation

int main(void) {
    FluidContext* ctx = fluid_create_context(128, 128, 0.016f, 0.1f, 1.0f, 0.1f, 9999, 0.00001f);
    ScenarioParams p;
    
    // You can ovverride scenario parameters here
    // p.inlet_velocity = 3.0f;
    
    Scenario scenario = load_scenario(lid_driven, ctx, &p);

    fluid_setup_physics(ctx, p, solve_pressure_rbgs);

#ifdef VALIDATE
    printf("Reynolds Number: %.2f\n", ctx->reynolds);
    printf("Omega: %.4f\n", ctx->omega);
#endif // VALIDATE

#ifdef DBG
    double start_time = GET_TIME_SEC();
#endif // DBG

    scenario.init(ctx, p);

    size_t steps_per_frame = 1;
    size_t num_frames = 10;

    for (size_t frame = 0; frame < num_frames; frame++) {
        for (size_t step = 0; step < steps_per_frame; step++) {
            fluid_step(ctx, p, scenario);
        }
        // Write To File.
#ifdef OUTPUT
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
#endif // OUTPUT
    }

#ifdef DBG
    double end_time = GET_TIME_SEC();
    double time_spent = (double)(end_time - start_time);
    int total_frame = num_frames * steps_per_frame;
    double fps = (double)total_frame / time_spent;
    printf("Simulated %d frames in %.2f seconds (%.2f FPS)\n", total_frame, time_spent, fps);
#endif // DBG

    fluid_destroy_context(ctx);

    return 0;
}