#include "scenarios.h"

#include <stddef.h>
#include <math.h>

#include "types.h"
#include "utilities.h"
#include "boundaries.h"

Scenario load_scenario(ScenarioType type, FluidContext* ctx, ScenarioParams* p) {
    Scenario s;
    
    // Default for all scenarios
    p->target_omega = 0.0f;

    switch (type) {
        case LID_DRIVEN:
        p->inlet_velocity = 1.0f;
        p->length_scale = (float)ctx->x * ctx->dx;

        s.init = init_lid_driven;
        s.apply_sources = apply_sources_lid_driven;
        s.apply_boundaries = apply_boundaries_lid_driven;
        break;

        case KARMAN_VORTEX:
            p->inlet_velocity = 1.0f;
            p->obstacle_x = ctx->x / 4.0f;
            p->obstacle_y = ctx->y / 2.0f;
            p->obstacle_radius = 10.0f;
            p->length_scale = 2.0f * p->obstacle_radius * ctx->dx;

            s.init = init_karman_vortex;
            s.apply_sources = apply_sources_karman_vortex;
            s.apply_boundaries = apply_boundaries_karman_vortex;
            break;

        case AIRFOIL:
            p->inlet_velocity = 2.0f;
            p->chord_length = ctx->x * ctx->dx * 0.4f;
            p->angle_of_attack = 0.0f;
            p->length_scale = p->chord_length;

            s.init = init_airfoil;
            s.apply_sources = apply_sources_airfoil;
            s.apply_boundaries = apply_boundaries_airfoil;
            break;

        case URBAN_CITY:
            p->inlet_velocity = 1.5f;
            p->length_scale = (float)ctx->x * 0.08f * ctx->dx;

            s.init = init_urban_city;
            s.apply_sources = apply_sources_urban_city;
            s.apply_boundaries = apply_boundaries_urban_city;
            break;

        default:
            // Fallback to lid-driven if unknown type
            p->length_scale = (float)ctx->x * ctx->dx;

            s.init = init_lid_driven;
            s.apply_sources = apply_sources_lid_driven;
            s.apply_boundaries = apply_boundaries_lid_driven;
    }
    return s;
}

// Lid-Driven Cavity Flow

void init_lid_driven(FluidContext* ctx, ScenarioParams p) {
    UNUSED(p);
    
    // Build fully enclosed solid boundaries
    bound_build_outer_walls(ctx, 1, 1, 1, 1);

    // Clear smoke inside solids
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (ctx->solid[IX(ctx, i, j)]) {
                ctx->smoke[IX(ctx, i, j)] = 0.0f;
            }
        }
    }
}

void apply_sources_lid_driven(FluidContext* ctx, ScenarioParams p) {
    UNUSED(p);

    // Add smoke source at the bottom center
    for (size_t i = 3; i < ctx->x - 3; i++) {
        ctx->smoke[IX(ctx, i, ctx->y - 3)] = 1.0f;
    }
}

void apply_boundaries_lid_driven(FluidContext* ctx, ScenarioParams p) {
    // Apply standard no-slip to all walls
    bound_apply_no_slip(ctx);

    // Unique condition for Lid-Driven: Moving top wall
    for (size_t i = 0; i < ctx->x; i++) {
        if (ctx->solid[IX(ctx, i, ctx->y - 1)]) {
            ctx->u[IX_U(ctx, i, ctx->y - 1)] = p.inlet_velocity;
            ctx->u[IX_U(ctx, i + 1, ctx->y - 1)] = p.inlet_velocity;
        }
    }
}

// Von Karman-Vortex Street

void init_karman_vortex(FluidContext* ctx, ScenarioParams p) {
    // Top and bottom walls only
    bound_build_outer_walls(ctx, 1, 1, 0, 0);

    // Build the cylinder obstacle
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if ((i - p.obstacle_x) * (i - p.obstacle_x) + (j - p.obstacle_y) * (j - p.obstacle_y) <= p.obstacle_radius * p.obstacle_radius) {
                ctx->solid[IX(ctx, i, j)] = 1;
            }
        }
    }

    // Initialize wind field (skip solids)
    for (size_t i = 0; i < ctx->x; i++) {
        for (size_t j = 0; j < ctx->y; j++) {
            if (!(ctx->solid[IX(ctx, i, j)]) && !(ctx->solid[IX(ctx, i - 1, j)])) {
                ctx->u[IX_U(ctx, i, j)] = p.inlet_velocity;
            }
        }
    }
}

void apply_sources_karman_vortex(FluidContext* ctx, ScenarioParams p) {
    UNUSED(p);
    
    // Constant smoke injection behind the inlet
    for (size_t j = 1; j < ctx->y - 1; j++) {
        if (j > (ctx->y / 2 - 2) && j < (ctx->y / 2 + 2)) {
            ctx->smoke[IX(ctx, 2, j)] = 1.0f;
        }
    }
}

void apply_boundaries_karman_vortex(FluidContext* ctx, ScenarioParams p) {
    bound_apply_no_slip(ctx);
    bound_apply_inlet_left(ctx, p.inlet_velocity, 0.0f, 0, 0); // Wind only, smoke is handled in apply_sources
    bound_apply_outlet_right(ctx);
}

// NACA 2412 Airfoil

void init_airfoil(FluidContext* ctx, ScenarioParams p) {
    UNUSED(p);
    
    // Clear domain and reset smoke
    for (size_t i = 0; i < ctx->x * ctx->y; i++) {
        ctx->solid[i] = 0;
        ctx->smoke[i] = 0.0f; 
    }

    // Airfoil dimensions and position
    float chord = ctx->x * ctx->dx * 0.4f; 
    float start_x = ctx->x * ctx->dx * 0.2f; 
    float center_y = ctx->y * ctx->dx * 0.5f;

    // NACA 2412 Parameters
    float m_camber = 0.02f; // Max camber (2%)
    float p_camber = 0.40f; // Max camber position (40%)
    float t_thick  = 0.12f; // Max thickness (12%)

    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            float px = i * ctx->dx - start_x;
            float py = j * ctx->dx - center_y;
            float x_c = px / chord;

            if (x_c >= 0.0f && x_c <= 1.0f) {
                float safe_x_c = (x_c > 0.0f) ? x_c : 0.0f;

                // Thickness distribution
                float y_t = 5.0f * t_thick * chord * (
                    0.2969f * sqrtf(safe_x_c) - 
                    0.1260f * x_c - 
                    0.3516f * x_c * x_c + 
                    0.2843f * x_c * x_c * x_c - 
                    0.1015f * x_c * x_c * x_c * x_c
                );

                // Trailing edge thickness guard to prevent MAC grid pressure singularities
                if (y_t < ctx->dx * 0.8f) {
                    y_t = ctx->dx * 0.8f;
                }

                // Camber line calculation
                float y_c = 0.0f;
                if (safe_x_c < p_camber) {
                    y_c = chord * (m_camber / (p_camber * p_camber)) * (2.0f * p_camber * safe_x_c - safe_x_c * safe_x_c);
                } else {
                    y_c = chord * (m_camber / ((1.0f - p_camber) * (1.0f - p_camber))) * ((1.0f - 2.0f * p_camber) + 2.0f * p_camber * safe_x_c - safe_x_c * safe_x_c);
                }

                // Flag interior solid cells
                if (py <= (y_c + y_t) && py >= (y_c - y_t)) {
                    ctx->solid[IX(ctx, i, j)] = 1;
                }
            }
        }
    }
}

void apply_sources_airfoil(FluidContext* ctx, ScenarioParams p) {
    UNUSED(ctx);
    UNUSED(p);
}

void apply_boundaries_airfoil(FluidContext* ctx, ScenarioParams p) {
    size_t center_j = ctx->y / 2;
    size_t smoke_band_radius = ctx->y / 64;

    bound_apply_inlet_left(ctx, p.inlet_velocity, 1.0f, center_j - smoke_band_radius, center_j + smoke_band_radius);
    bound_apply_outlet_right(ctx);
    bound_apply_slip_horizontal(ctx);
    bound_apply_no_slip(ctx);
}

// Urban City

void init_urban_city(FluidContext* ctx, ScenarioParams p) {
    UNUSED(p);

    // Clear domain and set background ambient air
    for (size_t i = 0; i < ctx->x * ctx->y; i++) {
        ctx->solid[i] = 0;
        ctx->smoke[i] = 0.1f;
    }

    // Zone 1: Frontal low-rise buildings
    BUILD_BLOCK(0.10f, 0.15f, 0.05f, 0.10f);
    BUILD_BLOCK(0.10f, 0.35f, 0.05f, 0.12f);
    BUILD_BLOCK(0.10f, 0.55f, 0.05f, 0.15f);
    BUILD_BLOCK(0.10f, 0.80f, 0.05f, 0.08f);

    // Zone 2: Commercial center (Creates massive wake region)
    BUILD_BLOCK(0.25f, 0.20f, 0.10f, 0.18f); 
    BUILD_BLOCK(0.22f, 0.50f, 0.15f, 0.25f); 

    // Zone 3: Narrow streets (Venturi corridors)
    BUILD_BLOCK(0.42f, 0.45f, 0.04f, 0.15f);
    BUILD_BLOCK(0.42f, 0.65f, 0.04f, 0.15f);
    BUILD_BLOCK(0.48f, 0.50f, 0.03f, 0.20f);

    // Zone 4: High-density residential (Staggered arrangement)
    BUILD_BLOCK(0.60f, 0.15f, 0.04f, 0.06f);
    BUILD_BLOCK(0.60f, 0.25f, 0.04f, 0.06f);
    BUILD_BLOCK(0.60f, 0.35f, 0.04f, 0.06f);
    
    BUILD_BLOCK(0.68f, 0.20f, 0.04f, 0.06f); 
    BUILD_BLOCK(0.68f, 0.30f, 0.04f, 0.06f);
    BUILD_BLOCK(0.68f, 0.40f, 0.04f, 0.06f);

    BUILD_BLOCK(0.60f, 0.70f, 0.06f, 0.08f);
    BUILD_BLOCK(0.60f, 0.82f, 0.06f, 0.08f);

    // Zone 5: Industrial facilities
    BUILD_BLOCK(0.80f, 0.25f, 0.08f, 0.04f); 
    BUILD_BLOCK(0.80f, 0.45f, 0.08f, 0.04f);
    BUILD_BLOCK(0.80f, 0.65f, 0.08f, 0.04f);

}

void apply_sources_urban_city(FluidContext* ctx, ScenarioParams p) {
    UNUSED(ctx);
    UNUSED(p);
}

void apply_boundaries_urban_city(FluidContext* ctx, ScenarioParams p) {
    // Full wall inlet with continuous smoke injection
    bound_apply_inlet_left(ctx, p.inlet_velocity, 1.0f, 1, ctx->y - 1);
    bound_apply_outlet_right(ctx);
    bound_apply_slip_horizontal(ctx);
    bound_apply_no_slip(ctx);
}