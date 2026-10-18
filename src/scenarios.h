#ifndef SCENARIOS_H
#define SCENARIOS_H

#include "types.h"

typedef enum {
    LID_DRIVEN,     // Classic lid-driven cavity flow
    KARMAN_VORTEX,  // Flow around a cylinder creating a von Karman vortex street
    AIRFOIL,        // Flow around a NACA 2412 airfoil at an angle of attack
    URBAN_CITY      // Flow in an urban city environment
} ScenarioType;

// Interface
typedef struct {
    void (*init)(FluidContext* ctx, ScenarioParams p);
    void (*apply_sources)(FluidContext* ctx, ScenarioParams p);
    void (*apply_boundaries)(FluidContext* ctx, ScenarioParams p);
} Scenario;

// Returns the init, source and boundary functions of a scenario; its default values live in setup.c.
Scenario load_scenario(ScenarioType type);

// Recomputes the parameters that are derived from other scenario parameters, currently the
// characteristic length scale. Call this after editing a parameter the length scale depends on
void scenario_update_derived(ScenarioType type, const FluidContext* ctx, ScenarioParams* p);

// Lid-Driven Cavity Flow

void init_lid_driven(FluidContext* ctx, ScenarioParams p);

void apply_sources_lid_driven(FluidContext* ctx, ScenarioParams p);

void apply_boundaries_lid_driven(FluidContext* ctx, ScenarioParams p);

// Von Karman-Vortex Street

void init_karman_vortex(FluidContext* ctx, ScenarioParams p);

void apply_sources_karman_vortex(FluidContext* ctx, ScenarioParams p);

void apply_boundaries_karman_vortex(FluidContext* ctx, ScenarioParams p);

// NACA 2412 Airfoil

void init_airfoil(FluidContext* ctx, ScenarioParams p);

// Unused
void apply_sources_airfoil(FluidContext* ctx, ScenarioParams p);

void apply_boundaries_airfoil(FluidContext* ctx, ScenarioParams p);

// Urban City

void init_urban_city(FluidContext* ctx, ScenarioParams p);

void apply_sources_urban_city(FluidContext* ctx, ScenarioParams p);

void apply_boundaries_urban_city(FluidContext* ctx, ScenarioParams p);

#endif // SCENARIOS_H