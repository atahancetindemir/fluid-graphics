#ifndef SCENARIOS_H
#define SCENARIOS_H

#include "types.h"

// Lid-Driven Cavity Flow

void init_lid_driven(FluidContext* ctx);
void apply_sources_lid_driven(FluidContext* ctx);
void apply_boundaries_lid_driven(FluidContext* ctx, ScenarioParams p);

// Von Karman-Vortex Street

void init_karman_vortex(FluidContext* ctx, ScenarioParams p);
void apply_sources_karman_vortex(FluidContext* ctx);
void apply_boundaries_karman_vortex(FluidContext* ctx, ScenarioParams p);

// NACA 2412 Airfoil

void init_airfoil(FluidContext* ctx, ScenarioParams p);
void apply_sources_airfoil(FluidContext* ctx, ScenarioParams p);
void apply_boundaries_airfoil(FluidContext* ctx, ScenarioParams p);

// Urban City

void init_urban_city(FluidContext* ctx, ScenarioParams p);
void apply_sources_urban_city(FluidContext* ctx, ScenarioParams p);
void apply_boundaries_urban_city(FluidContext* ctx, ScenarioParams p);

#endif // SCENARIOS_H