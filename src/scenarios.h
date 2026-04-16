#ifndef SCENARIOS_H
#define SCENARIOS_H

#include "types.h"

// Lid-Driven Cavity Flow

void init_lid_driven(FluidContext* ctx);
void apply_sources_lid_driven(FluidContext* ctx);
void apply_boundaries_lid_driven(FluidContext* ctx, ScenarioParams p);

// Karman-Vortex Street

void init_karman_vortex(FluidContext* ctx, ScenarioParams p);
void apply_sources_karman_vortex(FluidContext* ctx);
void apply_boundaries_karman_vortex(FluidContext* ctx, ScenarioParams p);

// Airfoil

void init_airfoil(void);
void apply_sources_airfoil(void);
void apply_boundaries_airfoil(void);

// Wind Over City

void init_wind_over_city(void);
void apply_sources_wind_over_city(void);
void apply_boundaries_wind_over_city(void);

#endif // SCENARIOS_H