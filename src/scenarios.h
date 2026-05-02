#ifndef SCENARIOS_H
#define SCENARIOS_H

#include "types.h"

typedef enum {
    LID_DRIVEN,     // Classic lid-driven cavity flow
    KARMAN_VORTEX,  // Flow around a cylinder creating a von Karman vortex street
    AIRFOIL,        // Flow around a NACA 2412 airfoil at zero angle of attack
    URBAN_CITY      // Flow in an urban city environment
} ScenarioType;

// Interface
typedef struct {
    void (*init)(FluidContext* ctx, ScenarioParams p);
    void (*apply_sources)(FluidContext* ctx, ScenarioParams p);
    void (*apply_boundaries)(FluidContext* ctx, ScenarioParams p);
} Scenario;

/**
 * @brief Loads a scenario based on the specified type and initializes the provided parameters.
 * @param type The type of scenario to load (e.g., LID_DRIVEN, KARMAN_VORTEX, AIRFOIL, URBAN_CITY).
 * @param ctx Pointer to the fluid context, which may be used to set scenario-specific parameters
 * @param p Pointer to a ScenarioParams struct that will be populated with scenario-specific parameters such as inlet velocity, length scale, and obstacle geometry.
 * @return A Scenario struct containing function pointers for initialization, source application, and boundary condition application
 */
Scenario load_scenario(ScenarioType type, FluidContext* ctx, ScenarioParams* p);

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
void apply_sources_airfoil(FluidContext* ctx, ScenarioParams p);
void apply_boundaries_airfoil(FluidContext* ctx, ScenarioParams p);

// Urban City

void init_urban_city(FluidContext* ctx, ScenarioParams p);
void apply_sources_urban_city(FluidContext* ctx, ScenarioParams p);
void apply_boundaries_urban_city(FluidContext* ctx, ScenarioParams p);

#endif // SCENARIOS_H