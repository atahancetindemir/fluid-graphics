#include "setup.h"

// Every scenario is dimensionless: U = 1 and a reference length of 1, so viscosity = 1 / Re.
static const ScenarioSetup setups[] = {
    // Ghia, Ghia & Shin (1982) cavity: L = 1 on 128 fluid cells, Re = 1000
    [LID_DRIVEN] = {
        .nx = 130, .ny = 130, .dx = 1.0f / 128.0f, .dt = 0.004f,
        .density = 1.0f, .viscosity = 0.001f,
        .tol = { 1e-4f, 1e-9f, 1000 },
        .params = { .inlet_velocity = 1.0f },
        .velocity_max = 1.0f, .pressure_min = -0.1f, .pressure_max = 0.2f,
    },
    // Cylinder D = 1 on 20 cells, Re = 100
    [KARMAN_VORTEX] = {
        .nx = 512, .ny = 256, .dx = 0.05f, .dt = 0.025f,
        .density = 1.0f, .viscosity = 0.01f,
        .tol = { 1e-4f, 1e-9f, 1000 },
        .params = { .inlet_velocity = 1.0f, .obstacle_x = 128.0f, .obstacle_y = 128.0f, .obstacle_radius = 10.0f },
        .velocity_max = 1.4f, .pressure_min = -0.3f, .pressure_max = 0.35f,
    },
    // NACA 2412, chord c = 1 on 64 cells, leading edge 2c from the inlet, Re = 1000
    [AIRFOIL] = {
        .nx = 512, .ny = 256, .dx = 1.0f / 64.0f, .dt = 0.0078f,
        .density = 1.0f, .viscosity = 0.001f,
        .tol = { 1e-4f, 1e-9f, 1000 },
        .params = { .inlet_velocity = 1.0f, .obstacle_x = 128.0f, .obstacle_y = 128.0f,
                    .chord_length = 64.0f, .angle_of_attack = 4.0f },
        .velocity_max = 1.2f, .pressure_min = -0.15f, .pressure_max = 0.1f,
    },
    // Block width b ~ 1 on ~20 cells, Re_b ~ 1000
    [URBAN_CITY] = {
        .nx = 256, .ny = 256, .dx = 0.05f, .dt = 0.02f,
        .density = 1.0f, .viscosity = 0.001f,
        .tol = { 1e-4f, 1e-9f, 1000 },
        .params = { .inlet_velocity = 1.0f },
        .velocity_max = 4.3f, .pressure_min = -3.2f, .pressure_max = 6.1f,
    },
};

const ScenarioSetup* scenario_setup(ScenarioType type) {
    return &setups[type];
}
