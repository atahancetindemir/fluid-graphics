#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef struct FluidContext FluidContext;
typedef void (*PressureSolver)(FluidContext* ctx, float* p, const float* div);
typedef void (*Preconditioner)(FluidContext* ctx, float* z, const float* r);

// Preconditioned Conjugate Gradient workspace
typedef struct {
    float* r;                       // Residual
    float* d;                       // Search direction
    float* q;                       // Laplacian of the search direction
    float* z;                       // Preconditioned residual
} PcgWorkspace;

// Diagnostics of the last pressure solve
typedef struct {
    size_t iterations;              // Iterations the solve ran
    int converged;                  // 1 if the threshold stopped the solve, 0 if the iteration cap did
    float delta_p;                  // Largest pressure change in the final iteration
    float residual;                 // True residual of the final pressure, max |Lap(p) - div * cp|
    float rhs;                      // Largest right-hand side term, max |div * cp|
    double solve_ms;                // Wall time of the solve, residual check excluded
} SolveStats;

// Fluid Simulation Context
struct FluidContext {

    // Domain Dimensions
    size_t x;                       // Grid width in pixels
    size_t y;                       // Grid height in pixels
    size_t num_cells;               // grid height * grid width in pixels

    // Physics Properties
    float dt;                       // Time step in seconds
    float dx;                       // Grid size in meters
    float dens;                     // Density of the fluid
    float visc;                     // Dynamic viscosity of the fluid

    // Solver Settings
    float omega;                    // Over-relaxation factor for pressure solver
    size_t poisson_iter;            // Iteration count for Poisson solver
    float threshold;                // Convergence threshold for iterative solvers
    PressureSolver pressure_solver; // Function pointer to the selected pressure solver
    Preconditioner precondition;    // Function pointer to the selected preconditioner

    // State
    float* u;                       // Horizontal velocity
    float* v;                       // Vertical velocity
    float* p;                       // Pressure, also the warm start of the next solve
    float* smoke;                   // Smoke
    uint8_t* solid;                 // Boundary Mask
    size_t step;                    // Steps taken since the last reset
    double time;                    // Simulated time since the last reset, the sum of every dt

    // Workspace
    float* div;                     // Divergence, the right-hand side of the pressure solve
    float* u_prev;                  // Ping-pong buffer for horizontal velocity
    float* v_prev;                  // Ping-pong buffer for vertical velocity
    float* smoke_prev;              // Ping-pong buffer for smoke
    PcgWorkspace cg;                // Conjugate Gradient vectors

    // Diagnostics
    SolveStats stats;               // Last pressure solve
};

// Scenario Parameters
typedef struct {
    float inlet_velocity;           // Characteristic velocity for the scenario
    float length_scale;             // Characteristic length scale for the scenario

    float obstacle_x;               // X-coordinate of the center of the obstacle
    float obstacle_y;               // Y-coordinate of the center of the obstacle
    size_t obstacle_radius;         // Radius of the obstacle

    float chord_length;             // Length of the airfoil's chord
    float angle_of_attack;          // Angle of attack for the airfoil

} ScenarioParams;

#endif // TYPES_H