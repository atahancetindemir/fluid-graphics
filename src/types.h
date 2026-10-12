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

// Stopping rule of the pressure solve: max|r| <= max(rtol * max|b|, atol), at most max_iter iterations
typedef struct {
    float rtol;                     // Tolerance relative to the largest right-hand side term
    float atol;                     // Absolute floor, only reached when the right-hand side is ~0
    size_t max_iter;                // Safety cap on iterations
} SolveTolerance;

// How the last pressure solve ended
typedef enum {
    SOLVE_CONVERGED,                // The true residual met the tolerance
    SOLVE_CAPPED,                   // max_iter was reached first
    SOLVE_STALLED                   // PCG broke down, or its recurrence residual drifted from the true one
} SolveStatus;

// Diagnostics of the last pressure solve
typedef struct {
    SolveStatus status;             // How the solve ended
    size_t iterations;              // Iterations the solve ran
    size_t replacements;            // PCG restarts from the true residual
    float delta_p;                  // Largest pressure change in the final iteration
    float initial_residual;         // True residual of the warm start, max |Lap(p) - div * cp|
    float residual;                 // True residual of the final pressure, max |Lap(p) - div * cp|
    float rhs;                      // Largest right-hand side term, max |div * cp|
    double solve_ms;                // Wall time of the solve
} SolveStats;

// Flux through the domain boundary, per unit depth
typedef struct {
    float inflow;                   // Total flux entering the domain
    float outflow;                  // Total flux leaving the domain
    float left;                     // Net outflow through the left side
    float right;                    // Net outflow through the right side
    float bottom;                   // Net outflow through the bottom side
    float top;                      // Net outflow through the top side
} MassBalance;

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
    SolveTolerance tol;             // Stopping rule of the pressure solve
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
    float continuity_rms;           // RMS over fluid cells of |net face flux| / U after the last step
    float continuity_max;           // Largest |net face flux| / U over fluid cells after the last step
    MassBalance mass;               // Boundary flux after the last step
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