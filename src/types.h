#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef struct FluidContext FluidContext;
typedef void (*PressureSolver)(FluidContext* ctx, float* p, const float* div);
typedef void (*Preconditioner)(FluidContext* ctx, float* z, const float* r);

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
    float reynolds;                 // Reynolds number
    float omega;                    // Over-relaxation factor for pressure solver
    int iter_count;                 // Iteration count for solver
    float threshold;                // Convergence threshold for iterative solvers
    PressureSolver pressure_solver; // Function pointer to the selected pressure solver

    // Vector Fields
    float* u;                       // Horizontal velocity
    float* v;                       // Vertical velocity
    
    // Solver Fields
    float* p;                       // Pressure
    float* div;                     // Divergence
    
    // Transport Fields
    float* smoke;                   // Smoke
    
    // Geometry
    uint8_t* solid;                 // Boundary Mask

    // Previous State
    float* u_prev;                  // Previous horizontal velocity
    float* v_prev;                  // Previous vertical velocity
    float* smoke_prev;              // Previous smoke

    Preconditioner precondition;    // Function pointer to the selected preconditioner
    float* cg_r;                    // Residual for Conjugate Gradient
    float* cg_d;                    // Direction for Conjugate Gradient
    float* cg_q;                    // Temporary vector for Conjugate Gradient
    float* cg_z;                    // Preconditioner vector for Conjugate Gradient
};

typedef struct {
    float inlet_velocity;           // Characteristic velocity for the scenario
    float length_scale;             // Characteristic length scale for the scenario
    float target_omega;             // Target over-relaxation factor for the pressure solver

    float obstacle_x;               // X-coordinate of the center of the obstacle
    float obstacle_y;               // Y-coordinate of the center of the obstacle
    size_t obstacle_radius;         // Radius of the obstacle

    float chord_length;             // Length of the airfoil's chord
    float angle_of_attack;          // Angle of attack for the airfoil

} ScenarioParams;

#endif // TYPES_H