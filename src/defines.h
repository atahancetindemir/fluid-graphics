#ifndef DEFINES_H
#define DEFINES_H

#include "scenarios.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846f
#endif

#define SCENARIO_KARMAN

#ifdef SCENARIO_LID_DRIVEN
    #define INIT_SCENARIO()       init_lid_driven(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_lid_driven(ctx, p)
    #define APPLY_SOURCES()       apply_sources_lid_driven(ctx)
#elif defined(SCENARIO_KARMAN)
    #define INIT_SCENARIO()       init_karman_vortex(ctx, p)
    #define APPLY_BOUNDARIES()    apply_boundaries_karman_vortex(ctx, p)
    #define APPLY_SOURCES()       apply_sources_karman_vortex(ctx)
#elif defined(SCENARIO_AIRFOIL)
    #define INIT_SCENARIO()       init_airfoil(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_airfoil(ctx)
    #define APPLY_SOURCES()       apply_sources_airfoil(ctx)
#elif defined(SCENARIO_WIND_OVER_CITY)
    #define INIT_SCENARIO()       init_wind_over_city(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_wind_over_city(ctx)
    #define APPLY_SOURCES()       apply_sources_wind_over_city(ctx)
#endif

#define IX(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_U(ctx, i, j) ((size_t)(i) * (ctx)->y + (size_t)(j))
#define IX_V(ctx, i, j) ((size_t)(i) * ((ctx)->y + 1) + (size_t)(j))

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define SWAP_PTR(x0, x) do { float* tmp = x0; x0 = x; x = tmp; } while(0)

// // Layout
// #define X 128
// #define Y 128

// // Parameters
// #define DT 0.016f
// #define DX 0.1f
// #define DENSITY 1.0f
// #define ITER 20
#define SEED 54
// #define VISCOSITY 0.01f

// // Grid Cell Type
// unsigned char solid[X][Y];

#endif // DEFINES_H