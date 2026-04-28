#ifndef DEFINES_H
#define DEFINES_H

#include <stddef.h>

#include "scenarios.h"

#if defined(LID_DRIVEN)
    #define INIT_SCENARIO()       init_lid_driven(ctx)
    #define APPLY_SOURCES()       apply_sources_lid_driven(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_lid_driven(ctx, p)
#elif defined(KARMAN_VORTEX)
    #define INIT_SCENARIO()       init_karman_vortex(ctx, p)
    #define APPLY_SOURCES()       apply_sources_karman_vortex(ctx)
    #define APPLY_BOUNDARIES()    apply_boundaries_karman_vortex(ctx, p)
#elif defined(AIRFOIL)
    #define INIT_SCENARIO()       init_airfoil(ctx, p)
    #define APPLY_SOURCES()       apply_sources_airfoil(ctx, p)
    #define APPLY_BOUNDARIES()    apply_boundaries_airfoil(ctx, p)
#elif defined(URBAN_CITY)
    #define INIT_SCENARIO()       init_urban_city(ctx, p)
    #define APPLY_SOURCES()       apply_sources_urban_city(ctx, p)
    #define APPLY_BOUNDARIES()    apply_boundaries_urban_city(ctx, p)
#endif

#define PI 3.14159265358979323846f

#endif // DEFINES_H