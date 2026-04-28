#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include <stddef.h>

#include "types.h"

// Applies no-slip condition (zero velocity) on all solid boundaries
void bound_apply_no_slip(FluidContext* ctx);

// Applies slip condition (zero normal velocity) on top and bottom boundaries
void bound_apply_slip_horizontal(FluidContext* ctx);

// Applies continuous wind and optional smoke band from the left boundary
void bound_apply_inlet_left(FluidContext* ctx, float velocity, float smoke_density, size_t smoke_start_j, size_t smoke_end_j);

// Applies zero-gradient condition on the right boundary for free flow
void bound_apply_outlet_right(FluidContext* ctx);

// Helper to build outer solid walls (1 = build, 0 = open)
void bound_build_outer_walls(FluidContext* ctx, int top, int bottom, int left, int right);

#endif // BOUNDARIES_H