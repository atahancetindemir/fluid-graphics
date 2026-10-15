#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include <stddef.h>

#include "types.h"

// Applies no-slip condition (zero velocity) on all solid boundaries; outer wall ghosts mirror the fluid side
void bound_apply_no_slip(FluidContext* ctx);

// Symmetry (free-slip) condition on the top and bottom boundaries
// Expects solid top and bottom rows, Call after bound_apply_no_slip
void bound_apply_free_slip_horizontal(FluidContext* ctx);

// Opening condition on the top and bottom boundaries: flow may cross them
// Expects non-solid top and bottom rows, where the pressure stays at zero
void bound_apply_opening_horizontal(FluidContext* ctx);

// Applies continuous wind and optional smoke band from the left boundary
// Expects a solid left column, Call after bound_apply_no_slip
void bound_apply_inlet_left(FluidContext* ctx, float velocity, float smoke_density, size_t smoke_start_j, size_t smoke_end_j);

// Applies zero-gradient condition on the right boundary for free flow
// The pressure stays at zero in the non-solid right column
void bound_apply_outlet_right(FluidContext* ctx);

// Helper to build outer solid walls (1 = build, 0 = open)
void bound_build_outer_walls(FluidContext* ctx, int top, int bottom, int left, int right);

#endif // BOUNDARIES_H