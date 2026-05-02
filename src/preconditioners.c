#include "preconditioners.h"

#include <string.h>

#include "types.h"
#include "utilities.h"

void precondition_jacobi(FluidContext* ctx, float* z, const float* r) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 1; i < ctx->x - 1; i++) {
        for (size_t j = 1; j < ctx->y - 1; j++) {
            size_t idx = IX(ctx, i, j);
            if (ctx->solid[idx]) { z[idx] = 0.0f; continue; }

            int neighbors = 0;
            if (!ctx->solid[IX(ctx, i-1, j)]) neighbors++;
            if (!ctx->solid[IX(ctx, i+1, j)]) neighbors++;
            if (!ctx->solid[IX(ctx, i, j-1)]) neighbors++;
            if (!ctx->solid[IX(ctx, i, j+1)]) neighbors++;

            // Diagonal of A is -neighboors
            float diag = (neighbors > 0) ? -(float)neighbors : -4.0f;
            z[idx] = r[idx] / diag;
        }
    }
}

void precondition_identity(FluidContext* ctx, float* z, const float* r) {
    memcpy(z, r, ctx->num_cells * sizeof(float));
}

void precondition_multigrid(FluidContext* ctx, float* z, const float* r) {
    memcpy(z, r, ctx->num_cells * sizeof(float));
}