#ifndef PRECONDITIONERS_H
#define PRECONDITIONERS_H

#include "types.h"

typedef enum {
    PRECOND_IDENTITY,   // No preconditioning
    PRECOND_JACOBI,     // Jacobi preconditioning
    PRECOND_MULTIGRID   // Multigrid preconditioning (not implemented yet)
} PrecondType;

// Jacobi preconditioner: M_inv = diag(A)^(-1)
void precondition_jacobi(FluidContext* ctx, float* z, const float* r);

// Identity preconditioner (no preconditoning)
void precondition_identity(FluidContext* ctx, float* z, const float* r);

// Placeholder for multigrid preconditioner (not implemented yet)
void precondition_multigrid(FluidContext* ctx, float* z, const float* r);

#endif // PRECONDITIONERS_H