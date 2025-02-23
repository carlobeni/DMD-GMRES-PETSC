#ifndef GEN_MAT_TOOLS_H
#define GEN_MAT_TOOLS_H

#include <petscksp.h>

PetscErrorCode GenZHONGAbn(PetscInt n, PetscScalar delta, PetscScalar const_min, PetscScalar const_max, Mat *A, Vec *b);

PetscErrorCode GenZHONGAb(PetscScalar delta, Mat *A, Vec *b);

#endif