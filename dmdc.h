#ifndef DMDC_H
#define DMDC_H

#include <petscksp.h>
#include "mat_tools.h"
#include "sample.h"

PetscErrorCode DMDc(Mat StateData, PetscInt m, Mat InputData, PetscInt tau, PetscScalar thresh,
                    Mat *Xapprox_out);

PetscErrorCode calculateDMDcWin(KSP ksp, PetscInt m, Mat A, Vec b, Vec x0, PetscReal tol,
                                PetscInt p, PetscInt tau, PetscInt num_win, PetscScalar thresh,
                                Mat *Xdmdc_out);
#endif