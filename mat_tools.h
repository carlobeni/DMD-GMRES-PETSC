#ifndef MAT_TOOLS_H
#define MAT_TOOLS_H

#include <petscksp.h>
#include <petscblaslapack.h>
#include <slepcsvd.h>

PetscErrorCode CopyMatrixByRows(Mat A, Mat *B, PetscInt rw1, PetscInt rw2);

PetscErrorCode CopyMatrixByColumns(Mat A, Mat *B, PetscInt c1, PetscInt c2);

PetscErrorCode CopyMatrixbyBlock(Mat A, Mat *B, PetscInt x1, PetscInt y1, PetscInt x2, PetscInt y2);

PetscErrorCode ConcatMatAMatB(Mat A, Mat B, Mat *C, PetscBool isbyColumn);

PetscErrorCode LoadMatBlock(Mat Mat_out, Mat Mat_block, PetscInt pos_col);

PetscErrorCode ComputeDiagonalInverse(Mat A, Mat *A_inv);

PetscErrorCode ComputeSVD_slepcs(Mat A, Mat *U_out, Mat *S_out, Mat *V_out,PetscReal thresh);

#endif
