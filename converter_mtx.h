#ifndef _CONVERTER_MTX_H
#define _CONVERTER_MTX_H

#include "petscmat.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct _MatrixInfo{
    int n;
    int m;
    int nnz;
} MatrixInfo;

PetscErrorCode MMTgetMatrix(const char *fin, Mat *A, MatrixInfo *minfo);

PetscErrorCode MMTgetVector(const char *fin, Vec *v, int *size);

PetscErrorCode MMTgetMatrixReal(const char *fin, Mat *A, MatrixInfo *minfo);

PetscErrorCode MMTgetVectorReal(const char *fin, Vec *v, int *size);

#endif
