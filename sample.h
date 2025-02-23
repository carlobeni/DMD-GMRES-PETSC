#ifndef SAMPLE_VIEW_H
#define SAMPLE_VIEW_H

#include <petscksp.h>

/* Estructura para almacenar la información del monitor */
typedef struct {
    // Matriz de soluciones por iteración
    PetscScalar **Xsol; 
    // Vector de normas de residuo por iteración
    PetscReal *residuals;
    // Tamaño de la matriz de soluciones
    PetscInt size;
    PetscInt max_iters;
} SolutionStorage;

PetscErrorCode CaptureMonitor(KSP ksp, PetscInt iter, PetscReal norm, void *ctx);

PetscErrorCode SampleKSPIterations(KSP ksp,PetscInt m, Mat A, Vec b, Vec x0, PetscInt num_samples, PetscReal tol, Mat *Xsol);

#endif
