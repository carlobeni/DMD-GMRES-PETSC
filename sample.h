#ifndef SAMPLE_VIEW_H
#define SAMPLE_VIEW_H

#include <petscksp.h>

/* Estructura para almacenar la información del monitor */
typedef struct {
  Mat               Xview;       /* Matriz donde se almacenan los p vectores (secuencial) */
  PetscInt          p;           /* Número de aproximaciones a almacenar */
  PetscInt          count;       /* Cantidad de muestreos realizados */
  PetscReal         tol;         /* Tolerancia para el método iterativo */
  PetscReal         *residuals;  /* Array (de tamaño p) con las normas residuales */
  PetscLogDouble   *iter_times; /* Array (de tamaño p) con el tiempo (por iteración) */
  PetscLogDouble    t_start;     /* Tiempo de inicio del proceso */
  PetscLogDouble    last_time;   /* Tiempo de la última iteración */
  PetscInt          vecSize;     /* Tamaño global del vector solución (no se usa para el copiado) */
} MonitorCtx;

PetscErrorCode SampleKSPIterations(KSP ksp,PetscInt m, Mat A, Vec b, Vec x0, PetscInt num_samples, PetscReal tol, Mat *Xsol);

#endif
