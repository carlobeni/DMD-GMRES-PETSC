#include "sample.h"
#include <string.h> /* Para memcpy, si se desea usar */

/* CaptureMonitor*/
PetscErrorCode CaptureMonitor(KSP ksp, PetscInt iter, PetscReal norm, void *ctx)
{
  SolutionStorage *sol_storage = (SolutionStorage *)ctx;
  Vec x;
  const PetscScalar *x_array;

  // Obtener el vector de solución en la iteración actual
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  PetscCall(VecGetArrayRead(x, &x_array));

  if (iter < sol_storage->max_iters)
  {
    for (PetscInt i = 0; i < sol_storage->size; i++)
    {
      sol_storage->Xsol[i][iter] = x_array[i];
    }
    sol_storage->residuals[iter] = norm;
  }

  // Liberar memoria
  PetscCall(VecRestoreArrayRead(x, &x_array));

  return 0;
}

/*-----------------------------------------------------------
  SampleKSPIterations:

  Parametros:
  - ksp: Contexto del solucionador
  - m: Número de vectores de la base de Krylov
  - A: Matriz del sistema
  - b: Vector del lado derecho
  - x0: Vector de inicialización
  - num_samples: Número de aproximaciones a almacenar
  - tol: Tolerancia para el método iterativo
  - Xsol: Matriz donde se almacenan las aproximaciones


   Funcionamiento:
   - Almacena en Xsol la matriz de iteraciones del ksp
   - Configura GMRES para realizar solo 1 iteración por llamada.
     Esto permite capturar el estado intermedio tras cada iteración.
     Se asegura además que el vector solución se trate como no nulo.
-----------------------------------------------------------*/
PetscErrorCode SampleKSPIterations(KSP ksp, PetscInt m, Mat A, Vec b, Vec x0, PetscInt num_samples, PetscReal tolerance, Mat *Xsol)
{
  PetscInt n;
  Vec x0_local;
  PetscCall(VecGetSize(x0, &n));

  // Configurar el ksp
  PetscCall(KSPSetTolerances(ksp, tolerance, PETSC_DEFAULT, PETSC_DEFAULT, num_samples)); // itermax = num_samples
  PetscCall(KSPGMRESSetRestart(ksp, m));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  /* Crear un vector local para x0 */
  PetscCall(VecDuplicate(x0, &x0_local));
  PetscCall(VecCopy(x0, x0_local));

  /* Crear la matriz para almacenar las aproximaciones: cada columna es una iteración */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, num_samples, NULL, Xsol));

  // Crear estructura para almacenar soluciones y normas de residuo
  SolutionStorage sol_storage;
  sol_storage.size = n;
  sol_storage.max_iters = num_samples;
  sol_storage.Xsol = (PetscScalar **)malloc(n * sizeof(PetscScalar *));
  for (PetscInt i = 0; i < n; i++)
    sol_storage.Xsol[i] = (PetscScalar *)malloc(num_samples * sizeof(PetscScalar));
  sol_storage.residuals = (PetscReal *)malloc(num_samples * sizeof(PetscReal));
  PetscCall(KSPMonitorSet(ksp, CaptureMonitor, (void *)&sol_storage, NULL));

  /* Resolver el sistema con el estado actualizado */
  PetscCall(KSPSolve(ksp, b, x0_local));

  /* Cancelar monitor */
  PetscCall(KSPMonitorCancel(ksp));

  // Obtener numero de iteraciones para convergencia
  PetscInt num_iters;
  PetscCall(KSPGetIterationNumber(ksp, &num_iters));

  /*
  PetscPrintf(PETSC_COMM_WORLD, "Numero de iteraciones: %d\n", num_iters);
  if (num_iters < num_samples-1)
    PetscPrintf(PETSC_COMM_WORLD, "ADVERTENCIA: Convergencia en %d iteraciones\n", num_iters);

  // Imprimir la matriz de resultados
  PetscPrintf(PETSC_COMM_WORLD, "\nMatriz Xsol (soluciones por iteración):\n");
  for (PetscInt i = 0; i < n; i++)
  {
    PetscPrintf(PETSC_COMM_WORLD, "Variable %d:", i);
    for (PetscInt j = 0; j < num_iters; j++)
    {
      PetscPrintf(PETSC_COMM_WORLD, " %g", (double)sol_storage.Xsol[i][j]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }

  // Imprimir la matriz de normas de residuos
  PetscPrintf(PETSC_COMM_WORLD, "\nMatriz residuals (normas de residuo por iteración):\n");
  for (PetscInt i = 0; i < num_iters; i++)
  {
    PetscPrintf(PETSC_COMM_WORLD, "Iter %d: %g\n", i+1, (double)sol_storage.residuals[i]);
  }
  */
  // Cargar solo num_iters iteraciones
  for (PetscInt i = 0; i < n; i++)
  {
    for (PetscInt j = 0; j < num_iters; j++)
      PetscCall(MatSetValue(*Xsol, i, j, sol_storage.Xsol[i][j], INSERT_VALUES));
  }

  // Ensamblar matriz
  PetscCall(MatAssemblyBegin(*Xsol, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Xsol, MAT_FINAL_ASSEMBLY));

  /* Liberar recursos */
  VecDestroy(&x0_local);
  for (PetscInt i = 0; i < n; i++)
    free(sol_storage.Xsol[i]);
  free(sol_storage.Xsol);
  free(sol_storage.residuals);

  return 0;
}
