#include "sample.h"
#include <string.h> /* Para memcpy, si se desea usar */

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
PetscErrorCode SampleKSPIterations(KSP ksp, PetscInt m, Mat A, Vec b, Vec x0, PetscInt num_samples, PetscReal tol, Mat *Xsol)
{
  PetscErrorCode ierr;
  PetscInt n, j, nlocal;
  PetscMPIInt size;
  PetscInt i;
  Vec x0_local;

  PetscCall(VecGetSize(x0, &n));
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  // Configurar el ksp
  PetscCall(KSPGMRESSetRestart(ksp, m));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, 1)); 

  /* Crear la matriz para almacenar las aproximaciones: cada columna es una iteración */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, num_samples, NULL, Xsol));

  /* Crear un vector local para x0 */
  PetscCall(VecDuplicate(x0, &x0_local));
  PetscCall(VecCopy(x0, x0_local));

  for (i = 0; i < num_samples; i++)
  {
    // Resolver el sistema con el estado actualizado
    PetscCall(KSPSolve(ksp, b, x0_local));

    /* Almacenar la aproximación actual en la columna i de Xsol */
    const PetscScalar *xarray;
    PetscCall(VecGetArrayRead(x0_local, &xarray));
    PetscCall(VecGetLocalSize(x0_local, &nlocal));
    for (j = 0; j < nlocal; j++)
    {
      PetscCall(MatSetValue(*Xsol, j, i, xarray[j], INSERT_VALUES));
    }
    PetscCall(VecRestoreArrayRead(x0_local, &xarray));
  }

  /* Ensamblar la matriz con las aproximaciones */
  PetscCall(MatAssemblyBegin(*Xsol, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*Xsol, MAT_FINAL_ASSEMBLY));

  /* Liberar recursos */
  VecDestroy(&x0_local);

  return 0;
}
