#include "gen_mat_tools.h"
#include "converter_mtx.h" /* Funciones para leer archivos .mtx */ // Cambiar nombre a load_mat_tools
#include "sample.h"        /* Función SampleKSPIterations */
#include "dmdc.h"          /* Función calculateDMDcViews */

#include <petscksp.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    Vec x0, b;
    Mat A;
    KSP ksp; // Contexto del solucionador
    PC pc;   // Contexto del precondicionador
    PetscReal tol = 1e-5;
    //char matrixInputFile[PETSC_MAX_PATH_LEN] = "mtx_collection/sherman5/sherman5.mtx";
    //char vectorInputFile[PETSC_MAX_PATH_LEN] = "mtx_collection/sherman5/sherman5_b.mtx";
    char solverType[PETSC_MAX_PATH_LEN] = "gmres";
    char preconditionerType[PETSC_MAX_PATH_LEN] = "none";
    PetscInt sizen;
    PetscScalar thresh = 1e-2;

    // Initialize Slepc permite calcular el SVD y cuenta con InitializePetsc
    PetscCall(SlepcInitialize(&argc, &argv, NULL,
                              "Uso: ./petsc_dmdc_svd_optimized\n\n"
                              "Resuelve Ax=b con GMRES y aplica DMDc optimizado con SVD alternativo.\n"));
    PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &sizen));
    if (sizen > 1)
        SETERRQ(PETSC_COMM_SELF, 1, "Ejecutar en un solo procesador");

    /* Generar A y b con los valores especificados */
    /* Parámetros de la matriz a generar */
    PetscInt n = 10;
    PetscScalar delta = 0.3, const_min = 1, const_max = 2.0;
    PetscCall(GenerateMatrixAndVector(n, delta, const_min, const_max, &A, &b));

    /* Configurar GMRES y el precondicionador */
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, solverType));
    PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, preconditionerType));
    PetscCall(KSPSetFromOptions(ksp));

    /* Vector de inicalizacion x0=b */
    PetscCall(VecDuplicate(b, &x0));
    PetscCall(VecCopy(b, x0));

    // Muestrear soluciones con SampleKSPIterations
    PetscInt p = 2, tau = 0, kkmax = 3, m = 4;
    Mat Xgmres;
    PetscCall(SampleKSPIterations(ksp, 1, A, b, x0, kkmax * p, tol, &Xgmres));
    //Mostrar Xgmres
    PetscCall(MatView(Xgmres, PETSC_VIEWER_STDOUT_WORLD));

    // Muestrear soluciones con calculateDMDcWin
    Mat Xdmdc;
    PetscCall(calculateDMDcWin(ksp, m, A, b, x0, tol, p, tau, kkmax, thresh, &Xdmdc));
    //Mostrar Xdmdc
    PetscCall(MatView(Xdmdc, PETSC_VIEWER_STDOUT_WORLD));

    /* Liberar recursos */
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x0));
    PetscCall(SlepcFinalize());
    return 0;
}
