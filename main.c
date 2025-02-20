#include "gen_mat_tools.h"
#include "converter_mtx.h" // Funciones para leer archivos .mtx
#include "sample.h"        // Función SampleKSPIterations
#include "dmdc.h"          // Función calculateDMDcViews

#include <petscksp.h>
#include <slepcsvd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

// Función para generar la matriz A y el vector b
PetscErrorCode GenerarMatrizYVector(PetscInt n, PetscScalar delta, Mat *A, Vec *b)
{
    PetscFunctionBegin;
    PetscScalar const_min = 0.1, const_max = 2.0;
    PetscCall(GenerateMatrixAndVector(n, delta, const_min, const_max, A, b));
    PetscFunctionReturn(0);
}

// Función para configurar el solucionador KSP
PetscErrorCode ConfigurarKSP(Mat A, KSP *ksp)
{
    PetscFunctionBegin;
    PC pc;
    PetscReal tol = 1e-5;
    char solverType[PETSC_MAX_PATH_LEN] = KSPGMRES;
    char preconditionerType[PETSC_MAX_PATH_LEN] = PCNONE;

    PetscCall(KSPCreate(PETSC_COMM_WORLD, ksp));
    PetscCall(KSPSetOperators(*ksp, A, A));
    PetscCall(KSPSetType(*ksp, solverType));
    PetscCall(KSPSetTolerances(*ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPGetPC(*ksp, &pc));
    PetscCall(PCSetType(pc, preconditionerType));
    PetscCall(KSPSetFromOptions(*ksp));
    PetscFunctionReturn(0);
}

// Función para crear directorios si no existen
void CrearDirectorioSiNoExiste(const char *path)
{
    struct stat st = {0};
    if (stat(path, &st) == -1)
    {
        mkdir(path, 0700);
    }
}

// Función para ejecutar los bucles principales
PetscErrorCode EjecutarBuclesPrincipales(PetscInt n, PetscScalar delta, PetscInt m_max, PetscInt p_max, PetscScalar thresh, PetscReal tol, PetscInt tau, PetscInt kkmax)
{
    PetscFunctionBegin;
    char basePath[PETSC_MAX_PATH_LEN];
    snprintf(basePath, sizeof(basePath), "test/ZHONGn%d", n);

    // Crear directorios necesarios
    char normDir[PETSC_MAX_PATH_LEN], tableDir[PETSC_MAX_PATH_LEN];
    snprintf(normDir, sizeof(normDir), "%s/norm", basePath);
    snprintf(tableDir, sizeof(tableDir), "%s/table", basePath);

    CrearDirectorioSiNoExiste("test");
    CrearDirectorioSiNoExiste(basePath);
    CrearDirectorioSiNoExiste(normDir);
    CrearDirectorioSiNoExiste(tableDir);

    // Generar matriz A y vector b
    Mat A;
    Vec b;
    PetscCall(GenerarMatrizYVector(n, delta, &A, &b));

    // Configurar KSP
    KSP ksp;
    PetscCall(ConfigurarKSP(A, &ksp));

    // Vector de inicialización x0 = b
    Vec x0;
    PetscCall(VecDuplicate(b, &x0));
    PetscCall(VecCopy(b, x0));

    // Bucle principal
    for (PetscInt m = 1; m <= m_max; ++m)
    {
        for (PetscInt p = 2; p <= p_max; ++p)
        {
            // Muestrear soluciones con SampleKSPIterations
            Mat X_gmres;
            PetscCall(SampleKSPIterations(ksp, m, A, b, x0, kkmax * p, tol, &X_gmres));

            // Muestrear soluciones con calculateDMDcWin
            Mat X_dmdc;
            PetscCall(calculateDMDcWin(ksp, m, A, b, x0, tol, p, tau, kkmax, thresh, &X_dmdc));

            // Calcular y almacenar normas residuales
            char normFile[PETSC_MAX_PATH_LEN];
            snprintf(normFile, sizeof(normFile), "%s/ZHONGn%d_normR_m%d_p%d_delta%.1f.txt", normDir, n, m, p, delta);
            FILE *normFp = fopen(normFile, "w");
            if (!normFp)
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "No se pudo abrir el archivo para escribir las normas residuales");

            for (PetscInt k = 0; k < kkmax*p; ++k)
            {
                Vec x_dmdc_k, x_gmres_k, res_dmdc, res_gmres;
                // Inicializar los vectores
                PetscCall(VecDuplicate(b, &x_dmdc_k));
                PetscCall(VecDuplicate(b, &x_gmres_k));
                PetscCall(VecDuplicate(b, &res_dmdc));
                PetscCall(VecDuplicate(b, &res_gmres));
                PetscCall(MatGetColumnVector(X_dmdc, x_dmdc_k, k));
                PetscCall(MatGetColumnVector(X_gmres, x_gmres_k, k));
                PetscCall(VecDuplicate(b, &res_dmdc));
                PetscCall(VecDuplicate(b, &res_gmres));

                // Calcular residuales gmres
                PetscCall(MatMult(A, x_gmres_k, res_gmres));
                PetscCall(VecAXPY(res_gmres, -1.0, b));

                // Calcular residuales dmdc
                PetscCall(MatMult(A, x_dmdc_k, res_dmdc));
                PetscCall(VecAXPY(res_dmdc, -1.0, b));

                // Calcular normas de los residuales
                PetscReal norm_res_dmdc, norm_res_gmres;
                PetscCall(VecNorm(res_dmdc, NORM_2, &norm_res_dmdc));
                PetscCall(VecNorm(res_gmres, NORM_2, &norm_res_gmres));

                // Escribir las normas en el archivo
                fprintf(normFp, "Iteración %d: Norma Residual DMDc = %g, Norma Residual GMRES = %g\n", k + 1, (double)norm_res_dmdc, (double)norm_res_gmres);

                // Liberar recursos
                PetscCall(VecDestroy(&x_dmdc_k));
                PetscCall(VecDestroy(&x_gmres_k));
                PetscCall(VecDestroy(&res_dmdc));
                PetscCall(VecDestroy(&res_gmres));
            }
            fclose(normFp);

            // Calcular la norma de la diferencia entre X_dmdc y X_gmres
            Mat X_diff;
            PetscCall(MatDuplicate(X_dmdc, MAT_COPY_VALUES, &X_diff));
            PetscCall(MatAXPY(X_diff, -1.0, X_gmres, DIFFERENT_NONZERO_PATTERN));
            PetscReal norm_X_diff;
            PetscCall(MatNorm(X_diff, NORM_1, &norm_X_diff));

            // Almacenar la norma de la diferencia en un archivo
            char tableFile[PETSC_MAX_PATH_LEN];
            snprintf(tableFile, sizeof(tableFile), "%s/ZHONGn%d_table_mmax%d_pmax%d_delta%.1f.txt", tableDir, n, m_max, p_max, delta);
            FILE *tableFp = fopen(tableFile, "a");
            if (!tableFp)
                SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "No se pudo abrir el archivo para escribir la tabla");
            fprintf(tableFp, "m = %d, p = %d: Norma de la diferencia = %g\n", m, p, (double)norm_X_diff);
            fclose(tableFp);

            // Liberar recursos
            PetscCall(MatDestroy(&X_gmres));
            PetscCall(MatDestroy(&X_dmdc));
            PetscCall(MatDestroy(&X_diff));
        }
    }

    // Liberar recursos
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x0));
}

int main(int argc, char **argv)
{
    PetscCall(SlepcInitialize(&argc, &argv, NULL, "Uso: ./programa\n"));
    PetscInt n = 4; // Tamaño de la matriz
    PetscScalar delta = 1.3; // Perturbación de la diagonal de la matriz
    PetscInt m_max = 4; 
    PetscInt p_max = 4;

    // Parámetros de ventanas
    PetscScalar thresh = 1e-2; // Umbral para SVD
    PetscReal tol = 1e-5; // Tolerancia para GMRES
    PetscInt tau = 0;
    PetscInt kkmax = 10; // Numero de ventanas

    PetscCall(EjecutarBuclesPrincipales(n, delta, m_max, p_max, thresh, tol, tau, kkmax));
    PetscCall(SlepcFinalize());
    return 0;
}
