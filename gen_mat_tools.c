#include "gen_mat_tools.h"
/* Genera una matriz y un vector aleatorios en el formato propuesto por BAOJIANG ZHONG*/
/*-----------------------------------------------------------
   GenZHONGAbn: Genera una matriz y un vector aleatorios en el formato propuesto por BAOJIANG ZHONG
    Parámetros:
        n: número de filas de la matriz
        delta: valor de la diagonal de la matriz
        const_min: valor mínimo de los elementos de la matriz
        const_max: valor máximo de los elementos de la matriz
        A: Matriz generada
        b: Vector generado
-----------------------------------------------------------*/
PetscErrorCode GenZHONGAbn(PetscInt n, PetscScalar delta, PetscScalar const_min, PetscScalar const_max, Mat *A, Vec *b)
{
    PetscInt i;
    PetscRandom rand;
    PetscScalar val;

    // Crear la matriz A
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 2, NULL, A));
    PetscCall(MatZeroEntries(*A)); // Inicializa A con ceros

    // Crear el vector b
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, b));

    // Crear un generador de números aleatorios
    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
    PetscCall(PetscRandomSetFromOptions(rand));

    // Llenar A y b
    for (i = 0; i < n; i++)
    {
        PetscCall(PetscRandomGetValueReal(rand, &val));
        val = const_min + (const_max - const_min) * val; // Escalar el valor aleatorio
        PetscCall(MatSetValue(*A, i, i, val, INSERT_VALUES));
        if (i < n - 1)
        {
            PetscCall(MatSetValue(*A, i, i + 1, delta, INSERT_VALUES));
            PetscCall(VecSetValue(*b, i, val + delta, INSERT_VALUES));
        }
        else
        {
            PetscCall(VecSetValue(*b, i, val, INSERT_VALUES));
        }
    }

    // Ensamblar matriz y vector
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(*b));
    PetscCall(VecAssemblyEnd(*b));

    // Liberar recursos
    PetscCall(PetscRandomDestroy(&rand));

    return 0;
}
/*
    GENZHONGAb:
    Genera la matriz A (4x4) y el vector b (4x1) de la ecuación de la forma propuesta por BAOJIANG ZHONG.

    Formulación:
    A = [0.5 delta 0 0; 0 1 delta 0; 0 0 1.5 delta; 0 0 0 2];
    b = [0.5+delta 1+delta 1.5+delta 2]';
*/
PetscErrorCode GenZHONGAb(PetscScalar delta, Mat *A, Vec *b){
    // Crear la matriz A
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 2, NULL, A));
    PetscCall(MatZeroEntries(*A)); // Inicializa A con ceros

    // Crear el vector b
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4, b));

    // Llenar A y b
    PetscCall(MatSetValue(*A, 0, 0, 0.5, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 0, 1, delta, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 1, 1, 1, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 1, 2, delta, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 2, 2, 1.5, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 2, 3, delta, INSERT_VALUES));
    PetscCall(MatSetValue(*A, 3, 3, 2, INSERT_VALUES));

    PetscCall(VecSetValue(*b, 0, 0.5 + delta, INSERT_VALUES));
    PetscCall(VecSetValue(*b, 1, 1 + delta, INSERT_VALUES));
    PetscCall(VecSetValue(*b, 2, 1.5 + delta, INSERT_VALUES));
    PetscCall(VecSetValue(*b, 3, 2, INSERT_VALUES));

    // Ensamblar matriz y vector
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(*b));
    PetscCall(VecAssemblyEnd(*b));
}

