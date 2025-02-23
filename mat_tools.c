#include "mat_tools.h"

/*
    ConcatMatAMatB:
    Concatena dos matrices densas A y B, ya sea por filas o por columnas.

    Parametros:
    - A: Matriz A
    - B: Matriz B
    - C: Matriz resultante de la concatenación
    - isbyColumn: Si es verdadero, se concatenan por columnas. Si es falso, se concatenan por filas.

*/
PetscErrorCode ConcatMatAMatB(Mat A, Mat B, Mat *C, PetscBool isbyColumn)
{
    PetscInt m, n, p, q, i;
    PetscScalar *arrayA, *arrayB;

    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatGetSize(B, &p, &q));

    if (isbyColumn)
    {
        if (m != p)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Para concatenación por columnas, A y B deben tener el mismo número de filas.");
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, n + q, NULL, C));
        PetscCall(MatDenseGetArray(A, &arrayA));
        PetscCall(MatDenseGetArray(B, &arrayB));

        PetscInt *rows;
        PetscCall(PetscMalloc1(m, &rows));
        for (i = 0; i < m; i++)
            rows[i] = i;

        PetscInt *colsA;
        PetscCall(PetscMalloc1(n, &colsA));
        for (i = 0; i < n; i++)
            colsA[i] = i;

        PetscInt *colsB;
        PetscCall(PetscMalloc1(q, &colsB));
        for (i = 0; i < q; i++)
            colsB[i] = n + i;

        PetscCall(MatSetValues(*C, m, rows, n, colsA, arrayA, INSERT_VALUES));
        PetscCall(MatSetValues(*C, m, rows, q, colsB, arrayB, INSERT_VALUES));

        PetscCall(PetscFree(rows));
        PetscCall(PetscFree(colsA));
        PetscCall(PetscFree(colsB));
        PetscCall(MatDenseRestoreArray(A, &arrayA));
        PetscCall(MatDenseRestoreArray(B, &arrayB));
    }
    else
    {
        if (n != q)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Para concatenación por filas, A y B deben tener el mismo número de columnas.");
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m + p, n, NULL, C));
        PetscCall(MatDenseGetArray(A, &arrayA));
        PetscCall(MatDenseGetArray(B, &arrayB));

        PetscInt *cols;
        PetscCall(PetscMalloc1(n, &cols));
        for (i = 0; i < n; i++)
            cols[i] = i;

        PetscInt *rowsA;
        PetscCall(PetscMalloc1(m, &rowsA));
        for (i = 0; i < m; i++)
            rowsA[i] = i;

        PetscInt *rowsB;
        PetscCall(PetscMalloc1(p, &rowsB));
        for (i = 0; i < p; i++)
            rowsB[i] = m + i;

        PetscCall(MatSetValues(*C, m, rowsA, n, cols, arrayA, INSERT_VALUES));
        PetscCall(MatSetValues(*C, p, rowsB, n, cols, arrayB, INSERT_VALUES));

        PetscCall(PetscFree(cols));
        PetscCall(PetscFree(rowsA));
        PetscCall(PetscFree(rowsB));
        PetscCall(MatDenseRestoreArray(A, &arrayA));
        PetscCall(MatDenseRestoreArray(B, &arrayB));
    }
    PetscCall(MatAssemblyBegin(*C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*C, MAT_FINAL_ASSEMBLY));

    return 0;
}
/*
    GetColumnVec:
    Obtiene una columna de una matriz densa y la guarda en un vector.

    Parametros:
    - A: Matriz densa
    - col: Columna a extraer
    - x: Vector donde se guardará la columna
*/
PetscErrorCode GetColumnVec(Mat A, PetscInt col, Vec *x)
{
    PetscInt m, n, i;
    const PetscScalar *Aarray;
    PetscScalar *vecArray;

    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, x));
    PetscCall(MatDenseGetArrayRead(A, &Aarray));
    PetscCall(VecGetArray(*x, &vecArray));

    for (i = 0; i < m; i++)
    {
        vecArray[i] = Aarray[i + col * m];
    }

    PetscCall(VecRestoreArray(*x, &vecArray));
    PetscCall(MatDenseRestoreArrayRead(A, &Aarray));
    return 0;
}

/*
    CopyMatrixByRows:
    Copia las filas de la matriz A (índices rw1 a rw2) en la matriz B.

    Parametros:
    - A: Matriz original
    - B: Matriz destino
    - rw1: Índice de la primera fila de A a copiar
    - rw2: Índice de la última fila de A a copiar

    Procedimiento:
    1. Transponer A para obtener At (de forma que las filas de A se convierten en columnas de At).
    2. Usar la función CopyMatrixByColumns para copiar las columnas de At (que son las filas de A) en B.
    3. Transponer B para obtener la matriz final con las filas de A copiadas.
*/
PetscErrorCode CopyMatrixByRows(Mat A, Mat *B, PetscInt rw1, PetscInt rw2)
{
    Mat At, Btemp;
    PetscInt n, m;

    PetscCall(MatGetSize(A, &m, &n));

    /* Verificar si los índices de fila están dentro del rango válido */
    if (rw1 < 0 || rw2 >= m || rw1 > rw2)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Los rangos de filas %d a %d no son válidos.\n", rw1, rw2));
        return PETSC_ERR_ARG_OUTOFRANGE;
    }

    /* Transponer A: At = A^T */
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));

    //Usar la funcion CopyMatrixByColumns
    PetscCall(CopyMatrixByColumns(At, &Btemp, rw1, rw2));

    /* Transponer Btemp para obtener B (de tamaño (rw2 - rw1 + 1) x n),
       que contendrá las filas de A copiadas */
    PetscCall(MatTranspose(Btemp, MAT_INITIAL_MATRIX, B));

    /* Liberar recursos */
    PetscCall(MatDestroy(&At));
    PetscCall(MatDestroy(&Btemp));

    return 0;
}

/*
    CopyMatrixByColumns:
    Copia las columnas de la matriz A (índices c1 a c2) en la matriz B.

    Parametros:
    - A: Matriz original
    - B: Matriz destino
    - c1: Índice de la primera columna de A a copiar
    - c2: Índice de la última columna de A a copiar

    Procedimiento:
      1. Para cada columna j en [c1, c2] se extrae el vector columna de A usando MatGetColumnVector.
      2. Se inserta el vector obtenido en la matriz B en la columna (j - c1).


*/
PetscErrorCode CopyMatrixByColumns(Mat A, Mat *B, PetscInt c1, PetscInt c2)
{
    PetscInt m, n, ncols_to_copy, i, r;
    PetscInt *rows = NULL;
    const PetscScalar *array;

    PetscCall(MatGetSize(A, &m, &n));

    /* Verificar si los índices de columna están dentro del rango válido */
    if (c1 < 0 || c2 >= n || c1 > c2)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Los rangos de columnas %d a %d no son válidos.\n", c1, c2));
        return PETSC_ERR_ARG_OUTOFRANGE;
    }

    ncols_to_copy = c2 - c1 + 1;

    /* Crear matriz B de tamaño m x (c2 - c1 + 1) */
    PetscCall(MatCreate(PETSC_COMM_WORLD, B));
    PetscCall(MatSetSizes(*B, PETSC_DECIDE, PETSC_DECIDE, m, ncols_to_copy));
    PetscCall(MatSetType(*B, MATDENSE));
    PetscCall(MatSetUp(*B));

    /* Crear un vector  de m filas y 1 columna sin usar A */
    Vec colVec;
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &colVec));

    /* Crear arreglo de índices de fila para la inserción masiva */
    PetscCall(PetscMalloc1(m, &rows));
    for (r = 0; r < m; r++)
    {
        rows[r] = r;
    }

    /* Extraer las columnas de A y copiarlas en B */
    for (i = c1; i <= c2; i++)
    {
        PetscCall(MatGetColumnVector(A, colVec, i));
        PetscCall(VecGetArrayRead(colVec, &array));
        {
            PetscInt col = i - c1;
            PetscCall(MatSetValues(*B, m, rows, 1, &col, array, INSERT_VALUES));
        }
        PetscCall(VecRestoreArrayRead(colVec, &array));
    }
    PetscCall(PetscFree(rows));

    /* Ensamblar la matriz B */
    PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));

    /* Liberar recursos */
    PetscCall(VecDestroy(&colVec));

    return 0;
}

/*
    CopyMatrixbyBlock:
    Copia un bloque de una matriz a otra de forma eficiente

    Parametros:
    - A: Matriz original
    - B: Matriz destino
    - x1: Fila inicial del bloque
    - y1: Columna inicial del bloque
    - x2: Fila final del bloque
    - y2: Columna final del bloque
*/
PetscErrorCode CopyMatrixbyBlock(Mat A, Mat *B, PetscInt x1, PetscInt y1, PetscInt x2, PetscInt y2)
{
    PetscInt m, n, i, j;
    PetscScalar *arrayA;

    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, x2 - x1 + 1, y2 - y1 + 1, NULL, B));
    PetscCall(MatDenseGetArray(A, &arrayA));

    for (i = x1; i <= x2; i++)
    {
        for (j = y1; j <= y2; j++)
        {
            PetscCall(MatSetValue(*B, i - x1, j - y1, arrayA[i + j * m], INSERT_VALUES));
        }
    }

    PetscCall(MatDenseRestoreArray(A, &arrayA));
    PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));

    return 0;
}
/*
    ComputeDiagonalInverse:
    Calcula la inversa de la diagonal de una matriz A y la guarda en una nueva matriz.

    Parametros:
    - A: Matriz original
    - A_inv: Matriz inversa de la diagonal de A
*/
PetscErrorCode ComputeDiagonalInverse(Mat A, Mat *A_inv) {
    Vec diag;
    PetscInt n, i;
    PetscScalar *diag_array, inv_value;

    PetscFunctionBegin;

    // Obtener el tamaño de la matriz
    MatGetSize(A, &n, NULL);

    // Crear una nueva matriz para la inversa
    MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 1, NULL, A_inv);

    // Obtener la diagonal de A
    VecCreateSeq(PETSC_COMM_SELF, n, &diag);
    MatGetDiagonal(A, diag);

    // Obtener acceso a los valores de la diagonal
    VecGetArray(diag, &diag_array);
    for (i = 0; i < n; i++) {
        if (diag_array[i] != 0.0) {
            inv_value = 1.0 / diag_array[i];
        } else {
            inv_value = 0.0; // O manejar error si se requiere
        }
        MatSetValue(*A_inv, i, i, inv_value, INSERT_VALUES);
    }
    VecRestoreArray(diag, &diag_array);

    // Ensamblar la matriz inversa
    MatAssemblyBegin(*A_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A_inv, MAT_FINAL_ASSEMBLY);

    // Limpiar memoria
    VecDestroy(&diag);

    PetscFunctionReturn(0);
}
/*
    ComputeSVD_slepcs:
    Calcula la descomposición en valores singulares de una matriz A usando SLEPc.

    Parametros:
    - A: Matriz a descomponer
    - U_out: Matriz U de la descomposición
    - S_out: Vector de valores singulares de la descomposición
    - V_out: Matriz V de la descomposición
*/
PetscErrorCode ComputeSVD_slepcs(Mat A, Mat *U_out, Mat *S_out, Mat *V_out, PetscReal thresh)
{
    SVD svd;
    Mat U, S, VT;
    PetscInt m, n, k;

    // Inicializar matrices de salida en NULL
    *U_out = NULL;
    *S_out = NULL;
    *V_out = NULL;

    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(SVDCreate(PETSC_COMM_WORLD, &svd));
    PetscCall(SVDSetOperators(svd, A, NULL));

    // Establecer LAPACK para evitar errores en EPS
    PetscCall(SVDSetType(svd, SVDLAPACK));
    PetscCall(SVDSetFromOptions(svd));
    PetscCall(SVDSolve(svd));
    PetscCall(SVDGetConverged(svd, &k));

    // Contar cuántos valores singulares son mayores al umbral
    PetscInt cont = 0;
    for (PetscInt i = 0; i < k; i++)
    {
        PetscScalar sigma;
        Vec u, vt;
        PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
        PetscCall(VecSetSizes(u, PETSC_DECIDE, m));
        PetscCall(VecSetFromOptions(u));

        PetscCall(VecCreate(PETSC_COMM_WORLD, &vt));
        PetscCall(VecSetSizes(vt, PETSC_DECIDE, n));
        PetscCall(VecSetFromOptions(vt));

        PetscCall(SVDGetSingularTriplet(svd, i, &sigma, u, vt));

        // Obtener tamaños de los vectores
        PetscInt mu, nv;
        PetscCall(VecGetSize(u, &mu));
        PetscCall(VecGetSize(vt, &nv));

        // Contar cuántos valores singulares son mayores al umbral
        if (sigma >= thresh)
            cont++;

        // Mostrar valores singulares
        PetscCall(VecDestroy(&u));
        PetscCall(VecDestroy(&vt));
    }

    // Si no hay valores singulares mayores al umbral, retornar
    if (cont == 0)
    {
        PetscPrintf(PETSC_COMM_SELF, "No se encontraron valores singulares mayores al umbral.\n");
        PetscCall(SVDDestroy(&svd));
        return 1;
    }

    // Si cont > k, ajustar a k
    if (cont > k)
        cont = k;

    // Crear matrices con las dimensiones ajustadas
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, cont, NULL, &U));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, cont, n, NULL, &VT));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, cont, cont, NULL, &S));

    // Asignar valores a las matrices U, S y VT
    for (PetscInt i = 0; i < cont; i++)
    {
        PetscScalar sigma;
        Vec u, vt;
        PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
        PetscCall(VecSetSizes(u, PETSC_DECIDE, m));
        PetscCall(VecSetFromOptions(u));

        PetscCall(VecCreate(PETSC_COMM_WORLD, &vt));
        PetscCall(VecSetSizes(vt, PETSC_DECIDE, n));
        PetscCall(VecSetFromOptions(vt));

        PetscCall(SVDGetSingularTriplet(svd, i, &sigma, u, vt));

        PetscScalar *array_u, *array_vt;
        PetscCall(VecGetArray(u, &array_u));
        PetscCall(VecGetArray(vt, &array_vt));

        // Asignar valores a las matrices U y VT
        for (PetscInt j = 0; j < m; j++)
        {
            PetscCall(MatSetValue(U, j, i, array_u[j], INSERT_VALUES));
        }
        for (PetscInt j = 0; j < n; j++)
        {
            PetscCall(MatSetValue(VT, i, j, array_vt[j], INSERT_VALUES));
        }

        // Asignar el valor singular a la matriz S
        PetscCall(MatSetValue(S, i, i, sigma, INSERT_VALUES));

        PetscCall(VecRestoreArray(u, &array_u));
        PetscCall(VecRestoreArray(vt, &array_vt));
        PetscCall(VecDestroy(&u));
        PetscCall(VecDestroy(&vt));
    }

    // Ensamblar las matrices U, S y VT
    PetscCall(MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(VT, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(VT, MAT_FINAL_ASSEMBLY));

    // Transponer VT para obtener V
    if (VT)
    {
        PetscCall(MatTranspose(VT, MAT_INITIAL_MATRIX, V_out));
        if (!(*V_out))
        {
            PetscPrintf(PETSC_COMM_WORLD, "Error: MatTranspose falló.\n");
            return 1;
        }
    }
    else
    {
        PetscPrintf(PETSC_COMM_WORLD, "Error: VT es NULL antes de transponer.\n");
        return 1;
    }

    // Asignar las matrices de salida
    *U_out = U;
    *S_out = S;

    // Liberar memoria
    PetscCall(SVDDestroy(&svd));

    // Liberar VT después de usarla
    if (VT)
        PetscCall(MatDestroy(&VT));

    return 0;
}
