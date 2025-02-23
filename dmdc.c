#include "dmdc.h"
/*-----------------------------------------------------------
  DMDc: Calcula la evolución aproximada del sistema en una ventana
  a partir de las aproximaciones muestreadas (StateData) y de la
  entrada (InputData). Se realizan dos SVD reales:
    1. Sobre Omega = [ StateData(:,1:end-1); InputData(:,1:end-1) ]
       se extrae U1 (primeras n filas) y U2 (últimas q filas) y Vtil.
    2. Sobre StateData para obtener la base Uhat.
  Finalmente se calcula:
    approxH = Uhat' * Xp * Vtil * diag(1./S_trunc) * (U1)' * Uhat
    approxB = Uhat' * Xp * Vtil * diag(1./S_trunc) * (U2)'
  y se evoluciona la dinámica en el espacio reducido para reconstruir Xapprox = Uhat*xtil.
-----------------------------------------------------------*/
PetscErrorCode DMDc(Mat StateData, PetscInt m, Mat InputData, PetscInt tau, PetscScalar thresh,
                    Mat *Xapprox_out)
{
  PetscInt n, p, q;

  PetscFunctionBegin;
  //PetscCall(PetscPrintf(PETSC_COMM_SELF, "----------Starting DMDc...----------\n"));

  PetscCall(MatGetSize(StateData, &n, &p));
  PetscCall(MatGetSize(InputData, &q, NULL));

  // Verificar que StateData tenga al menos 3 columnas
  if (p < 2)
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Las ventanas deben tener al menos 2 columnas, es decir p >= 2");

  /* --- Construir Omega --- */
  //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Building Omega...\n"));
  // X = StateData(:, 1:end-1): n x (p-1)
  Mat X;
  PetscCall(CopyMatrixByColumns(StateData, &X, 0, p - 2));
  // Xp = StateData(:, 2:end): n x (p-1)
  Mat Xp;
  PetscCall(CopyMatrixByColumns(StateData, &Xp, 1, p - 1));
  // Up = InputData(:, 1:end-1): q x (p-1)
  Mat Up;
  PetscCall(CopyMatrixByColumns(InputData, &Up, 0, p - 2));
  // Omega: [X;Up]: (n+q) x (p-1)
  Mat Omega;
  PetscCall(ConcatMatAMatB(X, Up, &Omega, PETSC_FALSE));

  /* SVD de Omega usando SLEPc */
  //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Computing SVD of Omega...\n"));
  Mat S_omega, U_omega, V_omega;
  PetscCall(ComputeSVD_slepcs(Omega, &U_omega, &S_omega, &V_omega, thresh));
  PetscCall(MatDestroy(&Omega));

  /* SVD de StateData usando SLEPc */
  //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Computing SVD of StateData...\n"));
  Mat S_state, U_state, V_state;
  PetscCall(ComputeSVD_slepcs(StateData, &U_state, &S_state, &V_state, thresh));

  // U_1 = U_omega(1:n, :);
  Mat U1;
  PetscCall(CopyMatrixByRows(U_omega, &U1, 0, n - 1));
  // Transponer U1 en otra matriz
  Mat U1t;
  PetscCall(MatTranspose(U1, MAT_INITIAL_MATRIX, &U1t));

  // U_2 = U_omega(n+1:n+q, :);
  Mat U2;
  PetscCall(CopyMatrixByRows(U_omega, &U2, n, n + q - 1));
  // Transponer U2 en otra matriz
  Mat U2t;
  PetscCall(MatTranspose(U2, MAT_INITIAL_MATRIX, &U2t));

  /* --- Calcular aproximación --- */
  // approxH = U_state' * Xp * V_omega * inv(S_omega) * U_1' * U_state;
  // approxB = U_state' * Xp * V_omega * inv(S_omega) * U_2';
  Mat approxH, approxB;
  Mat A1, A2, A3, A4, A5;
  // Transponer U_state
  PetscCall(MatTranspose(U_state, MAT_INITIAL_MATRIX, &A1));
  // Multiplicar A1 por Xp
  PetscCall(MatMatMult(A1, Xp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A2));
  // Multiplicar A2 por V_omega
  PetscCall(MatMatMult(A2, V_omega, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A3));
  // Calcular la inversa de S_omega
  Mat Sinv;
  PetscCall(ComputeDiagonalInverse(S_omega, &Sinv));
  // Multiplicar A3 por Sinv
  PetscCall(MatMatMult(A3, Sinv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A4));
  // Multiplicar A4 por U1'
  PetscCall(MatMatMult(A4, U1t, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A5));
  // Multiplicar A5 por U_state
  PetscCall(MatMatMult(A5, U_state, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &approxH));

  // Multiplicar A4 por U2'
  PetscCall(MatMatMult(A4, U2t, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &approxB));
  // Mostar approxB
  PetscCall(MatView(approxB, PETSC_VIEWER_STDOUT_WORLD));

  // Para approxB * u;
  Vec vec_approx_B; // Convertir approxB en vector
  PetscInt n_approx_B;
  PetscCall(MatGetSize(approxB, &n_approx_B, NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_approx_B, &vec_approx_B));
  PetscCall(MatGetColumnVector(approxB, vec_approx_B, 0));
  PetscCall(VecScale(vec_approx_B, m));

  // Construir Xtil
  Mat Xtil;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n_approx_B, p + tau, NULL, &Xtil));
  Vec x_init_state;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x_init_state));
  PetscCall(MatGetColumnVector(StateData, x_init_state, 0));

  Vec x_til;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_approx_B, &x_til));

  // x_til[0] = U_state_T * StateData(:, 1);
  PetscCall(MatMultTranspose(U_state, x_init_state, x_til));
  const PetscScalar *x_til_array;
  PetscCall(VecGetArrayRead(x_til, &x_til_array));
  for (PetscInt j = 0; j < n_approx_B; j++)
  {
    PetscCall(MatSetValue(Xtil, j, 0, x_til_array[j], INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(x_til, &x_til_array));

  // x_til[i+1] = approxH * x_til[i] + approxB * u;
  Vec x_til_next;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_approx_B, &x_til_next));
  for (PetscInt i = 1; i < p + tau; i++)
  {
    PetscCall(MatMultAdd(approxH, x_til, vec_approx_B, x_til_next));
    const PetscScalar *x_til_next_array;
    PetscCall(VecGetArrayRead(x_til_next, &x_til_next_array));
    for (PetscInt j = 0; j < n_approx_B; j++)
    {
      PetscCall(MatSetValue(Xtil, j, i, x_til_next_array[j], INSERT_VALUES));
    }
    PetscCall(VecRestoreArrayRead(x_til_next, &x_til_next_array));
    PetscCall(VecCopy(x_til_next, x_til));
  }
  PetscCall(MatAssemblyBegin(Xtil, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Xtil, MAT_FINAL_ASSEMBLY));

  // X_approx = U_state * Xtil;
  PetscCall(MatMatMult(U_state, Xtil, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Xapprox_out));

  // Liberar recursos
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Xp));
  PetscCall(MatDestroy(&Up));
  PetscCall(MatDestroy(&U1));
  PetscCall(MatDestroy(&U1t));
  PetscCall(MatDestroy(&U2));
  PetscCall(MatDestroy(&U2t));
  PetscCall(MatDestroy(&S_omega));
  PetscCall(MatDestroy(&U_omega));
  PetscCall(MatDestroy(&V_omega));
  PetscCall(MatDestroy(&S_state));
  PetscCall(MatDestroy(&U_state));
  PetscCall(MatDestroy(&V_state));
  PetscCall(MatDestroy(&approxH));
  PetscCall(MatDestroy(&approxB));
  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&A3));
  PetscCall(MatDestroy(&A4));
  PetscCall(MatDestroy(&A5));
  PetscCall(MatDestroy(&Sinv));
  PetscCall(VecDestroy(&vec_approx_B));
  PetscCall(MatDestroy(&Xtil));
  PetscCall(VecDestroy(&x_init_state));
  PetscCall(VecDestroy(&x_til));
  PetscCall(VecDestroy(&x_til_next));

  //PetscPrintf(PETSC_COMM_SELF, "----------DMDc completed successfully.----------\n");
  return 0;
}

/*-----------------------------------------------------------
  calculateDMDcWin:
    Parámetros:
    - ksp: contexto KSP
    - m: parámetro de reinicio del KSP
    - A: matriz del sistema
    - b: vector del lado derecho
    - x0: vector inicial
    - tol: tolerancia
    - p: ancho de ventana
    - tau: número de aproximaciones
    - num_win: número de ventanas
    - thresh: umbral para el SVD
    - Xdmdc_out: salida de la matriz Xdmdc

    Funcionamiento:
      - Muestreo de p iteraciones de GMRES (SampleKSPIterations)
      - Construcción de InputData (todos los valores iguales a m)
      - Llamada a DMDc para obtener Xapprox
      - Extracción de la submatriz Xdmdc formada por las columnas 2 a (p+τ+1)
    La salida es una única matriz Xdmdc en cuyas columnas se encuentran
    los p+τ vectores del modelo.
-----------------------------------------------------------*/
PetscErrorCode calculateDMDcWin(KSP ksp, PetscInt m, Mat A, Vec b, Vec x0, PetscReal tol,
                                PetscInt p, PetscInt tau, PetscInt num_win, PetscScalar thresh,
                                Mat *Xdmdc_out)
{
  PetscInt kk;
  Vec x_local;
  PetscCall(VecDuplicate(x0, &x_local));
  PetscCall(VecCopy(x0, x_local));

  PetscInt n;
  PetscCall(MatGetSize(A, &n, NULL));

  /* Crear Xout de tamaño n x ((p+tau)*num_win) */
  Mat Xout;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, (p+tau)*num_win, NULL, &Xout));
  PetscCall(MatZeroEntries(Xout));

  for (kk = 0; kk < num_win; kk++) {
    Mat Xsol, InputData, Xwin;
    PetscCall(SampleKSPIterations(ksp, m, A, b, x_local, p, tol, &Xsol));
    PetscCall(MatGetColumnVector(Xsol, x_local, p-1));

    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, p, NULL, &InputData));
    {
      PetscScalar *arr;
      PetscCall(MatDenseGetArray(InputData, &arr));
      for (PetscInt j = 0; j < p; j++) arr[j] = (PetscScalar)m;
      PetscCall(MatDenseRestoreArray(InputData, &arr));
    }

    PetscCall(DMDc(Xsol, m, InputData, tau, thresh, &Xwin));
    
    /* Copiar Xwin en Xout; índice corregido para matriz densa (columna-mayor) */
    PetscInt col_offset = kk * (p + tau);
    PetscInt nrows, ncols;
    PetscCall(MatGetSize(Xwin, &nrows, &ncols));
    {
      PetscScalar *win_array;
      PetscCall(MatDenseGetArray(Xwin, &win_array));
      for (PetscInt j = 0; j < ncols; j++) {
        for (PetscInt i = 0; i < nrows; i++) {
          PetscCall(MatSetValue(Xout, i, col_offset + j, win_array[i + j*nrows], INSERT_VALUES));
        }
      }
      PetscCall(MatDenseRestoreArray(Xwin, &win_array));
      //Mensaje de Ventana Cargada
      PetscPrintf(PETSC_COMM_SELF, "Ventana %d cargada.\n", kk+1);
    }

    PetscCall(MatDestroy(&Xsol));
    PetscCall(MatDestroy(&InputData));
    PetscCall(MatDestroy(&Xwin));
    //PetscPrintf(PETSC_COMM_SELF, "Ventana %d completada.\n", kk+1);
  }
  //Mensaje de salida de ventanas completadas con par (m,p)
  PetscPrintf(PETSC_COMM_SELF, "Ventanas completadas con (m,p) = (%d,%d)\n", m, p);

  PetscCall(MatAssemblyBegin(Xout, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Xout, MAT_FINAL_ASSEMBLY));

  *Xdmdc_out = Xout;
  PetscCall(VecDestroy(&x_local));
  return 0;
}
