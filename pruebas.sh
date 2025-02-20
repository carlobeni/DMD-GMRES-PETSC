#!/bin/bash
# Build the PETSc executable
make

# Define the PETSc executable and output file for each method
MATH_NAME="sherman5"
MATH_DIR="mtx_collection/${MATH_NAME}"
PETSC_EXEC="./exe_file"

TOLERANCE=1e-10

# List of KSP methods to test
KSP_METHODS=("gmres" "lgmres")

# Create a directory to store residual norms
mkdir -p residual_norms
mkdir -p residual_norms_python

# Loop through each KSP method
for method in "${KSP_METHODS[@]}"; do
    echo "Running test with KSP method: $method"

    # Run the test and redirect the output to a file
    $PETSC_EXEC -matrix_file "${MATH_DIR}/${MATH_NAME}.mtx" -vector_file "${MATH_DIR}/${MATH_NAME}_b.mtx" -tolerance $TOLERANCE -solver $method -ksp_monitor_short > "residual_norms/res_norm_$method.txt"

    # Extract norms residuals and save to a separate file
    grep "KSP Residual norm" "residual_norms/res_norm_${method}.txt" | awk '{print $1, $5}' > "residual_norms_python/value_${method}.txt"
done

echo "Tests completed. Residual norms saved in the 'residual_norms' folder."
python3 plot_residuals.py
