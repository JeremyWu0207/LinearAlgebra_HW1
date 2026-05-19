import torch
from linear_solver import (
    solve_linear_equations, 
    solve_linear_equations_by_inverse, 
    test_invertibility, 
    test_invertibility_by_determinants,
    test_span,
    in_column_space, 
    in_null_space,
    generate_column_space_basis,
    generate_null_space_basis,
    test_linear_dependence
)
from matrices_vectors import matrix_vector_product, matrix_sum, scalar_matrix

def run_column_space_test(matrix_M, vector_b, matrix_M_basis, test_name):
    print(f"\n[TEST CASE: {test_name} ]")
    
    # 1. First check if b is in the column space of the original matrix
    if in_column_space(matrix_M, vector_b):
        print(f"Status: SUCCESS")
        print(f"Observation: Vector is confirmed to be in the Column Space of Matrix.")

        # 2. TODO: Double-check if the basis can actually span vector_b
        # If in the span, print the following message
        # "Verification: PASS (The vector is successfully spanned by the basis)."
        # If not in the span, print the following message
        # "Verification: FAIL (The basis does not span the vector)."
        
        # 按照題目要求 透過test_span 來確認他可不可以組合出 vector_b
        if test_span(matrix_M_basis, vector_b):
            print("Verification: PASS (The vector is successfully spanned by the basis).")
        # 如果不行就跳Fail
        else:
            print("Verification: FAIL (The basis does not span the vector).")
    # 如果不再column space ->不可能
    else:
        print(f"Status: FAILED")
        print(f"Observation: Vector b is NOT in the Column Space of the matrix.")
    
    print("-" * 40)

def run_null_space_test(matrix_M, vector_b, matrix_M_basis, test_name):
    print(f"\n[TEST CASE: {test_name}]")
    
    # 1. First check if b is in the null space of the original matrix
    if in_null_space(matrix_M, vector_b):
        print(f"Status: SUCCESS")
        print(f"Observation: Vector is confirmed to be in the null space of Matrix.")
        
        # 2. TODO: Double-check if the basis can actually span vector_b
        # If in the span, print the following message
        # "Verification: PASS (The vector is successfully spanned by the basis)."
        # If not in the span, print the following message
        # "Verification: FAIL (The basis does not span the vector)." 
        # 在檢查完是不是在Null Space後與 run_column_space_test 上面大致相同
        if test_span(matrix_M_basis, vector_b):
            print("Verification: PASS (The vector is successfully spanned by the basis).")
        else:
            print("Verification: FAIL (The basis does not span the vector).")
    else:
        print(f"Status: FAILED")
        print(f"Observation: Vector is NOT in the Null Space of the matrix.")
    
    print("-" * 40)

# --- 1. Define Matrix A---
matrix_A = torch.tensor([
    [1.0, 2.0,  1.0, -1.0],
    [3.0, 2.0,  4.0,  4.0],
    [4.0, 4.0,  5.0,  3.0], 
    [2.0, 0.0,  1.0,  5.0]
], dtype=torch.float32)

# --- 2. Define Matrix B ---
matrix_B = torch.tensor([
    [1.0, 2.0,  1.0, -1.0],
    [3.0, 5.0,  4.0,  4.0], 
    [4.0, 4.0,  9.0,  3.0], 
    [2.0, 0.0,  1.0,  8.0]  
], dtype=torch.float32)

# --- 3. The 6 Test Vectors---
test_b_vectors = [
    (torch.tensor([[4.0], [4.0], [8.0], [2.0]]),    "Vector_Alpha"),
    (torch.tensor([[-5.0], [3.5], [0.0], [2.0]]),   "Vector_Beta"),
    (torch.tensor([[1.0], [0.0], [0.0], [0.0]]),    "Vector_Gamma"),
    (torch.tensor([[2.0], [1.0], [3.0], [4.0]]),    "Vector_Delta"),
    (torch.tensor([[0.0], [0.0], [0.0], [0.0]]),    "Vector_Zero"),
    (torch.tensor([[1.0], [2.0], [3.0], [4.0]]),    "Vector_Epsilon")
]

if __name__ == "__main__":
    print("Linear Algebra Lab: Column Space & Null Space Analysis")
    print("=" * 60)

    # Part 1: Matrix Property Check
    # TODO: 
    # 1. Include results in the report.
    # 2. Provide an analysis the relation between:
    #    - linear dependence
    #    - invertiblity
    #    - invertibility_by_determinants 
    #    - column space basis 
    #    - null space basis
    print(f"Matrix A - Is Dependent: {test_linear_dependence(matrix_A)}")
    print(f"Matrix A - Is Invertible: {test_invertibility(matrix_A)}")
    print(f"Matrix A - Is Invertible By Determinants: {test_invertibility_by_determinants(matrix_A)}")

    matrix_A_column_basis = generate_column_space_basis(matrix_A)
    matrix_A_null_basis = generate_null_space_basis(matrix_A)
    print(f"Matrix A - Basis Matrix of Col A\n: {matrix_A_column_basis}")
    print(f"Matrix A - Basis Matrix of Null A\n: {matrix_A_null_basis}")

    print("-" * 60)
    print(f"Matrix B - Is Dependent: {test_linear_dependence(matrix_B)}")
    print(f"Matrix B - Is Invertible: {test_invertibility(matrix_B)}")
    print(f"Matrix B - Is Invertible By Determinants: {test_invertibility_by_determinants(matrix_B)}")

    matrix_B_column_basis = generate_column_space_basis(matrix_B)
    matrix_B_null_basis = generate_null_space_basis(matrix_B)
    print(f"Matrix B - Basis Matrix of Col B\n: {matrix_B_column_basis}")
    print(f"Matrix B - Basis Matrix of Null B\n: {matrix_B_null_basis}")

    print("=" * 60)


    # Part 2: Test Scenarios for matrix A
    # TODO:
    # 1. Include results in the report.
    # 2. Provide an analysis of the relation between:
    #    - Vector is in the space 
    #    - Vector can be spanned by which set, why?
    print("\n[SCENARIO 1-1] Column Space Testing with Matrix A")
    print("=" * 60)

    for vec, name in test_b_vectors:
        run_column_space_test(matrix_A, vec, matrix_A_column_basis, name)

    print("\n[SCENARIO 1-2] Null Space Testing with Matrix A")
    print("=" * 60)
    for vec, name in test_b_vectors:
        run_null_space_test(matrix_A, vec, matrix_A_null_basis, name)

    # Part 3: Test Scenarios for matrix B
    # TODO:
    # 1. Include results in the report.
    # 2. Provide an analysis of the relation between:
    #    - Vector is in the space 
    #    - Vector can be spanned by which set, why?
    print("\n[SCENARIO 2-1] Column Space Testing with Matrix B")
    print("=" * 60)

    for vec, name in test_b_vectors:
        run_column_space_test(matrix_B, vec, matrix_B_column_basis, name) 

    print("\n[SCENARIO 2-2] Null Space Testing with Matrix B")
    print("=" * 60)
    for vec, name in test_b_vectors:
        run_null_space_test(matrix_B, vec, matrix_B_null_basis, name)   
