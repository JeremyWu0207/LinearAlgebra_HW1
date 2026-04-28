import torch
from matrices_vectors import matrix_vector_product
from gauss import gauss_elimination

def test_invertibility(matrix_M):
    """
    Checks if a square matrix is invertible.
    """
    rows, cols = matrix_M.shape

    # 1. Invertibility is only defined for square matrices
    if rows != cols:
        return False

    # 2. TODO: Impelement the invertibility check by using the property:
    # A square matrix is invertible if and only if its Reduced Row Echelon Form (RREF)
    # is the Identity Matrix.

    # 取得矩陣的 RREF (利用 gauss.py 中的 gauss_elimination)
    rref_M = gauss_elimination(matrix_M)
    
    # 設定浮點數運算的容差閾值
    zero_thresh = 1e-6
    
    # 捨棄 API，手動檢驗 RREF 是否等同於單位矩陣
    for i in range(rows):
        for j in range(cols):
            val = rref_M[i, j].item()
            if i == j:
                # 對角線元素應趨近於 1
                if abs(val - 1.0) > zero_thresh:
                    return False
            else:
                # 非對角線元素應趨近於 0
                if abs(val) > zero_thresh:
                    return False
                    
    return True

def test_span(matrix_M, vector_b):
    """
    TODO: Determine if vector_b is in the span of the columns of matrix_M.
    Hint: Does Mx = b have a solution?
    Hint: Use solve_linear_equations
    # (Should done in HW2, not necessary to report)
    """
    solution = solve_linear_equations(matrix_M, vector_b)
    return solution is not None

def test_linear_dependence(matrix_M):
    """
    TODO: Determine if the columns of matrix_M are linearly dependent.
    
    Logic: The columns are linearly dependent if there exists a NON-ZERO 
           vector 'x' such that Mx = 0.
    
    Challenge: Our 'solve_linear_equations' might return the trivial x=0.
    Hint: If the system has a unique solution (only x=0), they are Independent.
          If the system has infinite solutions (free variables), they are Dependent.
          Think about how your solver handles singular matrices.
    Hint: Nonzero: Euclidean norm > small epsilon (ex: 1e-6)
    Hint: Use solve_linear_equations
    # (Should done in HW2, not necessary to report)
    """
    b = torch.zeros((matrix_M.shape[0], 1), dtype=torch.float32)
    solution = solve_linear_equations(matrix_M, b)
    if solution is None:
        return False
    return torch.norm(solution).item() > 1e-6

def test_consistency(augmented_RREF):
    """
    Checks if the system Ax = b is consistent.
    """
    rows, cols = augmented_RREF.shape
    # TODO: Use the following property to determine consistency:
    # A system is inconsistent if and only if a row in the RREF looks like [0 0 ... 0 | b_i] 
    # where b_i != 0. 
    # Return False if inconsistent 
    # (Should done in HW1, not necessary to report)
    for r in range(rows):
        left_side = augmented_RREF[r, :-1]
        b_i = augmented_RREF[r, -1]
        if torch.all(torch.abs(left_side) < 1e-5).item() and torch.abs(b_i).item() >= 1e-5:
            return False

    return True # Consistent

def generate_solution(augmented_RREF):
    """
    Extracts a solution vector. Basic variables are solved, Free variables are set to 1.
    """
    rows, cols_aug = augmented_RREF.shape
    num_vars = cols_aug - 1
    solution = torch.ones((num_vars, 1), dtype=torch.float32) # Default free variables to 1
    
    # TODO: Identify pivot columns
    # Hint: Find the first nonzero in each row
    # (Should done in HW1, not necessary to report)
    pivot_cols = []
    for r in range(rows):
        for c in range(num_vars):
            if torch.abs(augmented_RREF[r, c]).item() > 1e-5:
                pivot_cols.append((r, c))
                break
    
    # TODO: Calculate basic variables
    # Hint: Note that we have assume all free variables are set to 1
    # (Should done in HW1, not necessary to report)
    for r, c in pivot_cols:
        sum_others = 0.0
        for j in range(num_vars):
            if j != c:
                sum_others += augmented_RREF[r, j] * solution[j, 0]
        solution[c, 0] = (augmented_RREF[r, -1] - sum_others) / augmented_RREF[r, c]
            
    return solution

def solve_linear_equations_by_inverse(A, b):
    """
    Solves Ax = b using the inverse matrix method: x = A^(-1)b.
    The inverse is calculated via Gaussian elimination on [A | I].
    """
    # 1. First, check if the matrix is invertible
    if not test_invertibility(A):
        return None

    num_vars = A.shape[1]
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    #TODO: Implement the matrix inversion method to get the inverse of A
    # 列數為 num_vars，行數為 num_vars * 2
    augmented_matrix = torch.zeros((num_vars, num_vars * 2), dtype=torch.float32)
    
    # 填入 A 矩陣與單位矩陣 I 的數值
    for i in range(num_vars):
        for j in range(num_vars):
            # 左半邊填入原矩陣 A
            augmented_matrix[i, j] = A[i, j]
        # 右半邊對角線填入 1，形成單位矩陣
        augmented_matrix[i, num_vars + i] = 1.0
        
    # 對增廣矩陣執行高斯消去法取得 RREF
    augmented_rref = gauss_elimination(augmented_matrix)
    
    # 右半部作為反矩陣
    A_inverse = augmented_rref[:, num_vars:]

    #TODO: Solve the solution by using inverse of A and return the result
    solution = matrix_vector_product(A_inverse, b)
    
    return solution


def solve_linear_equations(A, b):
    """
    Wrapper function to solve Ax = b.
    """

    # Create Augmented Matrix [A | b]
    augmented_matrix = torch.cat((A, b), dim=1)

    rows, cols_aug = augmented_matrix.shape
    num_vars = cols_aug - 1
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    
    # TODO: Use our custom gauss_elimination to obtain the Reduced Row Echelon Form.
    augmented_RREF = gauss_elimination(augmented_matrix)
    # Then, use test_consistency to inspect the reduced matrix.
    if not test_consistency(augmented_RREF):
        return None
    # Return None if the system is inconsistent. Otherwise, return the results of generate_solution.
    return generate_solution(augmented_RREF)
