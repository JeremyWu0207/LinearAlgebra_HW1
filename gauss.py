import torch
from matrices_vectors import scalar_matrix, matrix_sum

# --- Elementary Row Operations (EROs) ---

def row_interchange(R, i, j):
    """Interchanges row i and row j of matrix R."""
    # We use clone() to ensure we don't have reference issues during swapping
    temp = R[i].clone()
    # TODO: Perform row interchange
    R[i] = R[j].clone()
    R[j] = temp
    return R

def row_scaling(R, i, s):
    """Scales row i of matrix R by a scalar s."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function to finish this row operation
    # Hint: To use the scalar_matrix, a row of 1D tensor should be reshaped to a row 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    row_2d = R[i].reshape(1, cols)
    scaled_row_2d = scalar_matrix(s, row_2d)
    R[i] = scaled_row_2d.flatten()
    return R

def row_addition(R, i, j, s):
    """Adds s times row i to row j."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function and matrix_sum to finish this row operation
    # Hint: To use the scalar_matrix and matrix_sum, a row of 1D tensor should be reshaped to a row 
    # 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    row_i_2d = R[i].reshape(1, cols)
    row_j_2d = R[j].reshape(1, cols)
    scaled_i_2d = scalar_matrix(s, row_i_2d)
    summed_j_2d = matrix_sum(row_j_2d, scaled_i_2d)
    R[j] = summed_j_2d.flatten()
    return R

# --- Gaussian Elimination using EROs ---

def gauss_elimination(A):
    """
    Transforms matrix A into RREF using the encapsulated row operations.
    """
    R = A.clone().to(torch.float32)
    rows, cols = R.shape
    
    pivot_row = 0
    pivot_col = 0

    # Set zero thresh (if entry is larger than this value, treat it as zero. Otherwise, treat it as nonzero)
    zero_thresh = 1e-6

    # TODO: Implement step 1 to step 4 in the lecture slides (the forward phase to Row Echelon Form) 
    '''講義內容
    Step 1.
    Determine the left most nonzero column.
    • This is a pivot column
    • The topmost position in this column is a pivot position

    Step 2.
    In the pivot column, choose any nonzero entry in a row that is not above the pivot row, and
    perform the appropriate row interchange to bring this entry into the pivot position.

    Step 3.
    Perform appropriate row addition to each lower row in order to change each entry below
    the pivot position into 0.

    Step 4.
    Ignore the row containing the pivot position and all rows above it. If there is a nonzero row
    that is not ignored, repeat steps 1-4 on the submatrix that remains. 
    '''
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions
    while pivot_row < rows and pivot_col < cols:
        max_idx = pivot_row + torch.argmax(torch.abs(R[pivot_row:rows, pivot_col]))
        # 投影片寫找到一個非零，但我選擇找一個非零且取絕對值最大的，這樣可以增加數值穩定性
        if torch.abs(R[max_idx, pivot_col]) < zero_thresh:
            R[max_idx, pivot_col] = 0.0
            pivot_col += 1
            continue
            
        if max_idx != pivot_row:
            R = row_interchange(R, pivot_row, int(max_idx))
            
        for i in range(pivot_row + 1, rows):
            if torch.abs(R[i, pivot_col]) > zero_thresh:
                factor = -R[i, pivot_col] / R[pivot_row, pivot_col]
                R = row_addition(R, pivot_row, i, factor.item())
                
                #R[i, pivot_col] = 0.0

        pivot_row += 1
        pivot_col += 1

    # TODO: Implement step 5 and step 6 in the lecture slides (the backward phase to Reduced Row Echelon Form) 
    '''講義內容
    Step 5.
    Scale the leading entry at the last pivot position to be 1. Then add an appropriate multiple of 
    - this row to every preceding row so that each entry above the pivot position is 0.

    Step 6.
    If step 5 was performed on the first row, stop. Otherwise, repeat step 5 on the preceding
    '''
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions
    for i in range(rows - 1, -1, -1):
        pivot_idx = -1
        for j in range(cols):
            if torch.abs(R[i, j]) > zero_thresh:
                pivot_idx = j
                break
                
        if pivot_idx != -1:
            val = R[i, pivot_idx].item()
            if abs(val - 1.0) > zero_thresh:
                R = row_scaling(R, i, 1.0 / val)
                
            for k in range(i - 1, -1, -1):
                if torch.abs(R[k, pivot_idx]) > zero_thresh:
                    factor = -R[k, pivot_idx]
                    R = row_addition(R, i, k, factor.item())
                    
    return R