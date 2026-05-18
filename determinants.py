import torch
from gauss import row_interchange, row_addition

def calculate_determinant(matrix_M):
    """
    Calculates the determinant of a square matrix using the Gaussian elimination 
    logic (Forward Phase only) defined in gauss.py.
    
    Logic: 
    1. Transform the matrix to Row Echelon Form (REF).
    2. Tracking row swaps to flip the sign.
    3. Determinant = ((-1)^swaps) * product of diagonal entries.
    """
    R = matrix_M.clone().to(torch.float32)
    rows, cols = R.shape
    
    if rows != cols:
        return 0.0
    
    sign = 1.0
    pivot_row = 0
    pivot_col = 0
    # --- TODO: Forward Phase (Transforming to REF) ---
    # Hint: Use row interchange and row addition, not use row scaling
    # Hint: Implement the following while loop, update pivot_row and pivot_col properly
    # while pivot_row < rows and pivot_col < cols:
    #     ......
    threshold = torch.tensor(1e-6, dtype=R.dtype, device=R.device)
    while pivot_row < rows and pivot_col < cols:
        # 透過argmax取出從現在的列下面的最大值的相對位置 所以最後要加 piovt_row
        #取最大值的原因是因為避免除數過小加大誤差
        m = pivot_row + torch.argmax(torch.abs(R[pivot_row:rows, pivot_col])) 
        # 如果值太小就不算以免造成太大的誤差或出問題
        if torch.abs(R[m,pivot_col]) < threshold:
            R[m,pivot_col] = 0.0
            pivot_col += 1
            continue
        # 把他換到現在列的地方
        if m != pivot_row:
            R = row_interchange(R, pivot_row, int(m))
            sign *= -1.0
        # 把它下面都剪掉變成 Reduce Echelon Form
        for i in range(pivot_row + 1,rows):
            if torch.abs(R[i,pivot_col]) > threshold:
                a = -R[i,pivot_col] / R[pivot_row,pivot_col]
                R = row_addition(R, pivot_row, i, a.item())
        pivot_col += 1
        pivot_row += 1
    # --- Final Calculation ---
    # TODO: Replace the following line
    # Hint: The determinant is the sign-adjusted product of the diagonal elements
    #因為其他都是零所以REF的行列式只要左斜相乘就行了
    det = sign
    for i in range(rows):
        det *= R[i,i].item()
    return det
