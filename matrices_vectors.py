import torch

def scalar_matrix(s, M):
    rows = M.shape[0]
    cols = M.shape[1]
    # Initialize a tensor of zeros with the same shape as M
    sM = torch.zeros(M.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to scale each entry
    for i in range(rows):
        for j in range(cols):
            sM[i][j] = s * M[i][j]
    return sM

def matrix_sum(M1, M2):
    rows = M1.shape[0]
    cols = M1.shape[1]
    # Initialize a tensor of zeros with the same shape as M1
    M = torch.zeros(M1.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to add 
    # corresponding entries of M1 and M2 in a general manner.
    for i in range(rows):
        for j in range(cols):
            M[i][j] = M1[i][j] + M2[i][j]
    return M

def matrix_vector_product(M, vec):
    rows, cols = M.shape[0], M.shape[1]
    vec2 = torch.zeros((rows, 1), dtype=torch.float32)
    
    # TODO: Complete the functionality by implementing a for loop for a general linear combination.
    # Hint: Utilize the scalar_matrix() and matrix_sum() functions.
    for j in range(cols):
        col_j = torch.zeros((rows, 1), dtype=torch.float32)
        for i in range(rows):
            col_j[i, 0] = M[i, j]
            
        weight = vec[j, 0].item()
        scaled_col = scalar_matrix(weight, col_j)
        vec2 = matrix_sum(vec2, scaled_col)
    
    return vec2
def matrix_multiplication(M1, M2):
    m, n = M1.shape[0], M1.shape[1] # M1 is an m x n matrix
    _, p = M2.shape[0], M2.shape[1] # M2 is an n x p matrix
    M3 = torch.zeros((m, p), dtype=torch.float32)
    
    # TODO: Complete the functionality by implementing a for loop for matrix multiplication.
    # Hint: Utilize the matrix_vector_product for the implementation.
    # Hint: You may need reshape and flatten to control the shapes of vectors.
    for i in range(m):
        for j in range(p):
            for k in range(n):
                M3[i][j] += M1[i][k] * M2[k][j]
    
    return M3

def compute_rotation_matrix_2d(theta):
    # Initialize a 2x2 identity matrix 
    rot = torch.eye(2, dtype=torch.float32)
    # TODO: Complete the functionality by specifying a 2D rotation matrix
    # Using torch.cos and torch.sin for tensor-compatible math
    # theta > 0: counter-clockwise
    c = torch.cos(theta)
    s = torch.sin(theta)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return rot

def compute_y_mirror_matrix_2d():
    # Initialize a 2x2 identity matrix 
    mirror = torch.eye(2, dtype=torch.float32)
    # TODO: Complete the functionality by specifying a y-axis mirror matrix
    # This matrix flips the x-coordinate: [[-1, 0], [0, 1]]
    mirror[0, 0] = -1.0
    mirror[0, 1] = 0.0
    mirror[1, 0] = 0.0
    mirror[1, 1] = 1.0
    return mirror
