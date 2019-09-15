# https://towardsdatascience.com/
# math-for-data-science-collaborative-filtering-on-utility-matrices-e62fa9badaab

import numpy as np

UI = [[4,2,3,5], [3,2,4,2], [0,4,5,4], [3,2,4,4]]
UI_np = np.array(UI)


def mean(a):
    assert len(a) > 0, "mean: vector must have values"
    return sum(a)/len(a)

def var(a):
    assert len(a) > 0, "var: vector must have values"
    mu = mean(a)
    return sum([(a[i] - mu)**2 for i in range(len(a))]) / len(a)

def cov(a, b):
    assert len(a) == len(b), "cov: vectors must be same length"
    mu_a, mu_b = mean(a), mean(b)
    return sum([(a[i] - mu_a) * (b[i] - mu_b) for i in range(len(a))]) / len(a)

def pearson_corr(a, b):
    assert len(a) == len(b), "pearson_corr: vectors must be same length"
    sig_a, sig_b, sig_ab = var(a), var(b), cov(a, b)
    return sig_ab / np.sqrt(sig_a * sig_b)

def cosine_sim(a, b):
    assert len(a) == len(b), "cosine_sim: vectors must be same length"
    vlen_a, vlen_b = np.dot(a, a), np.dot(b, b) # vector (euclidean) length
    assert vlen_a > 0 and vlen_b > 0
    return np.dot(a, b) / np.sqrt( vlen_a * vlen_b )


def get_indices(A, num = 0.0):
    index_list = []
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if A[i][j] == num:
                index_list.append((i,j))
    return index_list

# RMSE(A, B) computes the root mean squared error between A and B.

# ignore_zeroes 

# ignore_list is a list of cell indices to remove from the RMSE calculation.
# Its intent is to filter the 
# ignore_list is itself ignored (and overwritten) if ignore_zeroes == True.
    
# NOTE: A and B must be numpy arrays of the same (2-dim) shape.
# RMSE will compute cell-wise differences.

def RMSE(A, B, ignore_zeroes = False, ignore_list = []):
    assert np.shape(A) == np.shape(B), "RMSE: inputs must be same shape"
    assert type(ignore_list) is list,  "RMSE: ignore_list must be list"
    
    if ignore_zeroes:
        ignore_list = get_indices(A, 0.0)
        ignore_list.extend(get_indices(B, 0.0))

    assert len(ignore_list) < A.shape[0] * A.shape[1], "RMSE: all ignored"
    s = 0.0
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if A[i][j] != 0.0 and B[i][j] != 0.0:
                s += (A[i][j] - B[i][j])**2
    return np.sqrt(s / (A.shape[0] * A.shape[1] - len(ignore_list)))


# User-to-User
UI_col_lens = [np.sqrt(np.dot(UI_np.T[i], UI_np.T[i])) for i in range(4)]
U1_cos = [cosine_sim(UI_np.T[0], UI_np.T[i]) for i in range(4)]
sum_I3_cos = sum(U1_cos) - U1_cos[0] # subtract off the self entry of 1
weighted_U1_I3 = np.dot(UI_np[2], U1_cos) / sum_I3_cos # zero term drops


# Item-to-Item
UI_row_lens = [np.sqrt(np.dot(UI_np[i], UI_np[i])) for i in range(4)]
I3_cos = [cosine_sim(UI_np[2], UI_np[i]) for i in range(4)]
sum_U1_cos = sum(I3_cos) - I3_cos[2]
weighted_I3_U1 = np.dot(UI_np.T[0], I3_cos) / sum_U1_cos # zero term drops


# SVD: not really SVD, but a "reverse SVD"...
# "This is where machine learning comes in."
# Alternating Least Squares to "recreate the utility matrix" UI 
# and fill in the missing value (0).

# using # of latent features as a hyperparameter for each of U and I, 
# initialize a ones matrix for each.

I_latent_f = 2
U_latent_f = 2

I = np.ones((np.shape(UI_np)[0], I_latent_f))
U = np.ones((U_latent_f, np.shape(UI_np)[1]))

UI_candidate = np.matmul(I, U) # right now, all 2


# We won't use the surprise here, as the example is small and we can 
# brute force the search for a local minimum in each cell, down to a certain 
# threshold. We'll do so, knowing that all results in the approximate UI matrix 
# must be in the [1,5] range. We'll expand this to (0,6) for more leeway.

# Since we know that checking one variable in a matrix multiplication for 
# an optimal value results in a quadratic function (i.e. parabola) with 
# positive leading coefficient, and so there is a unique global minimum for 
# that problem.

# Note that, in Yish's example, the end choice of 3.7 contributes roughly 
# 0.925 to the RMSE (all other values held equal). So, her result of 1.15 
# is mostly due to that 0. This is an issue in the calculation that suggests  
# the RMSE metric as it stands, including that 0, is a bit problematic.
    

# Find_Cell_Value_For_RMSE returns the value of the cell in the decomposition 
# of original 
def Find_Cell_Value_For_Min_Err(i, j, left, right, orig, f = RMSE, 
                                cell_on_left = True, 
                             orig_min = 0.01, orig_max = 5.99, step = 0.01,
                             primed_min_err = 0):
    
    assert left.shape[1] == right.shape[0]
    assert left.shape[0] == orig.shape[0] and right.shape[1] == orig.shape[1]

    if cell_on_left:
        assert i in range(left.shape[0]) and j in range(left.shape[1])

        min_err = primed_min_err if primed_min_err > 0 \
            else orig.shape[0] * orig.shape[1] * orig_max # TODO fix hack
        
        x = orig_min
        left_temp = np.copy(left)
        while (x < orig_max):
            left_temp[i][j] = x
            LR_candidate = np.matmul(left_temp, right)
            err = f(LR_candidate, orig) # assuming f = RMSE here...
            if min_err > err:
                min_err = err
                min_err_x = x
#                print(f"FCVFME: (i,j)=({i},{j}), x={x}, min_err={min_err}")
            x += step
            # brute force for real here - just do all the calculations!
        return min_err_x
        
        
    else: # KISS: don't rewrite the entire function, just transpose everything
        assert i in range(right.shape[0]) and j in range(right.shape[1])
        return Find_Cell_Value_For_Min_Err(j, i, right.T, left.T, orig.T, 
                                           f, True, orig_min, orig_max, step)

# returns a np.array of the same shape as M, having replaced every cell 
# of M with 
def Fill_Cells_For_Min_Err(M, right, orig, rows_first = True, f = RMSE, 
                             orig_min = 0.01, orig_max = 5.99, step = 0.01):
    assert M.shape[1] == right.shape[0]
    assert M.shape[0] == orig.shape[0] and right.shape[1] == orig.shape[1]
    
    if rows_first:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i][j] = Find_Cell_Value_For_Min_Err(i, j, M, right, orig, 
                                           f, True, orig_min, orig_max, step)    
    else:
        for j in range(M.shape[1]):
            for i in range(M.shape[0]):
                # should be same exact line as above
                M[i][j] = Find_Cell_Value_For_Min_Err(i, j, M, right, orig, 
                                           f, True, orig_min, orig_max, step)    

    return M


# ----------------------------
    
if __name__ == "__main__":

#    """
    print(f"User-to-User: cosines with U1: ")
    print(f"cos(U1, U2) = {U1_cos[1]}")
    print(f"cos(U1, U3) = {U1_cos[2]}")
    print(f"cos(U1, U4) = {U1_cos[3]}")
    print()
    print(f"sum of U1 cosines (minus its own of 1): {sum_I3_cos}")
    print(f"weighted ave for U1: {weighted_U1_I3}")
    
    print("\n")
    print(f"Item-to-Item: cosines with I3: ")
    print(f"cos(I3, I1) = {I3_cos[0]}")
    print(f"cos(I3, I2) = {I3_cos[1]}")
    print(f"cos(I3, I4) = {I3_cos[3]}")
    print()
    print(f"sum of U1 cosines (minus its own of 1): {sum_U1_cos}")
    print(f"weighted ave for U1: {weighted_I3_U1}")
#    """
