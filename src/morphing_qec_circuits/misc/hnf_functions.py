import numpy as np
import numba as nb
from itertools import product

@nb.njit(fastmath=True)
def _eea(a, b):
    x = 1
    y = 0
    x1 = 0
    y1 = 1
    a1 = a
    b1 = b
    while b1 != 0:
        q = a1 // b1
        x, x1 = (x1, x - q*x1)
        y, y1 = (y1, y - q*y1)
        a1, b1 = (b1, a1 - q*b1)
    return a1, x, y

@nb.njit(fastmath=True)
def _hnf_form_fast(A):
    n = A.shape[0]
    for c in range(n-1,-1,-1):
        for r in range(c):
            if A[r, c] != 0:
                g, k, l = _eea(A[c,c], A[r,c])
                Ar = - A[r,c]//g * A[c,:] + A[c,c]//g * A[r,:]
                Ac = k*A[c,:] + l*A[r,:]
                A[r,:] = Ar
                A[c,:] = Ac
    for c in range(n-1,-1,-1):
        if A[c,c] < 0:
            A[c,:] = -A[c,:]
        for r in range(c+1,n):
            A[r,:] = A[r,:] - A[r,c]//A[c,c] * A[c,:]
    return A

def hnf_form(A):
    """Calculates the Hermite Normal Form of a square matrix using the "naive algorithm" from https://eprint.iacr.org/2024/2089.pdf. Specifically, it returns a matrix H in row-style lower-triangular form such that:
        1. H = UA for some unimodular matrix U,
        2. H is lower triangular,
        3. The "pivot" of each row (i.e. last nonzero entry of each row from the left) is positive and strictly to the right of the pivot of the row above it, and
        4. The elements above each pivot are zero and the elements below each pivot are non-negative and strictly less than the pivot.

    Args:
        A: a square matrix (preferably a numpy array but also works for Iterable[Iterable]).

    Returns:
        The row-style lower-triangular Hermite Normal Form of A.

    Raises:
        ValueError: Expected a square matrix.
    """
    A = np.array(A, dtype = int)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square matrix, but got a {A.shape[0]}x{A.shape[1]} matrix")
    return _hnf_form_fast(A)

def hnf_iterator(size, det):
    """Iterates over square matrices in Hermite normal form of a given size and with a given determinant.

    Args:
        size: An Int setting the shape of the yielded square matrices.
        det: An Int specifying the determinant of the yielded matrices.

    Yields:
        Square matrices with shape (size x size) and determinant det in row-style lower-triangular Hermite Normal form.
    """
    initial_diag = np.ones(size - 1, dtype = int)
    initial_diag = np.append(initial_diag, det)
    hnf = np.diag(initial_diag).astype(int)
    yield hnf
    
    while True:
        for i,j in product(reversed(range(1,size)), reversed(range(size-1))):
            if i <= j:
                continue
            if hnf[i,j] != hnf[j,j] - 1:
                break
        else:
            max_val = hnf[size-1,size-1]
            for i in reversed(range(size-1)):
                max_val = max_val * hnf[i,i]
                if hnf[i,i] != max_val:
                    break
            else:
                return 
            for new_val in range(hnf[i,i] + 1, max_val + 1):
                if max_val % new_val == 0:
                    hnf[i,i] = new_val
                    break
            for j in range(i+1, size-1):
                hnf[j,j] = 1
            last_el = det
            for j in range(size-1):
                last_el = last_el // hnf[j,j]
            hnf[size-1, size-1] = last_el
            for i,j in product(reversed(range(1,size)), reversed(range(size-1))):
                if i > j:
                    hnf[i,j] = 0
            yield hnf
            continue
        
        hnf[i,j] += 1
        for k in range(i+1, size):
            hnf[k,j] = 0
        for ell in range(j+1, size-1):
            for k in range(ell+1, size):
                hnf[k, ell] = 0
        yield hnf