import numpy as np
import numba as nb
from .hnf_functions import _hnf_form_fast

"""
16/05/2026: numba is giving some deprecation warnings, these are to do with the list comprehensions I think, but using nb.typed.List() is giving me problems when I try to convert it to a numpy array so I'm leaving this for now.
"""

@nb.njit(fastmath = True)
def _permutations_numba(iterable):

    pool = np.array([i for i in iterable], dtype = np.int64)
    n = len(pool)
    r = n
    
    indices = np.array([i for i in range(n)], dtype = np.int64)
    cycles = np.array([i for i in range(n, n-r, -1)], dtype = np.int64)
    yield np.array([pool[i] for i in indices[:r]], dtype = np.int64)

    while n:
        for i in range(r-1,-1,-1):
            cycles[i] -= 1
            if cycles[i] == 0:
                new_indices = np.empty_like(indices)
                new_indices[:i] = indices[:i]
                new_indices[i:-1] = indices[i+1:]
                new_indices[-1] = indices[i]
                indices = new_indices.copy()
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield np.array([pool[i] for i in indices[:r]], dtype = np.int64)
                break
        else:
            return

@nb.njit(fastmath=True)
def _minimise_simple_permutations(lattice_signature, periodicity_vectors):
    minimal = np.full(
        (periodicity_vectors.shape[0], periodicity_vectors.shape[1]),
        np.iinfo(np.int64).max,
        dtype = np.int64,
    )
    for permutation_A in _permutations_numba(range(lattice_signature[0])):
        for permutation_B in _permutations_numba(range(lattice_signature[0], sum(lattice_signature))):
            permutation = np.zeros((sum(lattice_signature)), dtype = np.int64)
            permutation[:lattice_signature[0]] = permutation_A
            permutation[lattice_signature[0]:] = permutation_B
            new_periodicity_vectors = np.empty_like(periodicity_vectors)
            for i in range(periodicity_vectors.shape[0]):
                new_periodicity_vectors[:, i] = periodicity_vectors[:, permutation[i]].copy()
            to_check = _hnf_form_fast(new_periodicity_vectors)
            for i in range(to_check.shape[0]):
                for j in range(to_check.shape[1]):
                    if to_check[i,j] < minimal[i,j]:
                        minimal = to_check.copy()
                        break
                    elif to_check[i,j] > minimal[i,j]:
                        break
                else:
                    continue
                break
    return minimal

@nb.njit(fastmath=True)
def _standard_permutation_fast(lattice_signature, periodicity_vectors):
    minimal = np.full(
        (periodicity_vectors.shape[0], periodicity_vectors.shape[1]),
        np.iinfo(np.int64).max,
        dtype = np.int64,
    )
    for i in range(-1, lattice_signature[0]):
        for j in range(-1, lattice_signature[1]):
            new_periodicity_vectors = np.empty_like(periodicity_vectors)
            for k in range(lattice_signature[0]):
                if k == i:
                    new_periodicity_vectors[:,k] = - periodicity_vectors[:,0].copy()
                    for l in range(1, lattice_signature[0]):
                        new_periodicity_vectors[:,k] += - periodicity_vectors[:,l]
                else:
                    new_periodicity_vectors[:,k] = periodicity_vectors[:, k]
            for k in range(lattice_signature[1]):
                if k == j:
                    new_periodicity_vectors[:,k + lattice_signature[0]] = - periodicity_vectors[:,lattice_signature[0]]
                    for l in range(1, lattice_signature[1]):
                        new_periodicity_vectors[:, k + lattice_signature[0]] += - periodicity_vectors[:,l + lattice_signature[0]]
                else:
                    new_periodicity_vectors[:, k + lattice_signature[0]] = periodicity_vectors[:, k + lattice_signature[0]]
            to_check = _minimise_simple_permutations(
                lattice_signature,
                new_periodicity_vectors
            )
            for m in range(to_check.shape[0]):
                for n in range(to_check.shape[1]):
                    if to_check[m,n] < minimal[m,n]:
                        minimal = to_check.copy()
                        break
                    elif to_check[m,n] > minimal[m,n]:
                        break
                else:
                    continue
                break
    return minimal