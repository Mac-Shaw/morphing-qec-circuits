import numpy as np
import galois  # for matrix calculations over a finite field

import gurobipy as gp
from gurobipy import GRB

F2 = galois.GF(2)

def non_CSS_distance(h, symplectic_logical_basis, verbose = True, cutoff = np.inf):
    n_qubits = h.shape[1]//2
    dist = cutoff
    h = np.array(h).astype(np.int32)
    n_stabs = h.shape[0]
    zero_array = np.zeros((n_stabs)).astype(np.int32)
    logicals = F2(np.block([[symplectic_logical_basis["X"]],[symplectic_logical_basis["Z"]]]))
    for i in range(logicals.shape[0]):
        m = gp.Model()
        if not verbose:
            m.Params.OutputFlag = 0
            m.Params.LogToConsole = 0
        var = m.addMVar(shape=n_qubits * 2, vtype=GRB.BINARY, name="var")
        if not (0 in h.shape):
            dummy1 = m.addMVar(shape=n_stabs, vtype=GRB.INTEGER, name="dummy1")
            stab_constraint = m.addConstr(
                h @ var - 2 * dummy1 == zero_array, "stab_constraint"
            )
        dummy2 = m.addVar(vtype=GRB.INTEGER, name="dummy2")
        logical_constraint = m.addConstr(
            logicals[i] @ var - 2 * dummy2 == 1, "logical_constraint"
        )
        m.setObjective((1 - (1 - var[:n_qubits]) * (1 - var[n_qubits:])).sum(), GRB.MINIMIZE)
        if dist < np.inf:
            m.Params.Cutoff = dist
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            if m.ObjVal < dist:
                dist = m.ObjVal
        elif m.Status in [GRB.CUTOFF, GRB.INFEASIBLE]:
            pass
        else:
            raise Exception("Something went wrong, gurobi did not reach an optimum nor did it get cut off...")
    return round(dist)

def dx(hs, verbose = True, cutoff = np.inf):
    n_qubits = hs["X"].shape[1]
    if 0 in hs["X"].shape:
        n_qubits = hs["Z"].shape[1]
    n_logicals = (
        n_qubits - np.linalg.matrix_rank(hs["X"]) - np.linalg.matrix_rank(hs["Z"])
    )
    n_z_stabs = hs["Z"].shape[0]

    kerhx = hs["X"].null_space()
    if 0 in hs["X"].shape:
        kerhx = F2(np.eye(n_qubits).astype(int))
    temp = hs["Z"].row_reduce()
    imhzt = temp[temp.any((1))].copy()
    if 0 in imhzt.shape:
        z_logicals = kerhx
    else:
        stabs_to_logicals = imhzt.copy()
        z_logicals = F2([[]])
        for i in range(kerhx.shape[0]):
            new_vecs = np.append(stabs_to_logicals, [kerhx[i]], axis=0)
            if np.linalg.matrix_rank(new_vecs) != np.linalg.matrix_rank(
                stabs_to_logicals
            ):
                stabs_to_logicals = new_vecs.copy()
                if 0 in z_logicals.shape:
                    z_logicals = np.array([kerhx[i]]).copy()
                else:
                    z_logicals = np.append(z_logicals, [kerhx[i]], axis=0)
    dist = cutoff
    hz = np.array(hs["Z"]).astype(np.int32)
    zero_array = np.zeros((n_z_stabs)).astype(np.int32)
    for i in range(z_logicals.shape[0]):
        m = gp.Model()
        if not verbose:
            m.Params.OutputFlag = 0
            m.Params.LogToConsole = 0
        var = m.addMVar(shape=n_qubits, vtype=GRB.BINARY, name="var")
        if not (0 in hz.shape):
            dummy1 = m.addMVar(shape=n_z_stabs, vtype=GRB.INTEGER, name="dummy1")
            stab_constraint = m.addConstr(
                hz @ var - 2 * dummy1 == zero_array, "stab_constraint"
            )
        dummy2 = m.addVar(vtype=GRB.INTEGER, name="dummy2")
        logical_constraint = m.addConstr(
            z_logicals[i] @ var - 2 * dummy2 == 1, "logical_constraint"
        )
        m.setObjective(var.sum(), GRB.MINIMIZE)
        if dist < np.inf:
            m.Params.Cutoff = dist
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            if m.ObjVal < dist:
                dist = m.ObjVal
        elif m.Status in [GRB.CUTOFF, GRB.INFEASIBLE]:
            pass
        else:
            raise Exception("Something went wrong, gurobi did not reach an optimum nor did it get cut off...")
    return round(dist)


def dz(hs, verbose = False, cutoff = np.inf):
    swapped_hs = {"X": hs["Z"], "Z": hs["X"]}
    return dx(swapped_hs, verbose=verbose, cutoff = cutoff)


def distance(hs, check_x_and_z=True, separate_x_and_z=False, verbose=False, cutoff = np.inf):
    """
    Calculates the distance of a CSS code based on its X and Z parity check matrices, using Gurobi.

    Arguments:
        hs is a dictionary with keys "X" and "Z", and items that are numpy arrays corresponding to the X- and Z-parity check matrices, respectively.
        check_x_and_z is a bool. If True, then both the X- and Z-distance is calculated. If False, only the X-distance is calculated: this setting
            is useful if the code is guaranteed to have X-distance equal to Z-distance.
        separate_x_and_z is a bool. If True, then the function returns both the X- and the Z-distance separately; if False then a single integer
            is returned corresponding to the distance = min(X-distance, Z-distance).
        verbose is a bool, if False then the text output from the Gurobi solver is suppressed.
        cutoff is an Int (or np.inf) that provides an upper-bound to the returned distance. If the solver is able to lower-bound the distance to be d >= cutoff,
            the solver will stop and the function will simply return cutoff.

    Returns:
        If separate_x_and_z = False, an integer corresponding to the distance of the code.
        If separate_x_and_z = True, a dictionary with keys "X" and "Z" and items that are integers corresponding to the X- and Z-distance respectively.
    """
    x_distance = dx(hs, verbose=verbose, cutoff = cutoff)
    if check_x_and_z:
        z_distance = dz(hs, verbose=verbose, cutoff = x_distance)
        if separate_x_and_z:
            return {"X": x_distance, "Z": z_distance}
        else:
            return min(x_distance, z_distance)
    else:
        if separate_x_and_z:
            return {"X": x_distance, "Z": x_distance}
        else:
            return x_distance

def get_symplectic_logical_basis(h):
    """
    Adapted from Andrew's code
    """
    gauge_matrix = h
    n_qubits = h.shape[1]//2
    zero_block = np.zeros((n_qubits, n_qubits)).astype(int)
    identity_block = np.eye(n_qubits).astype(int)
    omega = galois.GF(2)(np.block([[zero_block, identity_block], [identity_block, zero_block]]))
    gauge_centraliser = (gauge_matrix@omega).null_space()
    
    """
    Adapted from Andrew's code. Yes I respelled center -> centre because I'm a fucking wanker.
    """
    
    pairs1=[]
    pairs2=[]
    centre=[]
    C = gauge_centraliser@omega@(gauge_centraliser.T)
    marked=np.zeros(gauge_centraliser.shape[0])
    for i in range(gauge_centraliser.shape[0]):
        if marked[i]==1 or np.count_nonzero(gauge_centraliser[i,:])==0:
            continue
        if np.count_nonzero(C[i,:])==0:
            centre.append(gauge_centraliser[i,:])
        else:
            pairs1.append(gauge_centraliser[i,:])
            nonzeros = np.nonzero(C[i,:])[0]
            j = nonzeros[0] # first noncommuting
            pairs2.append(gauge_centraliser[j,:])
            for k in nonzeros[1:]:
                gauge_centraliser[k,:]=gauge_centraliser[k,:]+gauge_centraliser[j,:]
                C[k,:]=C[k,:]+C[j,:]
                C[:,k]=C[:,k]+C[:,j]
            nonzeros = np.nonzero(C[j,:])[0]
            assert i in nonzeros
            for k in nonzeros:
                if k!=i:
                    gauge_centraliser[k,:]=gauge_centraliser[k,:]+gauge_centraliser[i,:]
                    C[k,:]=C[k,:]+C[i,:]
                    C[:,k]=C[:,k]+C[:,i]
            marked[j]=1
        marked[i]=1
    return {"X": galois.GF(2)(pairs1), "Z": galois.GF(2)(pairs2)}


def nkd_params(hs, check_x_and_z=True, separate_x_and_z=False, verbose = False, quiet="set_by_verbose", cutoff = np.inf):
    """
    Returns the number of physical qubits (n), number of logical qubits (k), and distance (d) of a CSS code from its X- and Z-parity check matrices.

    Arguments:
        hs is either:
            a dictionary with keys "X" and "Z", and items that are numpy arrays corresponding to the X- and Z-parity check matrices, respectively; or
            a numpy array corresponding to the parity check matrix. If the code is CSS then this will have block form [[Hx 0] [0 Hz]], but there is
                no requirement that the code be CSS for this to work.
        check_x_and_z is a bool. If True, then both the X- and Z-distance is calculated. If False, only the X-distance is calculated: this setting
            is useful if the code is guaranteed to have X-distance equal to Z-distance.
        separate_x_and_z is a bool. If True, then the distance parameter contains both the X- and the Z-distance separately; if False then the distance is a
            single integer corresponding to the distance = min(X-distance, Z-distance).
        verbose is a bool, if False then the text output from the Gurobi solver is suppressed.
        quiet is the opposite of verbose, kept for backwards compatibility.
        cutoff is an Int (or np.inf) that provides an upper-bound to the returned distance. If the solver is able to lower-bound the distance to be d >= cutoff,
            the solver will stop and the function will simply return cutoff.

    Returns:
        If separate_x_and_z = False, or if hs is a combined X and Z parity check matrix, a length-3 tuple of Ints (n,k,d).
        Otherwise, the same as above except d is a dictionary with keys "X" and "Z" and items that are integers representing the X-
            and Z-distance respectively.
    """
    if quiet != "set_by_verbose":
        verbose = not quiet

    if type(hs) == dict:
        n_qubits = hs["X"].shape[1]
        if 0 in hs["X"].shape:
            n_qubits = hs["Z"].shape[1]
        n_logicals = (
            n_qubits - np.linalg.matrix_rank(hs["X"]) - np.linalg.matrix_rank(hs["Z"])
        )
        if n_logicals <= 10**(-10):
            return (n_qubits, n_logicals, 0)
        return (
            n_qubits,
            n_logicals,
            distance(
                hs,
                check_x_and_z=check_x_and_z,
                separate_x_and_z=separate_x_and_z,
                verbose=verbose,
                cutoff = cutoff
            ),
        )
    else:
        h = hs
        n_qubits = h.shape[1] // 2
        logical_basis = get_symplectic_logical_basis(h)
        n_logicals = logical_basis["X"].shape[0]
        if n_logicals <= 10**(-10):
            return (n_qubits, n_logicals, 0)
        return (
            n_qubits,
            n_logicals,
            non_CSS_distance(
                h,
                logical_basis,
                verbose = verbose,
                cutoff = cutoff,
            ),
        )
