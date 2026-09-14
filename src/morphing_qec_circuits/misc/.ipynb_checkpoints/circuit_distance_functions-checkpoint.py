import numpy as np
import stim
import random
from collections import Counter
from mle_decoder import MLEDecoder
import gurobipy as gp
from gurobipy import GRB

def _include_observable_as_detector(dem, observables_to_include):
    new_detector_index = dem.num_detectors
    new_dem = stim.DetectorErrorModel()
    for deminstruction in dem:
        if deminstruction.type == "error":
            observable_flips = [
                targ.val
                for targ in deminstruction.targets_copy()
                if targ.is_logical_observable_id()
            ]
            to_include = False
            for observable_flip in observable_flips:
                if observables_to_include[observable_flip]:
                    to_include = not to_include
            new_targets = [
                targ
                for targ in deminstruction.targets_copy()
                if targ.is_relative_detector_id()
            ]
            if to_include:
                new_targets.append(stim.target_relative_detector_id(new_detector_index))
            new_dem.append(
                stim.DemInstruction(
                    deminstruction.type, deminstruction.args_copy(), new_targets
                )
            )
        else:
            new_dem.append(deminstruction)
    return new_dem

def circuit_distance(circuit, error_basis = "both", basis = None, return_corrections = False, cutoff = np.inf, verbose = False):
    """
    Returns the circuit distance of the circuit, calculated using Gurobi.

    Arguments:
        circuit is a stim.Circuit with observables and detectors, of which the circuit distance is to be calculated.
        error_basis is a Str "both", "X", or "Z", that determines the basis of errors to look for. In particular, if there are 2*k total observables in the
            circuit, the first k of which are X observables and the other k are Z observables, then setting error_basis = "X" will only look for errors that
            flip one of the Z observables (in other words, it calculates the X-type circuit distance). If the code is not CSS, or the observables do not have
            this structure, you should always set error_basis = "both".
        return_corrections is a bool, when set to False the function returns only the distance, when set to True the function returns a tuple containing
            the distance, the stim.DetectorErrorModel used for the distance calculation, and the rows of the DEM that constitute a minimum-weight logical error.
        cutoff is an Int (or np.inf) that provides an upper-bound to the returned distance. If the solver is able to lower-bound the distance to be d >= cutoff,
            the solver will stop and the function will simply return cutoff.
        verbose is a bool that when set to False suppresses the text output of the Gurobi solver.

    Raises:
        ValueError: error_basis must be X, Z, or both

    Returns:
        If return_corrections = False, returns an integer that is equal to the circuit-distance of the circuit, or the cutoff, whichever is lower.
        If return_corrections = True, returns a tuple with the distance, detector error model, and minimum-weight logical error.
    """
    initialise_decoder = lambda dem: MLEDecoder(dem)
    dem = circuit.detector_error_model().flattened()
    if error_basis == "both":
        k = dem.num_observables
        observable_basis = "both"
    elif error_basis in ["X", "Z"]:
        k = dem.num_observables // 2
        observable_basis = {"X": "Z", "Z": "X"}[error_basis]
    else:
        raise ValueError("error_basis must be X, Z, or both")
    n_detectors = dem.num_detectors

    dist = cutoff
    
    h_matrix = []
    logicals = []
    minimum_error_correction = stim.DetectorErrorModel()
    dem_errors_only = stim.DetectorErrorModel().flattened()
    for dem_instruction in dem:
        if dem_instruction.type == "error":
            dem_errors_only.append(dem_instruction)
            h_matrix_to_append = [0 for _ in range(n_detectors)]
            logicals_to_append = [0 for _ in range(k)]
            for dem_target in dem_instruction.targets_copy():
                if dem_target.is_relative_detector_id():
                    h_matrix_to_append[dem_target.val] = 1
                elif dem_target.is_logical_observable_id():
                    if observable_basis == "X" and dem_target.val >= k:
                        continue
                    elif observable_basis == "Z" and dem_target.val < k:
                        continue

                    if observable_basis == "Z":
                        logicals_to_append[dem_target.val - k] = 1
                    else:
                        logicals_to_append[dem_target.val] = 1
            h_matrix.append(h_matrix_to_append)
            logicals.append(logicals_to_append)
    h_matrix = np.array(h_matrix, dtype = np.int32).T
    logicals = np.array(logicals, dtype = np.int32).T
    zero_array = np.zeros((h_matrix.shape[0]), dtype = np.int32)
    
    for i in range(logicals.shape[0]):
        m = gp.Model()
        if not verbose:
            m.Params.OutputFlag = 0
            m.Params.LogToConsole = 0
        var = m.addMVar(shape=h_matrix.shape[1], vtype=GRB.BINARY, name="var")
        if not (0 in h_matrix.shape):
            dummy1 = m.addMVar(shape=h_matrix.shape[0], vtype=GRB.INTEGER, name="dummy1")
            stab_constraint = m.addConstr(
                h_matrix @ var - 2 * dummy1 == zero_array, "detector_constraint"
            )
        dummy2 = m.addVar(vtype=GRB.INTEGER, name="dummy2")
        logical_constraint = m.addConstr(
            logicals[i] @ var - 2 * dummy2 == 1, "logical_constraint"
        )
        m.setObjective(var.sum(), GRB.MINIMIZE)
        if dist < np.inf:
            m.Params.Cutoff = dist
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            if m.ObjVal < dist:
                dist = m.ObjVal
                if return_corrections:
                    logical_fault_prediction = var.X
                    error_correction = stim.DetectorErrorModel()
                    for j, x in enumerate(logical_fault_prediction):
                         if x == 1:
                             error_correction.append(dem_errors_only[j])
                    minimum_error_correction = error_correction
        elif m.Status in [GRB.CUTOFF, GRB.INFEASIBLE]:
            pass
        else:
            raise Exception("Something went wrong, gurobi did not reach an optimum nor did it get cut off...")
    
    if return_corrections:
        return round(dist), dem_errors_only, minimum_error_correction
    else:
        return round(dist)
    

def circuit_distance_ub(
    circuit, initialise_decoder, error_basis="both", n_shots=None, return_statistics=False, return_corrections=False
):
    """
    Returns an upper-bound to the circuit distance of the circuit, calculated using a decoder.

    For example, one can use this function to exactly calculate the distance if one uses a maximum-likelihood decoder:
        initialise_decoder = lambda dem: MLEDecoder(dem)
        n_shots = "linear"
        distance = circuit_distance_ub(circuit, initialise_decoder, n_shots = n_shots)
    The returned distance here will be equal to the output of circuit_distance(circuit).

    Arguments:
        circuit is a stim.Circuit with observables and detectors, of which the circuit distance is to be calculated.
        initialise_decoder is a function that takes a stim.DetectorErrorModel as input, and returns a decoding object that can be used to decode the dem.
        error_basis is a Str "both", "X", or "Z", that determines the basis of errors to look for. In particular, if there are 2*k total observables in the
            circuit, the first k of which are X observables and the other k are Z observables, then setting error_basis = "X" will only look for errors that
            flip one of the Z observables (in other words, it calculates the X-type circuit distance). If the code is not CSS, or the observables do not have
            this structure, you should always set error_basis = "both".
        n_shots is an Int that determines the number of observables that the decoder will decode. If n_shots = None, it is automatically set to the maximum
            number of non-trivial observables (2**k - 1), while if n_shots = "linear" then the observables are chosen in such a way that it is guaranteed
            that the true minimum-weight logical error anticommutes with at least one of the chosen observables.
        return_statistics is a bool, when set to False the function returns the distance upper-bound, when set to True the function returns a Counter object
            that contains the weights of all of the shots of the decoder.
        return_corrections is a bool, when set to False the function returns only the distance upper-bound (or, if return_statistics = True, the Counter object
            with the statistics of the calculated distance upper-bounds), when set to True the function returns a tuple containing
            the distance upper-bound/Counter object, the stim.DetectorErrorModel used for the distance calculation, and the rows of the DEM that constitute a
            minimum-weight logical error.
    
    Raises:
        ValueError: error_basis must be X, Z, or both

    Returns:
        If return_corrections = False, returns an integer that is an upper-bound to the circuit-distance of the circuit.
        If return_corrections = True, returns a tuple with the distance upper-bound, detector error model, and the corresponding logical error.
        If return_statistics = True, the distance upper-bound in the above is replaced by a Counter object that contains the number of times each
            distance upper-bound was obtained. Moreover, a list of logical errors is returned, one for each shot of the decoder.
    """
    dem = circuit.detector_error_model().flattened()
    if error_basis == "both":
        k = dem.num_observables
        observable_basis = "both"
    elif error_basis in ["X", "Z"]:
        k = dem.num_observables // 2
        observable_basis = {"X": "Z", "Z": "X"}[error_basis]
    else:
        raise Exception("error_basis must be X, Z, or both")
    n_detectors = dem.num_detectors

    random_shots = True
    if n_shots == None:
        n_shots = 2**k - 1
        random_shots = False
    elif n_shots == "linear":
        random_shots = False
        n_shots = k
    elif n_shots >= 2**k - 1:
        n_shots = 2**k - 1
        random_shots = False

    d_estimates = [0 for i in range(n_shots)]
    if return_corrections:
        error_corrections = []
        dem_errors_only = stim.DetectorErrorModel()
        for dem_instruction in dem:
            if dem_instruction.type == "error":
                dem_errors_only.append(dem_instruction)
    for i in range(1, n_shots + 1):
        if random_shots:
            rand_int = random.randrange(1, 2**k)
            observables_to_include = [bool(int(x)) for x in bin(rand_int)[2:].zfill(k)]
        elif n_shots == 2**k - 1:
            observables_to_include = [bool(int(x)) for x in bin(i)[2:].zfill(k)]
        elif n_shots == k:
            observables_to_include = [0 for _ in range(k)]
            observables_to_include[i-1] = 1
        else:
            raise Exception("something has gone horribly wrong")
        if observable_basis == "X":
            observables_to_include += [False] * k
        elif observable_basis == "Z":
            observables_to_include = ([False] * k) + observables_to_include
        new_dem = _include_observable_as_detector(dem, observables_to_include)
        decoder = initialise_decoder(new_dem)
        defects = np.array([False] * n_detectors + [True])
        logical_fault_prediction = decoder.decode_to_faults_array(defects)
        if return_corrections:
            error_correction = stim.DetectorErrorModel()
            for j, x in enumerate(logical_fault_prediction):
                 if x == 1:
                     error_correction.append(dem_errors_only[j])
            error_corrections.append(error_correction)
        d_estimates[i - 1] = int(sum(logical_fault_prediction))

    d_estimate = min(d_estimates)
    if return_statistics:
        to_output = Counter(d_estimates)
    else:
        to_output = d_estimate
        error_corrections = error_corrections[d_estimates.index(d_estimate)]

    if return_corrections:
        return to_output, dem, error_corrections
    else:
        return to_output
