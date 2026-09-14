from ..code_params import Abelian_2BGA_iterator
from ..misc import circuit_distance, contraction_tree_iterator
from copy import deepcopy
from collections import defaultdict
from itertools import permutations

def Abelian_2BGA_two_round_morphing_search(
    size_G_list,
    lattice_signature_list,
    parameter_restrictions = {"num_logicals": lambda k: k > 0},
    parameter_minimisations = ["num_qubits", "depth", "connectivity", "gate_count"],
    parameter_maximisations = ["num_logicals", "circuit_distance", "mid_cycle_distance", "swap_ancillas"],
    check_mid_cycle_distance = True,
    keep_equally_performant_circuits = True,
):
    """
    Searches over two-round purely contracting homomorphism-based mophing circuits for Abelian 2BGA codes, as defined in arXiv:2604.09797. The circuits can
        be filtered, minimised and/or maximised based on a set of chosen parameter values. Allowed parameter values are
            num_qubits: the total number of qubits in the morphing circuit (so, the number of mid-cycle qubits, or 2 * the number of end-cycle qubits)
            num_logicals: the number of logical qubits
            mid_cycle_distance: the distance of the mid-cycle code. This cannot be minimised.
            end_cycle_distance: the distance of the end-cycle code. This cannot be minimised.
            circuit_distance: the circuit-level distance of the morphing circuit. This cannot be minimised.
                If this parameter is used, then the end-cycle distance is automatically calculated before attempting to calculate the cicuit-level distance.
            depth: the depth of each contraction circuit.
            connectivity: the connectivity degree of each qubit in the morphing circuit.
            gate_count: the number of CNOT gates that each qubit participates in across both contraction circuits.
            swap_ancillas: if the order of the last CNOT gate in the contraction circuit is swapped, does this then interchange the end-cycle data and
                end-cycle ancilla qubits? True (1) if so, False (0) if not. This property is useful for leakage reduction.

    Arguments:
        size_G_list is a list of integers, with each integer representing the size of the group defining the mid-cycle Abelian 2BGA code. The number of
            qubits in each mid-cycle code will be 2 * size_G, and the number of qubits in each end-cycle code will be size_G.
        lattice_signature_list is a list of lattice_signatures to search over when defining the mid-cycle Abelian 2BGA code.
        parameter_restrictions is a dictionary, with keys corresponding to allowed parameters and items being functions that return True if the parameter
            is in the allowed range, and False otherwise. The search will only return morphing circuits that satisfy all parameter restrictions.
        parameter_minimisations is a list containing parameters to be minimised.
        parameter_maximisations is a list containing parameters to be maximised.
        check_mid_cycle_distance is a bool. When set to True, the mid-cycle distance is calculated before continuing. This may be desired if one is
            also interested in finding the best mid-cycle codes, independent of the morphing circuits. It is also used to reduce the number of morphing
            circuits tested - if the mid-cycle distance is lower than the best end-cycle or circuit distance found. Whether or not this speeds up the
            search overall is not clear, since the mid-cycle codes typically have a higher distance than the end-cycle codes.
        keep_equally_performant_circuits is a bool. When set to True, the search returns all of the best performing morphing circuits; when set to False,
            it only returns one.

    Raises:
        ValueError: mid_cycle_distance cannot be minimised. It can be maximised, restricted, or ignored.
        ValueError: end_cycle_distance cannot be minimised. It can be maximised, restricted, or ignored.
        ValueError: circuit_distance cannot be minimised. It can be maximised, restricted, or ignored.
        ValueError: parameter_restrictions has an unknown parameter
        ValueError: parameter_minimisations has an unknown parameter
        ValueError: parameter_maximisations has an unknown parameter
        
    Returns:
        A dictionary containing the following items:
            "best_mid_cycle_codes": a list of the best mid-cycle Abelian 2BGA codes,
            "best_mid_cycle_performance_parameters": a list (in the same order as best_mid_cycle_codes) of the corresponding performance parameters of the mid-cycle codes,
            "best_morphing_circuits": a list of the best morphing circuits, specified by its mid-cycle code, contraction tree, homomorphism image, homomorphism action, and b_shift,
            "best_morphing_performance_parameters": a list (in the same order as best_morphing_circuits) of the corresponding performance parameters of the morphing circuit.
    """
    best_mid_cycle_codes = []
    best_mid_cycle_performance_parameters = []
    best_morphing_circuits = []
    best_morphing_performance_parameters = []
    all_end_cycle_codes = defaultdict(list)
    all_end_cycle_performance_parameters = defaultdict(list)

    if "mid_cycle_distance" in parameter_minimisations:
        raise ValueError("mid_cycle_distance cannot be minimised. It can be maximised, restricted, or ignored.")
    if "end_cycle_distance" in parameter_minimisations:
        raise ValueError("end_cycle_distance cannot be minimised. It can be maximised, restricted, or ignored.")
    if "circuit_distance" in parameter_minimisations:
        raise ValueError("circuit_distance cannot be minimised. It can be maximised, restricted, or ignored.")

    parameter_keys = [
        "num_qubits",
        "num_logicals",
        "mid_cycle_distance",
        "end_cycle_distance",
        "circuit_distance",
        "depth",
        "connectivity",
        "gate_count",
        "swap_ancillas",
    ]
    mid_cycle_parameter_keys = ["num_qubits", "num_logicals", "mid_cycle_distance"]

    check_circuit_distance = "circuit_distance" in (list(parameter_restrictions.keys()) + parameter_minimisations + parameter_maximisations)
    check_end_cycle_distance = "end_cycle_distance" in (list(parameter_restrictions.keys()) + parameter_minimisations + parameter_maximisations)
    check_end_cycle_distance = check_end_cycle_distance or check_circuit_distance
    
    for key in parameter_restrictions.keys():
        if not key in parameter_keys:
            raise ValueError(f"parameter_restrictions has an unknown parameter {key}")
    for key in parameter_keys:
        if not key in parameter_restrictions.keys():
            parameter_restrictions[key] = lambda x: True


    mid_cycle_parameter_minimisations = []
    for parameter in parameter_minimisations:
        if not parameter in parameter_keys:
            raise ValueError(f"parameter_minimisations has an unknown parameter {parameter}")
        if parameter in mid_cycle_parameter_keys:
            mid_cycle_parameter_minimisations.append(parameter)

    mid_cycle_parameter_maximisations = []
    for parameter in parameter_maximisations:
        if not parameter in parameter_keys:
            raise ValueError(f"parameter_maximisations has an unknown parameter {parameter}")
        if parameter in mid_cycle_parameter_keys:
            mid_cycle_parameter_maximisations.append(parameter)
    
    if check_mid_cycle_distance and not "mid_cycle_distance" in mid_cycle_parameter_maximisations:
        mid_cycle_parameter_maximisations.append("mid_cycle_distance")

    if "circuit_distance" in parameter_maximisations:
        if "end_cycle_distance" in parameter_maximisations:
            parameter_maximisations.remove("end_cycle_distance")
        if "mid_cycle_distance" in parameter_maximisations:
            parameter_maximisations.remove("mid_cycle_distance")

    if "end_cycle_distance" in parameter_maximisations:
        if "mid_cycle_distance" in parameter_maximisations:
            parameter_maximisations.remove("mid_cycle_distance")

    

    def mid_cycle_comparison(parameters_0, parameters_1):
        if all([parameters_0[key] >= parameters_1[key] for key in mid_cycle_parameter_minimisations]) and all([parameters_0[key] <= parameters_1[key] for key in mid_cycle_parameter_maximisations]):
            return 1
        elif all([parameters_0[key] <= parameters_1[key] for key in mid_cycle_parameter_minimisations]) and all([parameters_0[key] >= parameters_1[key] for key in mid_cycle_parameter_maximisations]):
            return 0
        else:
            return None

    def morphing_comparison(parameters_0, parameters_1):
        if all([parameters_0[key] >= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] <= parameters_1[key] for key in parameter_maximisations]):
            return 1
        elif all([parameters_0[key] <= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] >= parameters_1[key] for key in parameter_maximisations]):
            return 0
        else:
            return None

    def possibly_best_parameters(this_object, these_parameters, previous_best_objects, previous_best_parameters, comparison_function, keep_equally_performant_parameters):
        for those_parameters in previous_best_parameters:
            if not keep_equally_performant_parameters:
                if all([these_parameters[key] == those_parameters[key] for key in (parameter_minimisations + parameter_maximisations)]):
                    return False
            comparison = comparison_function(these_parameters, those_parameters)
            if comparison == 1:
                return False
        return True

    def update_best_parameters(this_object, these_parameters, previous_best_objects, previous_best_parameters, comparison_function, keep_equally_performant_parameters):
        temp = deepcopy(previous_best_parameters)
        for those_parameters in temp:
            if not keep_equally_performant_parameters:
                if all([these_parameters[key] == those_parameters[key] for key in (parameter_minimisations + parameter_maximisations)]):
                    break
            comparison = comparison_function(these_parameters, those_parameters)
            if comparison == 1:
                break
            elif comparison == 0:
                previous_best_objects.pop(previous_best_parameters.index(those_parameters))
                previous_best_parameters.remove(those_parameters)
        else:
            previous_best_objects.append(deepcopy(this_object))
            previous_best_parameters.append(deepcopy(these_parameters))

    for size_G in size_G_list:
        for lattice_signature in lattice_signature_list:
            for this_mid_cycle_code in Abelian_2BGA_iterator(size_G, lattice_signature):
                this_mid_cycle_nk_params = this_mid_cycle_code.nk_params()
                these_performance_parameters = {}
                these_performance_parameters["num_qubits"] = this_mid_cycle_nk_params[0]
                these_performance_parameters["num_logicals"] = this_mid_cycle_nk_params[1]
                if not parameter_restrictions["num_qubits"](these_performance_parameters["num_qubits"]):
                    continue
                if not parameter_restrictions["num_logicals"](these_performance_parameters["num_logicals"]):
                    continue
                if check_mid_cycle_distance:
                    #checking whether this mid-cycle code has interesting parameters
                    mid_cycle_nkd_params = this_mid_cycle_code.nkd_params()
                    these_performance_parameters["mid_cycle_distance"] = mid_cycle_nkd_params[2]
                    if not parameter_restrictions["mid_cycle_distance"](these_performance_parameters["mid_cycle_distance"]):
                        continue
                update_best_parameters(
                    this_mid_cycle_code,
                    these_performance_parameters,
                    best_mid_cycle_codes,
                    best_mid_cycle_performance_parameters,
                    mid_cycle_comparison,
                    True
                )
    
                #the mid-cycle distance bounds the end-cycle and the circuit distance. Checking that the bound given by the
                #  mid-cycle distance satisfies the restrictions on the end-cycle and circuit distance.
                if check_mid_cycle_distance:
                    for possible_distance in range(these_performance_parameters["mid_cycle_distance"]):
                        if parameter_restrictions["end_cycle_distance"](possible_distance):
                            break
                    else:
                        continue
                    
                    for possible_distance in range(these_performance_parameters["mid_cycle_distance"]):
                        if parameter_restrictions["circuit_distance"](possible_distance):
                            break
                    else:
                        continue
                
                labels_list = [["A", "B"][i] + str(j) for i in range(2) for j in range(len([this_mid_cycle_code.a_list, this_mid_cycle_code.b_list][i]))]
                for homomorphism_action in this_mid_cycle_code.homomorphism_action_iterator([2]):
                    for b_shift_int in range(2):
                        b_shift = [b_shift_int]
                        for contraction_tree in contraction_tree_iterator(len(labels_list)):
                            for permutation in permutations(labels_list):
                                permuted_contraction_tree = [
                                    [
                                        [
                                            permutation[contraction_tree[i][j][k]] for k in range(2)
                                        ] for j in range(len(contraction_tree[i]))
                                    ] for i in range(len(contraction_tree))
                                ]
                                morphing_parameters = (
                                    permuted_contraction_tree,
                                    [2],
                                    homomorphism_action,
                                    b_shift,
                                )
                                if this_mid_cycle_code.valid_morphing_parameters(
                                    *morphing_parameters,
                                    method = "graph",
                                ):
                                    morphing_specifications = this_mid_cycle_code.get_morphing_specifications(
                                        *morphing_parameters,
                                        safe = False,
                                    )
                                    morphing_performance_parameters = this_mid_cycle_code.morphing_performance_parameters(
                                        *morphing_parameters,
                                    )
                                    these_performance_parameters["depth"] = morphing_performance_parameters[1]
                                    these_performance_parameters["connectivity"] = morphing_performance_parameters[2]
                                    these_performance_parameters["gate_count"] = morphing_performance_parameters[3]
                                    these_performance_parameters["swap_ancillas"] = morphing_performance_parameters[4]
                                    #temporarily set an upper-bound on the end-cycle and circuit distances for now, which will be replaced later.
                                    these_performance_parameters["end_cycle_distance"] = these_performance_parameters["mid_cycle_distance"]
                                    these_performance_parameters["circuit_distance"] = these_performance_parameters["mid_cycle_distance"]
                                    if not parameter_restrictions["depth"](these_performance_parameters["depth"]):
                                        continue
                                    if not parameter_restrictions["connectivity"](these_performance_parameters["connectivity"]):
                                        continue
                                    if not parameter_restrictions["gate_count"](these_performance_parameters["gate_count"]):
                                        continue
                                    if not parameter_restrictions["swap_ancillas"](these_performance_parameters["swap_ancillas"]):
                                        continue
                                    morphing_circuit = {
                                        "mid_cycle_code": this_mid_cycle_code,
                                        "contraction_tree": permuted_contraction_tree,
                                        "homomorphism_image": [2],
                                        "homomorphism_action": homomorphism_action,
                                        "b_shift": b_shift,
                                    }
                                    keep_going = possibly_best_parameters(
                                        morphing_circuit,
                                        these_performance_parameters,
                                        best_morphing_circuits,
                                        best_morphing_performance_parameters,
                                        morphing_comparison,
                                        keep_equally_performant_circuits,
                                    )
                                    these_performance_parameters["end_cycle_distance"] = None
                                    these_performance_parameters["circuit_distance"] = None
    
                                    if not keep_going:
                                        continue
                                    
                                    if check_end_cycle_distance:
                                        this_end_cycle_code = this_mid_cycle_code.two_round_end_cycle_code(
                                            morphing_parameters[0],
                                            morphing_parameters[2],
                                            morphing_parameters[3],
                                        )
                                        lattice_signature_tuple = tuple(this_end_cycle_code.lattice_signature)
                                        periodicity_vectors = this_end_cycle_code.periodicity_vectors
                                        if len(periodicity_vectors) == 1:
                                            periodicity_vectors_tuple = tuple(periodicity_vectors)
                                        else:
                                            periodicity_vectors_tuple = tuple(map(tuple, this_end_cycle_code.periodicity_vectors))
                                        if periodicity_vectors_tuple in all_end_cycle_codes[lattice_signature_tuple]:
                                            end_cycle_nkd = all_end_cycle_performance_parameters[lattice_signature_tuple][
                                                all_end_cycle_codes[lattice_signature_tuple].index(periodicity_vectors_tuple)
                                            ]
                                            these_performance_parameters["end_cycle_distance"] = end_cycle_nkd[2]
                                        else:
                                            if check_mid_cycle_distance:
                                                end_cycle_nkd = this_end_cycle_code.nkd_params(cutoff = these_performance_parameters["mid_cycle_distance"])
                                            else:
                                                end_cycle_nkd = this_end_cycle_code.nkd_params()
                                            these_performance_parameters["end_cycle_distance"] = end_cycle_nkd[2]
                                            for equiv in this_end_cycle_code.list_equivalent_params(swap_AB = True):
                                                equiv_lattice_signature_tuple = tuple(equiv.lattice_signature)
                                                equiv_periodicity_vectors_tuple = tuple(map(tuple, equiv.periodicity_vectors))
                                                all_end_cycle_codes[equiv_lattice_signature_tuple].append(equiv_periodicity_vectors_tuple)
                                                all_end_cycle_performance_parameters[equiv_lattice_signature_tuple].append(end_cycle_nkd)
                                        
                                        if not parameter_restrictions["end_cycle_distance"](these_performance_parameters["end_cycle_distance"]):
                                            continue
                    
                                        for possible_distance in range(these_performance_parameters["end_cycle_distance"]):
                                            if parameter_restrictions["circuit_distance"](possible_distance):
                                                break
                                        else:
                                            continue
    
                                        these_performance_parameters["circuit_distance"] = these_performance_parameters["end_cycle_distance"]
                                    
                                        keep_going = possibly_best_parameters(
                                            morphing_circuit,
                                            these_performance_parameters,
                                            best_morphing_circuits,
                                            best_morphing_performance_parameters,
                                            morphing_comparison,
                                            keep_equally_performant_circuits,
                                        )
                                        these_performance_parameters["circuit_distance"] = None
    
                                        if not keep_going:
                                            continue
                                    
                                    if check_circuit_distance:
                                        if check_end_cycle_distance:
                                            cutoff = these_performance_parameters["end_cycle_distance"]
                                        elif check_mid_cycle_distance:
                                            cutoff = these_performance_parameters["mid_cycle_distance"]
                                        else:
                                            cutoff = np.inf
                                        these_performance_parameters["circuit_distance"] = morphing_specifications.two_round_circuit_distance(cutoff = cutoff)
                                    update_best_parameters(
                                        morphing_circuit,
                                        these_performance_parameters,
                                        best_morphing_circuits,
                                        best_morphing_performance_parameters,
                                        morphing_comparison,
                                        keep_equally_performant_circuits,
                                    )
    return {
        "best_mid_cycle_codes": best_mid_cycle_codes,
        "best_mid_cycle_performance_parameters": best_mid_cycle_performance_parameters,
        "best_morphing_circuits": best_morphing_circuits,
        "best_morphing_performance_parameters": best_morphing_performance_parameters,
    }