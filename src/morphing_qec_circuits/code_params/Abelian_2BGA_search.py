from .Abelian_2BGA_params import Abelian_2BGA_iterator, self_dual_Abelian_2BGA_iterator
from copy import deepcopy

def Abelian_2BGA_search(
    size_G_list,
    lattice_signature_list,
    parameter_restrictions = {"num_logicals": lambda k: k > 0},
    parameter_minimisations = ["num_qubits", "weight"],
    parameter_maximisations = ["num_logicals", "distance"],
    keep_equally_performant_codes = True,
):
    """
    Searches over Abelian 2BGA codes. The codes can
        be filtered, minimised and/or maximised based on a set of chosen parameter values. Allowed parameter values are
            num_qubits: the total number of qubits in the morphing circuit (so, the number of mid-cycle qubits, or 2 * the number of end-cycle qubits)
            num_logicals: the number of logical qubits
            distance: the distance of the code. This cannot be minimised.
            weight: the weight of the stabilisers of the code.

    Arguments:
        size_G_list is a list of integers, with each integer representing the size of the group defining the mid-cycle Abelian 2BGA code. The number of
            qubits in each mid-cycle code will be 2 * size_G, and the number of qubits in each end-cycle code will be size_G.
        lattice_signature_list is a list of lattice_signatures to search over when defining the mid-cycle Abelian 2BGA code.
        parameter_restrictions is a dictionary, with keys corresponding to allowed parameters and items being functions that return True if the parameter
            is in the allowed range, and False otherwise. The search will only return morphing circuits that satisfy all parameter restrictions.
        parameter_minimisations is a list containing parameters to be minimised.
        parameter_maximisations is a list containing parameters to be maximised.

    Raises:
        ValueError: distance cannot be minimised. It can be maximised, restricted, or ignored.
        ValueError: parameter_restrictions has an unknown parameter
        ValueError: parameter_minimisations has an unknown parameter
        ValueError: parameter_maximisations has an unknown parameter
        
    Returns:
        A dictionary containing the following items:
            "best_codes": a list of the best Abelian 2BGA codes,
            "best_performance_parameters": a list (in the same order as best_codes) of the corresponding performance parameters of the codes.
    """
    best_codes = []
    best_performance_parameters = []

    if "distance" in parameter_minimisations:
        raise ValueError("distance cannot be minimised. It can be maximised, restricted, or ignored.")
    
    parameter_keys = [
        "num_qubits",
        "num_logicals",
        "distance",
        "weight",
    ]

    check_distance = "distance" in (list(parameter_restrictions.keys()) + parameter_minimisations + parameter_maximisations)
    
    for key in parameter_restrictions.keys():
        if not key in parameter_keys:
            raise ValueError(f"parameter_restrictions has an unknown parameter {key}")
    for key in parameter_keys:
        if not key in parameter_restrictions.keys():
            parameter_restrictions[key] = lambda x: True

    def parameter_comparison(parameters_0, parameters_1):
        if all([parameters_0[key] >= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] <= parameters_1[key] for key in parameter_maximisations]):
            return 1
        elif all([parameters_0[key] <= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] >= parameters_1[key] for key in parameter_maximisations]):
            return 0
        else:
            return None

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

    for lattice_signature in lattice_signature_list:
        for size_G in size_G_list:
            for this_code in Abelian_2BGA_iterator(size_G, lattice_signature):
                this_nk_params = this_code.nk_params()
                these_performance_parameters = {}
                these_performance_parameters["num_qubits"] = this_nk_params[0]
                these_performance_parameters["num_logicals"] = this_nk_params[1]
                these_performance_parameters["weight"] = sum(lattice_signature) + 2
                if not parameter_restrictions["num_qubits"](these_performance_parameters["num_qubits"]):
                    continue
                if not parameter_restrictions["num_logicals"](these_performance_parameters["num_logicals"]):
                    continue
                if check_distance:
                    #checking whether this code has interesting parameters
                    nkd_params = this_code.nkd_params()
                    these_performance_parameters["distance"] = nkd_params[2]
                    if not parameter_restrictions["distance"](these_performance_parameters["distance"]):
                        continue
                update_best_parameters(
                    this_code,
                    these_performance_parameters,
                    best_codes,
                    best_performance_parameters,
                    parameter_comparison,
                    True
                )
    
    return {
        "best_codes": best_codes,
        "best_performance_parameters": best_performance_parameters,
    }

def self_dual_Abelian_2BGA_search(
    size_G_list,
    lattice_signature_list,
    parameter_restrictions = {"num_logicals": lambda k: k > 0},
    parameter_minimisations = ["num_qubits", "weight"],
    parameter_maximisations = ["num_logicals", "distance"],
    keep_equally_performant_codes = True,
):
    """
    Searches over self-dual Abelian 2BGA codes. The codes can
        be filtered, minimised and/or maximised based on a set of chosen parameter values. Allowed parameter values are
            num_qubits: the total number of qubits in the morphing circuit (so, the number of mid-cycle qubits, or 2 * the number of end-cycle qubits)
            num_logicals: the number of logical qubits
            distance: the distance of the code. This cannot be minimised.
            weight: the weight of the stabilisers of the code.

    Arguments:
        size_G_list is a list of integers, with each integer representing the size of the group defining the mid-cycle Abelian 2BGA code. The number of
            qubits in each mid-cycle code will be 2 * size_G, and the number of qubits in each end-cycle code will be size_G.
        lattice_signature_list is a list of lattice_signatures to search over when defining the mid-cycle Abelian 2BGA code.
        parameter_restrictions is a dictionary, with keys corresponding to allowed parameters and items being functions that return True if the parameter
            is in the allowed range, and False otherwise. The search will only return morphing circuits that satisfy all parameter restrictions.
        parameter_minimisations is a list containing parameters to be minimised.
        parameter_maximisations is a list containing parameters to be maximised.

    Raises:
        ValueError: distance cannot be minimised. It can be maximised, restricted, or ignored.
        ValueError: parameter_restrictions has an unknown parameter
        ValueError: parameter_minimisations has an unknown parameter
        ValueError: parameter_maximisations has an unknown parameter
        
    Returns:
        A dictionary containing the following items:
            "best_codes": a list of the best Abelian 2BGA codes,
            "best_performance_parameters": a list (in the same order as best_codes) of the corresponding performance parameters of the codes.
    """
    best_codes = []
    best_performance_parameters = []

    if "distance" in parameter_minimisations:
        raise ValueError("distance cannot be minimised. It can be maximised, restricted, or ignored.")
    
    parameter_keys = [
        "num_qubits",
        "num_logicals",
        "distance",
        "weight",
    ]

    check_distance = "distance" in (list(parameter_restrictions.keys()) + parameter_minimisations + parameter_maximisations)
    
    for key in parameter_restrictions.keys():
        if not key in parameter_keys:
            raise ValueError(f"parameter_restrictions has an unknown parameter {key}")
    for key in parameter_keys:
        if not key in parameter_restrictions.keys():
            parameter_restrictions[key] = lambda x: True

    def parameter_comparison(parameters_0, parameters_1):
        if all([parameters_0[key] >= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] <= parameters_1[key] for key in parameter_maximisations]):
            return 1
        elif all([parameters_0[key] <= parameters_1[key] for key in parameter_minimisations]) and all([parameters_0[key] >= parameters_1[key] for key in parameter_maximisations]):
            return 0
        else:
            return None

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

    for lattice_signature in lattice_signature_list:
        for size_G in size_G_list:
            for this_code in self_dual_Abelian_2BGA_iterator(size_G, lattice_signature):
                this_nk_params = this_code.nk_params()
                these_performance_parameters = {}
                these_performance_parameters["num_qubits"] = this_nk_params[0]
                these_performance_parameters["num_logicals"] = this_nk_params[1]
                these_performance_parameters["weight"] = sum(lattice_signature) + 2
                if not parameter_restrictions["num_qubits"](these_performance_parameters["num_qubits"]):
                    continue
                if not parameter_restrictions["num_logicals"](these_performance_parameters["num_logicals"]):
                    continue
                if check_distance:
                    #checking whether this code has interesting parameters
                    nkd_params = this_code.nkd_params()
                    these_performance_parameters["distance"] = nkd_params[2]
                    if not parameter_restrictions["distance"](these_performance_parameters["distance"]):
                        continue
                update_best_parameters(
                    this_code,
                    these_performance_parameters,
                    best_codes,
                    best_performance_parameters,
                    parameter_comparison,
                    True
                )
    
    return {
        "best_codes": best_codes,
        "best_performance_parameters": best_performance_parameters,
    }