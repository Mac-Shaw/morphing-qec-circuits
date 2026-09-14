import numpy as np
import stim
from copy import deepcopy
from timeit import default_timer as timer
from surface_sim.models import CircuitNoiseModel, NoiselessModel


def separate_XZ_dem_from_circuit(circuit, **dem_kwargs):
    """
    Returns an X-type detector error model and a Z-type detector error model from a circuit.

    Arguments:
        circuit is a stim.Circuit. Detectors should be labelled using "chromobius" convention, so that the 4th coordinate is 0, 1, or 2 for X-detectors
            and 3, 4, or 5 for Z-detectors. There must be 2*k observables, the first k of which are X-type observables and the second k of which are
            Z-type observables.

    Returns:
        A dictionary with keys "X" and "Z" and items that are the X-type and Z-type detector error models (respectively).
    """
    detector_X_indices = []
    detector_Z_indices = []
    n_observables = circuit.num_observables
    i = 0
    for instruction in circuit.flattened():
        if instruction.name in ["DETECTOR"]:
            if int(instruction.gate_args_copy()[3]) <= 2:
                detector_X_indices.append(i)
            else:
                detector_Z_indices.append(i)
            i += 1
    dem = circuit.detector_error_model().flattened()
    X_dem = stim.DetectorErrorModel(**dem_kwargs)
    Z_dem = stim.DetectorErrorModel(**dem_kwargs)
    for instruction in dem:
        if instruction.type == "error":
            args = instruction.args_copy()
            targs = instruction.targets_copy()
            X_detector_targs = [
                stim.target_relative_detector_id(detector_X_indices.index(targ.val))
                for targ in targs
                if (targ.is_relative_detector_id() and targ.val in detector_X_indices)
            ]
            Z_detector_targs = [
                stim.target_relative_detector_id(detector_Z_indices.index(targ.val))
                for targ in targs
                if (targ.is_relative_detector_id() and targ.val in detector_Z_indices)
            ]
            X_logical_targs = [
                targ
                for targ in targs
                if (targ.is_logical_observable_id() and targ.val < n_observables // 2)
            ]
            Z_logical_targs = [
                stim.target_logical_observable_id(targ.val - n_observables // 2)
                for targ in targs
                if (targ.is_logical_observable_id() and targ.val >= n_observables // 2)
            ]
            X_instruction = stim.DemInstruction(
                "error", args, X_detector_targs + X_logical_targs
            )
            Z_instruction = stim.DemInstruction(
                "error", args, Z_detector_targs + Z_logical_targs
            )
            X_dem.append(X_instruction)
            Z_dem.append(Z_instruction)
    return {"X": X_dem, "Z": Z_dem}, {"X": detector_X_indices, "Z": detector_Z_indices}


bp_osd_kwargs = {
    "bp_method": "minimum_sum",
    "osd_method": "osd_cs",
    "osd_order": 20,
    "max_iter": 10_000,
}


def sample_logical_flips(
    sampler,
    decoder,
    batch_size=0,
    target_flips=0,
    max_time=0,
    separate_XZ=False,
    XZ_detector_indices={},
):
    """
    Samples logical flips from a sampler using a decoder.

    Arguments:
        sampler is a stim.CompiledDemSampler to sample from.
        decoder is a compiled decoder with a decode_batch method that can be called to run the decoder.
        batch_size is an int that represents the number of shots to perform in a single batch before checking whether target_flips or max_time have been achieved.
        target_flips is an int. Once the sampler has at least target_flips logical flips, the sampler will stop.
        max_time is an int or float that represents the maximum amount of time spent decoding in seconds. Once max_time has been spent, the current batch of decoding will finish
            and then the sampler will stop.
        separate_XZ is a bool, if set to True then the X- and Z-sectors will be decoded separately according to XZ_detector_indices.
        XZ_detector_indices is a dictionary with keys "X" and "Z" and items that are each a list of the indices of the dem that correspond to X checks and Z checks respectively.
            This is only used if separate_XZ = True.

    Returns:
        A dictionary with keys "X" and "Z" and items that are the X-type and Z-type detector error models (respectively).
    """
    n_flips = 0
    n_shots = 0

    if separate_XZ:
        X_decoder = decoder["X"]
        Z_decoder = decoder["Z"]
        detector_X_indices = XZ_detector_indices["X"]
        detector_Z_indices = XZ_detector_indices["Z"]

    start = timer()
    while (n_flips < target_flips) and (timer() < start + max_time):
        defects, log_flips, errors_sampled = sampler.sample(shots=batch_size)#, separate_observables=True)
        n_observables = len(log_flips[0, :])
        if separate_XZ:
            X_predictions = X_decoder.decode_batch(defects[:, detector_X_indices])
            Z_predictions = Z_decoder.decode_batch(defects[:, detector_Z_indices])
            predictions = np.concatenate((X_predictions, Z_predictions), axis=1)
        else:
            predictions = decoder.decode_batch(defects)
        n_flips += np.sum(np.any(predictions != log_flips, axis=1))
        n_shots += batch_size
    return int(n_flips), n_shots


def observables_to_detectors(circuit):
    """
    Takes an input circuit, removes all of its detectors, and then replaces its observables with detectors that depend on the same set of measurements.

    Arguments:
        circuit is a stim.Circuit.

    Returns:
        A stim circuit whose detectors correspond to the observables of the input circuit.
    """
    new_circuit = stim.Circuit()
    n_observables = circuit.num_observables
    rec_targets = [[] for _ in range(n_observables)]
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            continue
        elif instruction.name == "OBSERVABLE_INCLUDE":
            arg = round(instruction.gate_args_copy()[0])
            targets = instruction.targets_copy()
            for target in targets:
                rec_targets[arg].append(target.value)
        elif instruction.name in ["M", "MX", "MY", "MR", "MRX", "MRY", "MPP"]:
            to_add = 1 if instruction.name == "MPP" else len(instruction.targets_copy())
            for i in range(len(rec_targets)):
                for j in range(len(rec_targets[i])):
                    rec_targets[i][j] += -to_add
            new_circuit.append(instruction)
        else:
            new_circuit.append(instruction)
    for i in range(n_observables):  # range should be equal to the number of observables
        new_circuit.append(
            "DETECTOR", [stim.target_rec(target) for target in rec_targets[i]]
        )
    return new_circuit


def filter_circuit_detectors(
    circuit, detector_list=None, coordinates_condition=lambda coordinates: True
):
    """
    Kept for backwards compatibility.
    """
    new_circuit = stim.Circuit()
    if detector_list == None:
        detector_list = range(circuit.num_detectors)
    i = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if i in detector_list and coordinates_condition(
                instruction.gate_args_copy()
            ):
                new_circuit.append(instruction)
                i += 1
            else:
                i += 1
        else:
            new_circuit.append(instruction)
    return new_circuit


def filter_dem_detectors(
    dem, detector_list=None, coordinates_condition=lambda coordinates: True
):
    """
    Kept for backwards compatibility.
    """
    declarations = stim.DetectorErrorModel()
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "detector":
            declarations.append(dem_instruction)
    if detector_list == None:
        detector_list = range(dem.num_detectors)
    new_detector_list = []
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "detector":
            if dem_instruction.targets_copy()[0].val in detector_list:
                if coordinates_condition(
                    np.array(dem_instruction.args_copy()).astype(int)
                ):
                    new_detector_list.append(dem_instruction.targets_copy()[0].val)
    new_dem = stim.DetectorErrorModel()
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "error":
            new_targs = []
            targs = dem_instruction.targets_copy()
            for targ in targs:
                if targ.is_logical_observable_id or targ.val in new_detector_list:
                    new_targs.append(targ)
            new_instruction = stim.DemInstruction(
                "error", dem_instruction.args_copy(), new_targs
            )
            new_dem.append(new_instruction)
        else:
            new_dem.append(dem_instruction)
    return new_dem


def filter_detectors(object, **kwargs):
    """
    Kept for backwards compatibility.
    """
    if type(object) == stim._stim_sse2.Circuit:
        return filter_circuit_detectors(object, **kwargs)
    elif type(object) == stim._stim_sse2.DetectorErrorModel:
        return filter_dem_detectors(object, **kwargs)

def qec_circuit_to_XZ_memory_experiment(
        qec_circuits,
        coordinate_circuit,
        detector_circuits,
        logical_operator_basis,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
            ),
        ):
    """
    A function that takes a list of syndrome extraction circuits and returns a memory experiment of depth T.
    The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of
        ideal QEC, with T rounds of noisy QEC in between.
    
    Arguments:
        qec_circuits is one of stim.Circuit, Str, List[stim.Circuit], or List[Str] representing the syndrome extraction circuits that will be applied.
        coordinate_circuit is either a stim.Circuit or Str that sets the coordinates of all the qubits in the syndrome extraction circuit.
        detector_circuits is one of stim.Circuit, Str, List[stim.Circuit], or List[Str] that corresponds to the detectors to be included after each
            corresponding QEC circuit.
        logical_operator_basis is a dictionary with keys "X" and "Z" and items that are List[stim.PauliString], representing a basis of logical operators
            at the end of the circuit. Despite the keys, the operators do not have to be physically of X or Z type, but all "X" logical operators must commute,
            and all "Z" logical operators must commute.
        noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
        T is the number of noisy QEC rounds in the memory experiment.
        noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model object
            that defines the noise model of the memory experiment.

    Raises:
        ValueError: Each element of qec_circuits must be str or stim.Circuit
        ValueError: coordinate_circuit must be str or stim.Circuit
        ValueError: Each element of detector_circuits must be str or stim.Circuit
        ValueError: T is too small for the mall, make him bigger
        ValueError: A qec_circuit has an instruction that I wasn't expecting!
    
    Returns:
        A stim circuit for the memory experiment.
    """
    if type(qec_circuits) != list:
        qec_circuits = [qec_circuits]
    if type(detector_circuits) != list:
        detector_circuits = [detector_circuits]
    n_qubits = []
    for qec_circuit in qec_circuits:
        if type(qec_circuit) == str:
            qec_circuit = stim.Circuit(qec_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"Each element of qec_circuits must be str or stim.Circuit, but is instead type {type(qec_circuit)}")
        n_qubits.append(qec_circuit.num_qubits)
    n_qubits = max(n_qubits)
    if type(coordinate_circuit) == str:
        coordinate_circuit = stim.Circuit(coordinate_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"coordinate_circuit must be str or stim.Circuit, but is instead type {type(coordinate_circuit)}")
    for detector_circuit in detector_circuits:
        if type(detector_circuit) == str:
            detector_circuit = stim.Circuit(detector_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"Each element of detector_circuits must be str or stim.Circuit, but is instead type {type(detector_circuits)}")
        
    if T < 1:
        raise ValueError("T is too small for the mall, make him bigger")
    
    n_logicals = len(logical_operator_basis["X"])

    qubit_dictionary = {str(i): i for i in range(n_qubits)}
    all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
    noise_model = noise_model_generator(qubit_dictionary, noise_setup)
    noiseless_model = NoiselessModel(all_qubits_dictionary)
    tick = noiseless_model.tick()
    
    noisy_qec_circuits = [stim.Circuit() for _ in range(len(qec_circuits))]
    non_idle_list = []
    reset_lists = [[]]
    currently_resetting = True
    unitary_qec_circuits = [stim.Circuit()]
    for j in range(len(qec_circuits)):
        for instruction in qec_circuits[j]:
            if instruction.name  == "R":
                if not currently_resetting:
                    reset_lists.append([])
                    unitary_qec_circuits.append(stim.Circuit())
                    currently_resetting = True
                reset_lists[-1] += [i.qubit_value for i in instruction.targets_copy()]
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "RX":
                if not currently_resetting:
                    reset_lists.append([])
                    unitary_qec_circuits.append(stim.Circuit())
                    currently_resetting = True
                reset_lists[-1] += [i.qubit_value for i in instruction.targets_copy()]
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset_x([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "RY":
                if not currently_resetting:
                    reset_lists.append([])
                    unitary_qec_circuits.append(stim.Circuit())
                    currently_resetting = True
                reset_lists[-1] += [i.qubit_value for i in instruction.targets_copy()]
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset_y([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "M":
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.measure([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "MX":
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.measure_x([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "MY":
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.measure_y([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "H":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.hadamard([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "S":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.s_gate([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "S_DAG":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.s_dag_gate([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "XCX":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.xcx([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "XCY":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.xcy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "XCZ":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.xcz([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "YCX":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.ycx([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "YCY":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.ycy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "YCZ":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.ycz([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "CX":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.cnot([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "CY":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.cy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "CZ":
                currently_resetting = False
                non_idle_list += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.cphase([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[-1].append(instruction)
            elif instruction.name == "TICK":
                noisy_qec_circuits[j] += noise_model.idle([str(i) for i in range(n_qubits) if not i in non_idle_list])
                non_idle_list = []
                noisy_qec_circuits[j] += noise_model.tick()
            else:
                raise NotImplementedError(f"A qec_circuit has an instruction {instruction.name} that I wasn't expecting!")
                
    
    backpropagated_logicals = deepcopy(logical_operator_basis)
    for i in range(T+2*len(qec_circuits)-1,-1,-1):
        for pauli in ["X", "Z"]:
            for j in range(n_logicals):
                post_reset_operator = backpropagated_logicals[pauli][j].before(unitary_qec_circuits[i%len(qec_circuits)])
                for reset_qubit in reset_lists[i%len(qec_circuits)]:
                    post_reset_operator[reset_qubit] = 0
                backpropagated_logicals[pauli][j] = post_reset_operator
                
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for instruction in coordinate_circuit:
        coords = instruction.gate_args_copy()
        max_x = max(max_x, coords[0])
        min_y = min(min_y, coords[1])
        max_y = max(max_y, coords[1])

    delta_y = max_y - min_y + 1
    
    delta_y = max_y - min_y + 1
    
    extra_coordinate_circuit = stim.Circuit()
    for i in range(n_logicals):
        extra_coordinate_circuit.append("QUBIT_COORDS", i + n_qubits, [max_x+1-((max_x+min_y)%1)+(i//delta_y), min_y+(i%delta_y)])
            
    circ = coordinate_circuit
    circ += extra_coordinate_circuit
    
    # The noiseless part of the circuit. To be safe, after the Bell state is prepared, all the qec_circuits are run once each.
    
    circ += noiseless_model.reset_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.reset_z([str(i) for i in range(n_qubits)])
    
    circ += tick
    for i in range(n_logicals):
        circ.append(stim.CircuitInstruction("MPP", stim.target_combined_paulis(backpropagated_logicals["Z"][i])))
        circ.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [i + n_logicals]))
        circ += tick
        
    for i in range(n_logicals):
        for j in range(len(backpropagated_logicals["X"][i])):
            qubit_Pauli_op = backpropagated_logicals["X"][i][j]
            if qubit_Pauli_op != 0:
                Pauli_str = ["I", "X", "Y", "Z"][qubit_Pauli_op]
                circ.append(stim.CircuitInstruction("C" + Pauli_str, [n_qubits + i, j]))
        circ += tick
    
    for i in range(len(qec_circuits)):
        circ += qec_circuits[i]

    # perfect Bell state prepared, now run T cycles

    repeated_circ = stim.Circuit()
    for i in range(len(qec_circuits)):
        repeated_circ += noisy_qec_circuits[i]
        repeated_circ += detector_circuits[i]

    circ += repeated_circ * (T//len(qec_circuits))
    
    for i in range(T%len(qec_circuits)):
        circ += noisy_qec_circuits[i]
        circ += detector_circuits[i]
    
    # final noiseless QEC rounds. Again to be safe, all the qec_circuits are run once each.
    # This can sometimes add some detectors that have no support on the noisy part of the circuit,
    #    this is ok though because it doesn't meaningfully change the dem.

    for i in range(len(qec_circuits)):
        circ += qec_circuits[(i+T)%len(qec_circuits)]
        circ += detector_circuits[(i+T)%len(qec_circuits)]

    for i in range(n_logicals):
        for j in range(len(logical_operator_basis["X"][i])):
            qubit_Pauli_op = logical_operator_basis["X"][i][j]
            if qubit_Pauli_op != 0:
                Pauli_str = ["I", "X", "Y", "Z"][qubit_Pauli_op]
                circ.append(stim.CircuitInstruction("C" + Pauli_str, [n_qubits + i, j]))
        circ += tick
    
    circ += noiseless_model.measure_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(i - n_logicals)], [i]
        )
    circ += tick

    for i in range(n_logicals):
        circ.append(stim.CircuitInstruction("MPP", stim.target_combined_paulis(logical_operator_basis["Z"][i])))
        circ.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [i + n_logicals]))
        circ += tick
    
    return circ

def forward_propagate_pauli(circuit, pauli):
    """
    Forward-propagates a Pauli operator through a stim circuit (ignoring noise) and returns the propagated Pauli along with the measurements that involve it.
    The Pauli operator should not forward propagate into any resets or any anticommuting measurements.
    A measurement is 'involved' in the Pauli operator if the operator coincides with the basis of the measurement on all of its qubits.
    When the measurement is involved, its support is removed from the forward-propagating Pauli operator and the measurement is added to a list of stim.target_rec() objects that are
        returned at the end of the function.

    Arguments:
        circuit is a stim.Circuit through which to propagate the Pauli operator.
        pauli is a stim.PauliString to be propagated through the circuit.

    Raises:
        Exception: The propagated pauli ran into a reset, something has gone wrong.
        Exception: The propagated pauli ran into an anticommuting measurement, something has gone wrong.
        NotImplementedError: why on earth are you using MXX MYY and MZZ instructions... sigh... I guess it's time to write this bit of the function then...

    Returns:
        propagated_pauli is a stim.PauliString representing the support of the Pauli operator after the circuit, and
        measurement_recs is a list of stim.target_rec() objects that refer to the measurements involved in the Pauli operator. Each target_rec points backwards to a measurement
            from the end of the circuit in the order that the measurements are done in the circuit.
    """
    propagated_pauli = pauli.copy()
    measurement_rec_indices = []
    for instruction in circuit.without_noise().flattened():
        if instruction.name in ["R", "RX", "RY", "RZ"]:
            if set(propagated_pauli.pauli_indices()).isdisjoint(instruction.targets_copy()):
                continue
            else:
                raise Exception("The propagated pauli ran into a reset, something has gone wrong.")
        elif instruction.name in ["M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"]:
            measurement_rec_indices = [i - instruction.num_measurements for i in measurement_rec_indices]
            
            measure_pauli_type_int = {"M": 3, "MX": 1, "MY": 2, "MZ": 3, "MR": 3, "MRX": 1, "MRY": 2, "MRZ": 3}[instruction.name]
            temp_pauli = propagated_pauli.copy()
            qubits_measured_here = [i.value for i in instruction.targets_copy()]
            for index in propagated_pauli.pauli_indices():
                if index in qubits_measured_here:
                    if propagated_pauli[index] == measure_pauli_type_int:
                        measurement_rec_indices.append(qubits_measured_here.index(index) - instruction.num_measurements)
                        temp_pauli[index] = 0
                    else:
                        raise Exception("The propagated pauli ran into an anticommuting measurement, something has gone wrong.")
            propagated_pauli = temp_pauli.copy()
        elif instruction.name in ["MPP"]:
            measurement_rec_indices = [i - instruction.num_measurements for i in measurement_rec_indices]

            for i in range(len(instruction.target_groups())):
                target_group = instruction.target_groups()[i]
                xs = np.zeros((n_qubits), dtype = np.bool_)
                zs = np.zeros((n_qubits), dtype = np.bool_)
                for target in target_group:
                    if target.is_x_target:
                        xs[target.value] = True
                    elif target.is_y_target:
                        xs[target.value] = True
                        zs[target.value] = True
                    elif target.is_z_target:
                        zs[target.value] = True
                    else:
                        raise Exception("jeez something has gone horribly wrong")
                measurement_pauli_string = stim.PauliString.from_numpy(xs = xs, zs = zs)
                if propagated_pauli.commutes(measurement_pauli_string):
                    intersection = set(propagated_pauli.pauli_indices()).intersection(measurement_pauli_string.pauli_indices())
                    if len(intersection) == measurement_pauli_string.weight():
                        for j in intersection:
                            if propagated_pauli[j] != measurement_pauli_string[j]:
                                break
                        else:
                            #this means that measurement_pauli_string is a strict subset of propagated_pauli
                            measurement_rec_indices.append(i - instruction.num_measurements)
                            propagated_pauli[j] *= measurement_pauli_string
        elif instruction.name in ["MXX", "MYY", "MZZ"]:
            raise NotImplementedError("why on earth are you using MXX MYY and MZZ instructions... sigh... I guess it's time to write this bit of the function then...")
        elif instruction.name in ["MPAD"]:
            measurement_rec_indices = [i - len(instruction.targets_copy()) for i in measurement_rec_indices]
            
        elif instruction.name in ["DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"]:
            continue

        else:
            propagated_pauli = propagated_pauli.after(instruction).copy()
                
    measurement_recs = [stim.target_rec(i) for i in measurement_rec_indices]
    return propagated_pauli, measurement_recs