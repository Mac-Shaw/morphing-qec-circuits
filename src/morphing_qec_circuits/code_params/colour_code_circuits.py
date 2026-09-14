import stim
from .bb_circuit_params import BBCircuitParams
from surface_sim.models import CircuitNoiseModel, NoiselessModel

class ToricColourCodeCircuitParams(BBCircuitParams):
    def __init__(
        self,
        name,
        connectivity,
        circuit_type,
        toric_colour_code_params,
        homomorphism_str,
        circuit=stim.Circuit(),
    ):
        """
        Kept for backwards compatibility, I don't think I use this anywhere.
        """
        self.name = name
        self.circuit_type = (connectivity, circuit_type)
        self.bb_params = toric_colour_code_params
        self.homomorphism_str = homomorphism_str
        self.circuit = circuit

    def compile_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
    ):
        specification = self._get_specifications()
        self.circuit = specification.get_morphing_XZ_memory_circuit(
            noise_setup, T, noise_model_generator=noise_model_generator
        )

    def _get_specifications(self):
        return self.bb_params.get_colour_code_morphing_specification(
            self.homomorphism_str, self.circuit_type[0], self.circuit_type[1]
        )

def toric_colour_code_circuit_params(
    d, geometry="", homomorphism_str="x", connectivity="hex", circuit_type="shifted"
):
    """
    Kept for backwards compatibility, I don't think I use this anywhere.
    """
    if geometry == "twisted":
        if d % 8 == 0:
            toric_colour_code_params = toric_colour_code(d, geometry = geometry)
        else:
            raise Exception(
                "d must be divisible by 8 for the morphing twisted toric colour code circuit"
            )
    elif geometry in ["", "standard", "non-twisted"]:
        geometry = "non-twisted"
        if d % 4 == 0:
            toric_colour_code_params = toric_colour_code(d, geometry = geometry)
        else:
            raise Exception(
                "d must be divisible by 4 for the morphing standard, non-twisted toric colour code circuit"
            )
    elif geometry == "standard-for-half-shifted-hex":
        if d % 2 == 0:
            toric_colour_code_params = toric_colour_code(d, geometry = geometry)
        else:
            raise Exception(
                "d must be divisible by 2 for the morphing standard-for-half-shifted-hex toric colour code circuit"
            )
    elif geometry == "twisted-for-half-shifted-hex":
        if d % 4 == 0:
            toric_colour_code_params = toric_colour_code(d, geometry = geometry)
        else:
            raise Exception(
                "d must be divisible by 12 for the morphing rotated toric colour code circuit"
            )
    elif geometry == "rotated":
        if d % 12 == 0:
            toric_colour_code_params = toric_colour_code(d, geometry = geometry)
        else:
            raise Exception(
                "d must be divisible by 12 for the morphing rotated toric colour code circuit"
            )
    else:
        raise NotImplementedError(
            "geometry must be standard / non-twisted, twisted, or rotated."
        )
    name = f"d={d}, {geometry} toric colour code; {connectivity} {circuit_type} {homomorphism_str} morphing circuit"
    return ToricColourCodeCircuitParams(
        name, connectivity, circuit_type, toric_colour_code_params, homomorphism_str
    )

def get_triangular_colour_code_incoming_XZ_memory_circuit(
    d,
    p,
):
    """
    Returns a stim circuit for a planar, triangular colour code memory experiment under incoming noise.
    The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of
        ideal QEC, with one layer of data-qubit errors in between.
    
    Arguments:
        d is an Int representing the distance of the colour code.
        p is a float representing the probability of a Pauli error occurring in a single-qubit depolarising noise channel that is applied to each data qubit.

    Raises:
        ValueError: d must be odd for this function!
    
    Returns:
        A stim circuit for the memory experiment.
    """

    if (d%2) != 1:
        raise Exception("d must be odd for this function!")

    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if x < 0:
            return False
        elif y >= -2*x/3 + 2*d - 7/6:
            return False
        elif y <= 2*x/3 - 1/2:
            return False
        else:
            return True

    def is_north_ancilla(coords):
        x = coords[0]
        y = coords[1]
        if qubit_in_bulk(coords):
            if y % 2 == 0:
                if ((x+1)%2)*2 == y%4:
                    return True
        return False

    def is_data_qubit(coords):
        return qubit_in_bulk(coords) and (not is_north_ancilla(coords)) and (not is_north_ancilla([coords[0], coords[1]+1]))

    coords_list = [(x, y) for x in range(3*d//2+2) for y in range(2*d+1) if is_data_qubit([x,y])]
    n_qubits = len(coords_list)

    # round indices line up with the layers in Fig 1b of the Google paper arxiv:2412.14256
    
    stabilisers = {pauli: [] for pauli in ["X", "Z"]}
    colours = []

    for x in range(3*d//2+2):
        for y in range(2*d+1):
            if not is_north_ancilla([x,y]):
                continue
            stabiliser = []
            
            directions = [[1,0],[0,1],[-1,0]]
            for i in range(3):
                N_data_coords = (x + directions[i][0], y + directions[i][1])
                if qubit_in_bulk(N_data_coords):
                    N_data = coords_list.index(N_data_coords)
                    stabiliser.append(N_data)
                S_data_coords = (x - directions[i][0], y - 1 - directions[i][1])
                if qubit_in_bulk(S_data_coords):
                    S_data = coords_list.index(S_data_coords)
                    stabiliser.append(S_data)
            [stabilisers[pauli].append(stabiliser) for pauli in ["X", "Z"]]
            colours.append((-x)%3)

    n_faces = len(stabilisers["X"])

    observables = {pauli: [[coords_list.index((0,y)) for y in range(2*d-1) if (y%4) in [0,3]]] for pauli in ["X", "Z"]}

    n_logicals = 1# kept so I can reuse previous code

    qubit_dictionary = {str(i): i for i in range(n_qubits)}
    all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
    noiseless_model = NoiselessModel(all_qubits_dictionary)
    tick = noiseless_model.tick()

    #copy/paste of the coordinates_circuit function from morphing_specifications.py

    circuit_str = ""

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(len(coords_list)):
        coord = coords_list[i]
        min_x = min(round(coord[0]), min_x)
        max_x = max(round(coord[0]), max_x)
        min_y = min(round(coord[1]), min_y)
        max_y = max(round(coord[1]), max_y)
    for i in range(len(coords_list)):
        coord = coords_list[i]
        circuit_str += f"QUBIT_COORDS({-min_x + round(coord[0])}, {(max_y - round(coord[1]))}) {i}\n"
        # the minus makes sure y always is pointing up.
        # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)
    for j in range(n_logicals):
        circuit_str += f"QUBIT_COORDS({-min_x + max_x + 1 + (j // (-min_y+max_y+1))}, {-min_y + max_y - (j % (-min_y+max_y+1))}) {len(coords_list) + j}\n"

    circ = stim.Circuit(circuit_str)

    # prepare an error-free Bell state between each logical qubit and each perfect dummy qubit

    circ += noiseless_model.reset_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.reset([str(i) for i in range(n_qubits)])
    circ += tick

    for i in range(n_logicals):
        for qubit in observables["X"][i]:
            circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
        circ += tick

    projection_str = ""
    for pauli in ["X", "Z"]:
        for stabiliser in stabilisers[pauli]:
            projection_str += "\nTICK\n"
            projection_str += "MPP"
            projection_str += " "
            for i in range(len(stabiliser)):
                if i > 0:
                    projection_str += "*"
                projection_str += pauli + str(stabiliser[i])
    circ += stim.Circuit(projection_str)
    circ += tick

    # perfect state prepared, now insert the incoming noise.

    circ.append("DEPOLARIZE1", range(n_qubits), p)

    #perfect projection and measurements

    circ += stim.Circuit(projection_str)
    
    for i in range(n_faces):
        circ.append(
            "DETECTOR",
            [stim.target_rec(-4*n_faces + i), stim.target_rec(-2*n_faces + i)],
            (0, 0, 0, colours[i]),
        )
        circ.append(
            "DETECTOR",
            [stim.target_rec(-3*n_faces + i), stim.target_rec(-n_faces + i)],
            (0, 0, 0, colours[i] + 3),
        )
        
    # perfect measurement of the observables
    for i in range(n_logicals):
        for qubit in observables["X"][i]:
            circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
        circ += tick

    circ += noiseless_model.measure_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.measure([str(i) for i in range(n_qubits)])

    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(-n_qubits - n_logicals + i)], i
        )
    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-n_qubits + qubit) for qubit in observables["Z"][i]],
            n_logicals + i,
        )
    return circ

def get_triangular_colour_code_superdense_XZ_memory_circuit(
    d,
    noise_setup,
    T,
    noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
        qubit_dictionary, noise_setup
    ),
):
    """
    Returns a stim circuit for a planar, triangular colour code memory experiment using a superdense syndrome extraction circuit.
    The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of
        ideal QEC, with T rounds of noisy QEC in between.
    
    Arguments:
        d is an Int representing the distance of the colour code.
        noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
        T is the number of noisy QEC rounds in the memory experiment.
        noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model object
            that defines the noise model of the memory experiment.

    Raises:
        ValueError: T must be larger than or equal to 3 for this function
        ValueError: d must be odd for this function!
    
    Returns:
        A stim circuit for the memory experiment.
    """

    if T < 2:
        raise ValueError("T must be larger than or equal to 3 for this function")

    if (d%2) != 1:
        raise Exception("d must be odd for this function!")

    def qubit_in_bulk(coords):
        x = coords[0]
        y = coords[1]
        if x < 0:
            return False
        elif y >= -2*x/3 + 2*d - 7/6:
            return False
        elif y <= 2*x/3 - 1/2:
            return False
        else:
            return True

    def is_north_ancilla(coords):
        x = coords[0]
        y = coords[1]
        if qubit_in_bulk(coords):
            if y % 2 == 0:
                if ((x+1)%2)*2 == y%4:
                    return True
        return False

    coords_list = [(x, y) for x in range(3*d//2+2) for y in range(2*d+1) if qubit_in_bulk([x,y])]
    n_qubits = len(coords_list)
    
    data_qubit_str = [str(coords_list.index((x,y)))
                   for x in range(3*d//2+2)
                   for y in range(2*d+1)
                   if (qubit_in_bulk([x,y]) and not (is_north_ancilla([x,y]) or is_north_ancilla([x,y+1])))]

    # round indices line up with the layers in Fig 1b of the Google paper arxiv:2412.14256
    gate_lists = [[] for _ in range(10)]
    measure_lists = {pauli: [[] for _ in range(10)] for pauli in ["X", "Z"]}
    reset_lists = {pauli: [[] for _ in range(10)] for pauli in ["X", "Z"]}

    stabilisers = {pauli: [] for pauli in ["X", "Z"]}
    colours = []
    Z_observable_to_include = []

    for x in range(3*d//2+2):
        for y in range(2*d+1):
            if not is_north_ancilla([x,y]):
                continue
            #north ancillas reset in X and measure in Z; south ancillas reset in Z and measure in X.
            #It's unclear to my why they do this swapping between X and Z, but I'm just doing it for consistency with Google's work.
            N_ancilla = coords_list.index((x, y))
            S_ancilla = coords_list.index((x, y-1))
            stabiliser = []

            reset_lists["X"][0].append(str(N_ancilla))
            reset_lists["Z"][0].append(str(S_ancilla))
            measure_lists["X"][9].append(str(S_ancilla))
            measure_lists["Z"][9].append(str(N_ancilla))

            gate_lists[1].append(str(N_ancilla))
            gate_lists[1].append(str(S_ancilla))
            gate_lists[8].append(str(S_ancilla))
            gate_lists[8].append(str(N_ancilla))

            directions = [[1,0],[0,1],[-1,0]]
            for i in range(3):
                N_data_coords = (x + directions[i][0], y + directions[i][1])
                if qubit_in_bulk(N_data_coords):
                    N_data = coords_list.index(N_data_coords)
                    stabiliser.append(N_data)
                    gate_lists[i+2].append(str(N_data))
                    gate_lists[i+2].append(str(N_ancilla))
                    gate_lists[i+5].append(str(N_ancilla))
                    gate_lists[i+5].append(str(N_data))
                S_data_coords = (x - directions[i][0], y - 1 - directions[i][1])
                if qubit_in_bulk(S_data_coords):
                    S_data = coords_list.index(S_data_coords)
                    stabiliser.append(S_data)
                    gate_lists[i+2].append(str(S_data))
                    gate_lists[i+2].append(str(S_ancilla))
                    gate_lists[i+5].append(str(S_ancilla))
                    gate_lists[i+5].append(str(S_data))
            [stabilisers[pauli].append(stabiliser) for pauli in ["X", "Z"]]
            colours.append((-x)%3)
            if x in [0,1]:
                Z_observable_to_include.append(len(measure_lists["Z"][9])-1)

    n_faces = len(stabilisers["X"])
    Z_detector_to_include = [[] for _ in range(n_faces)]
    i = 0
    for x in range(3*d//2+2):
        for y in range(2*d+1):
            if not is_north_ancilla([x,y]):
                continue
            directions = [[1,0],[0,1],[-1,0]]
            own_measurement = True
            for direction in directions:
                if qubit_in_bulk([x + direction[0], y + direction[1]]):
                    own_measurement ^= True
            if own_measurement:
                Z_detector_to_include[i].append(i)
            if qubit_in_bulk([x - 2, y]):
                Z_detector_to_include[i].append(measure_lists["Z"][9].index(str(coords_list.index((x-2,y)))))
            if qubit_in_bulk([x + 2, y]):
                Z_detector_to_include[i].append(measure_lists["Z"][9].index(str(coords_list.index((x+2,y)))))
            i += 1

    observables = {pauli: [[coords_list.index((0,y)) for y in range(2*d-1) if (y%4) in [0,3]]] for pauli in ["X", "Z"]}

    n_logicals = 1# kept so I can reuse previous code

    qubit_dictionary = {str(i): i for i in range(n_qubits)}
    all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
    noise_model = noise_model_generator(qubit_dictionary, noise_setup)
    noiseless_model = NoiselessModel(all_qubits_dictionary)
    tick = noiseless_model.tick()

    #copy/paste of the coordinates_circuit function from morphing_specifications.py

    circuit_str = ""

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(len(coords_list)):
        coord = coords_list[i]
        min_x = min(round(coord[0]), min_x)
        max_x = max(round(coord[0]), max_x)
        min_y = min(round(coord[1]), min_y)
        max_y = max(round(coord[1]), max_y)
    for i in range(len(coords_list)):
        coord = coords_list[i]
        circuit_str += f"QUBIT_COORDS({-min_x + round(coord[0])}, {(max_y - round(coord[1]))}) {i}\n"
        # the minus makes sure y always is pointing up.
        # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)
    for j in range(n_logicals):
        circuit_str += f"QUBIT_COORDS({-min_x + max_x + 1 + (j // (-min_y+max_y+1))}, {-min_y + max_y - (j % (-min_y+max_y+1))}) {len(coords_list) + j}\n"

    circ = stim.Circuit(circuit_str)

    # prepare an error-free Bell state between each logical qubit and each perfect dummy qubit

    circ += noiseless_model.reset_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.reset(data_qubit_str)
    circ += tick

    for i in range(n_logicals):
        for qubit in observables["X"][i]:
            circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
        circ += tick

    projection_str = ""
    for pauli in ["X", "Z"]:
        for stabiliser in stabilisers[pauli]:
            projection_str += "\nTICK\n"
            projection_str += "MPP"
            projection_str += " "
            for i in range(len(stabiliser)):
                if i > 0:
                    projection_str += "*"
                projection_str += pauli + str(stabiliser[i])
    circ += stim.Circuit(projection_str)
    circ += tick

    # perfect state prepared, now run the T rounds of noisy QEC.

    for i in range(10):
        circ += noise_model.cnot(gate_lists[i])
        circ += noise_model.measure_x(measure_lists["X"][i])
        circ += noise_model.measure(measure_lists["Z"][i])
        circ += noise_model.reset_x(reset_lists["X"][i])
        circ += noise_model.reset(reset_lists["Z"][i])
        circ += noise_model.idle(
            [
                str(j)
                for j in range(n_qubits)
                if str(j)
                not in (
                    gate_lists[i]
                    + measure_lists["X"][i]
                    + measure_lists["Z"][i]
                    + reset_lists["X"][i]
                    + reset_lists["Z"][i]
                )
            ]
        )
        circ += noise_model.tick()

    for i in range(2*n_faces):
        circ.append(
            "DETECTOR",
            [stim.target_rec(-4*n_faces + i), stim.target_rec(-2*n_faces + i)],
            (0, 0, 0, colours[i%n_faces] + 3 * (i//n_faces)),
        )
    circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(-n_faces + i) for i in Z_observable_to_include], 1)
    
    if T > 1:
        repeat_circ = stim.Circuit()
        for i in range(10):
            repeat_circ += noise_model.cnot(gate_lists[i])
            repeat_circ += noise_model.measure_x(measure_lists["X"][i])
            repeat_circ += noise_model.measure(measure_lists["Z"][i])
            repeat_circ += noise_model.reset_x(reset_lists["X"][i])
            repeat_circ += noise_model.reset(reset_lists["Z"][i])
            repeat_circ += noise_model.idle(
                [
                    str(j)
                    for j in range(n_qubits)
                    if str(j)
                    not in (
                        gate_lists[i]
                        + measure_lists["X"][i]
                        + measure_lists["Z"][i]
                        + reset_lists["X"][i]
                        + reset_lists["Z"][i]
                    )
                ]
            )
            repeat_circ += noise_model.tick()
        for i in range(n_faces):
            repeat_circ.append(
                "DETECTOR",
                [stim.target_rec(-4*n_faces + i), stim.target_rec(-2*n_faces + i)],
                (0, 0, 0, colours[i]),
            )
        for i in range(n_faces):
            repeat_circ.append(
                "DETECTOR",
                [stim.target_rec(-3*n_faces + j) for j in Z_detector_to_include[i]] + [stim.target_rec(-n_faces + i)],
                (0, 0, 0, colours[i] + 3),
            )
        repeat_circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(-n_faces + i) for i in Z_observable_to_include], 1)
        circ += repeat_circ * (T - 1)
    #I don't think I need to explicitly write out this last round??

    circ += stim.Circuit(projection_str)
    
    for i in range(n_faces):
        circ.append(
            "DETECTOR",
            [stim.target_rec(-4*n_faces + i), stim.target_rec(-2*n_faces + i)],
            (0, 0, 0, colours[i]),
        )
    for i in range(n_faces):
        circ.append(
            "DETECTOR",
            [stim.target_rec(-3*n_faces + j) for j in Z_detector_to_include[i]] + [stim.target_rec(-n_faces + i)],
            (0, 0, 0, colours[i] + 3),
        )

    # perfect measurement of the observables
    for i in range(n_logicals):
        for qubit in observables["X"][i]:
            circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
        circ += tick

    circ += noiseless_model.measure_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.measure([str(i) for i in range(n_qubits)])

    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(-n_qubits - n_logicals + i)], i
        )
    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-n_qubits + qubit) for qubit in observables["Z"][i]],
            n_logicals + i,
        )
    return circ