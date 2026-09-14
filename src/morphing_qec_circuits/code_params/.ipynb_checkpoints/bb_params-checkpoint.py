import numpy as np
import stim
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from ..morphing import MorphingSpecifications
from .Abelian_2BGA_params import Abelian2BGAParams
from surface_sim.models import CircuitNoiseModel, NoiselessModel

F2 = galois.GF(2)

BB_codes_dict = {
    "72": BBParams(6, 6, [[3, 0], [0, 1], [0, 2]], [[0, 3], [1, 0], [2, 0]]),
    "90": BBParams(15, 3, [[9, 0], [0, 1], [0, 2]], [[0, 0], [2, 0], [7, 0]]),
    "108": BBParams(9, 6, [[3, 0], [0, 1], [0, 2]], [[0, 3], [1, 0], [2, 0]]),
    "144": BBParams(12, 6, [[3, 0], [0, 1], [0, 2]], [[0, 3], [1, 0], [2, 0]]),
    "288": BBParams(12, 12, [[3, 0], [0, 2], [0, 7]], [[0, 3], [1, 0], [2, 0]]),
    "360": BBParams(30, 6, [[9, 0], [0, 1], [0, 2]], [[0, 3], [25, 0], [26, 0]]),
    "756": BBParams(21, 18, [[3, 0], [0, 10], [0, 17]], [[0, 5], [3, 0], [19, 0]]),
}

class BBParams(Abelian2BGAParams):
    """A Bivariate Bicycle (BB) code. Subclass of Abelian2BGAParams.

    In addition to the parameters and presentation inherited from Abelian2BGAParams, BBParams also has the attributes l and m, defined such that
        group = [l, m].
    """
    def __init__(self, l, m, a_list, b_list):
        """Creates a Bivariate Bicycle (BB) code, and computes both the group and lattice presentations of the code.

        Args:
            l is an integer specifying the order of the first cyclic group in the bivariate group
            m is an integer specifying the order of the second cyclic group in the bivariate group
            a_list is a List[List[Int]] where each element of the list represents an element in the group.
            b_list is a List[List[Int]] where each element of the list represents an element in the group.
        """
        self.l = l
        self.m = m
        super().__init__([l, m], a_list, b_list)

    def valid_three_round_morphing_parameters(self, homomorphism_str):
        """
        Checks if a given Bivariate Bicycle code specification and homorphism define a valid three-round morphing circuit.

        Args:
            homomorphism_str must be "x", "y", "xy" or "x2y" depending on the choice of homomorphism

        Raises:
            ValueError: What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!

        Returns:
            True if there exists a permutation of A and B, and a value of b_shift, such that the three-round morphing circuit is valid.
        """
        if homomorphism_str == "x":
            if self.l % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            if self.m % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) == 3, len(self.b_list) == 3]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if all([(homomorphism_list.count(i) == 1) for i in range(3)]):
                    continue
                else:
                    return False
            return True

    def prepare_for_three_round_morphing(self, homomorphism_str):
        """
        Reorders the elements in A and B so that the Abelian 2BGA code is valid for three-round morphing.
        
        Arguments:
            homomorphism_str must be "x", "y", "xy" or "x2y" depending on the choice of homomorphism.
        
        Raises:
            ValueError: What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!
            ValueError: The morphing specifications are not suitable for three-round morphing!
        """
        if homomorphism_str == "x":
            if self.l % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            if self.m % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) == 3, len(self.b_list) == 3]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if all([(homomorphism_list.count(i) == 1) for i in range(3)]):
                    new_list = []
                    for i in range(3):
                        new_list.append(c_list[homomorphism_list.index(i)])
                    c_list[:] = new_list[:]
                else:
                    raise ValueError(
                        "The morphing specifications are not suitable for three-round morphing!"
                    )
        else:
            raise ValueError(
                "The morphing specifications are not suitable for three-round morphing!"
            )

    def get_three_round_morphing_specifications(self, homomorphism_str):
        """
        Returns a morphing specification for a three-round morphing circuit. 
        
        Arguments:
            homomorphism_str is a Str "x", "y", "xy", or "x2y", representing the homomorphism_actions [[1],[0]], [[0],[1]], [[1],[1]], and [[2],[1]], respectively.

        Raises:
            ValueError: What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!
            ValueError: The inputted midcycle parameters do not form a valid middle-out circuit.
        
        Returns:
            The morphing specifications for the morphing circuit.
        """
        if homomorphism_str == "x":
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_three_round_morphing(homomorphism_str)

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        if not self.valid_three_round_morphing_parameters(homomorphism_str):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self._get_real_coordinates_list()

        contraction_round_functions = {
            "X": homomorphism,
            "Z": homomorphism,
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self._get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(3)] for c_round in range(3)]
        measurement_lists = {
            pauli: [[] for c_round in range(3)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                c_round = homomorphism([i, j])
                left_x_qubits = [
                    coordinates_list.index([0, (ael[0] + i) % l, (ael[1] + j) % m])
                    for ael in a_list
                ]
                right_x_qubits = [
                    coordinates_list.index([1, (bel[0] + i) % l, (bel[1] + j) % m])
                    for bel in b_list
                ]
                z_indices = [
                    (a_list[0][0] + b_list[0][0] + i) % l,
                    (a_list[0][1] + b_list[0][1] + j) % m,
                ]
                right_z_qubits = [
                    coordinates_list.index(
                        [1, (z_indices[0] - ael[0]) % l, (z_indices[1] - ael[1]) % m]
                    )
                    for ael in a_list
                ]
                # Note that right_z_qubits[0] = right_x_qubits[0]
                gate_lists[c_round][0].append([left_x_qubits[0], right_x_qubits[0]])
                gate_lists[c_round][0].append([left_x_qubits[1], right_x_qubits[2]])
                gate_lists[c_round][0].append([left_x_qubits[2], right_x_qubits[1]])
                gate_lists[c_round][1].append([left_x_qubits[0], left_x_qubits[1]])
                gate_lists[c_round][1].append([right_z_qubits[1], right_z_qubits[0]])
                gate_lists[c_round][2].append([left_x_qubits[0], left_x_qubits[2]])
                gate_lists[c_round][2].append([right_z_qubits[2], right_z_qubits[0]])
                measurement_lists["X"][c_round].append(left_x_qubits[0])
                measurement_lists["Z"][c_round].append(right_z_qubits[0])

        complex_coordinates_list = [
            2 * coordinates[1] + 2j * coordinates[2] + (1 + 1j) * coordinates[0]
            for coordinates in coordinates_list
        ]

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications

    def get_swap2_morphing_specifications(self, homomorphism_str):
        """
        Returns a morphing specification for a morphing circuit where the second last CNOT gate has been swapped in comparison to the standard morphing circuit. This swap2 morphing circuit is a two-round homomorphism-based morphing circuit that most-straightforwardly generalises the contraction tree from the morphing circuits in Table VIII of arXiv:2407.16336 to higher-weight Abelian 2BGA codes. 
        
        Arguments:
            homomorphism_str is a Str "x", "y", or "xy", representing the homomorphism_actions [[1],[0]], [[0],[1]], and [[1],[1]], respectively.

        Raises:
            ValueError: What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!
            ValueError: The inputted midcycle parameters do not form a valid middle-out circuit.
        
        Returns:
            The morphing specifications for the morphing circuit.
        """
        if homomorphism_str == "x":
            homomorphism = lambda group_element: group_element[0] % 2
        elif homomorphism_str == "y":
            homomorphism = lambda group_element: group_element[1] % 2
        elif homomorphism_str == "xy":
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 2
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!"
            )

        if not self.valid_standard_morphing_parameters(homomorphism_str):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_standard_morphing(homomorphism_str)

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self._get_real_coordinates_list()

        contraction_round_functions = {
            "X": lambda el: (homomorphism(el) + homomorphism(a_list[0])) % 2,
            "Z": lambda el: (homomorphism(el) + homomorphism(b_list[0]) + 1) % 2,
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self._get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(max([n_a, n_b]))] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                round_g = homomorphism(
                    [i, j]
                )  # this is the round in which the left qubit labelled by [i,j] corresponds to the element g in Table IV
                round_h = (homomorphism([i, j]) + 1) % 2  # same but for h in Table IV
                this_left_qubit = coordinates_list.index([0, i, j])
                right_g_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-a_list[0][0] + bel[0] + i) % l,
                            (-a_list[0][1] + bel[1] + j) % m,
                        ]
                    )
                    for bel in b_list
                ]
                right_h_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-ael[0] + b_list[0][0] + i) % l,
                            (-ael[1] + b_list[0][1] + j) % m,
                        ]
                    )
                    for ael in a_list
                ]
                right_nucleus = coordinates_list.index(
                    [
                        1,
                        (-a_list[1][0] + b_list[1][0] + i) % l,
                        (-a_list[1][1] + b_list[1][1] + j) % m,
                    ]
                )
                for k in range(2, n_b):
                    gate_lists[round_g][k - 2].append(
                        [this_left_qubit, right_g_qubits[k]]
                    )
                for k in range(2, n_a):
                    gate_lists[round_h][k - 2].append(
                        [right_h_qubits[k], this_left_qubit]
                    )
                gate_lists[round_g][-2].append([right_g_qubits[1], this_left_qubit])
                gate_lists[round_h][-2].append([this_left_qubit, right_h_qubits[1]])
                gate_lists[round_g][-1].append([this_left_qubit, right_nucleus])
                gate_lists[round_h][-1].append([right_nucleus, this_left_qubit])
                measurement_lists["Z"][round_g].append(right_nucleus)
                measurement_lists["X"][round_h].append(right_nucleus)

        complex_coordinates_list = [
            2 * coordinates[1] + 2j * coordinates[2] + (1 + 1j) * coordinates[0]
            for coordinates in coordinates_list
        ]

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications

    def get_IBM_XZ_memory_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
    ):
        """
        Returns a stim circuit for a memory experiment using the bare-ancilla syndrome extraction circuit from Table 5 of arXiv:2308.07915.
        The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of
            ideal QEC, with T rounds of noisy QEC in between.
        The output circuit of this method is equivalent to using the get_bare_ancilla_XZ_memory_circuit by setting
            X_order = ["", "A1", "B1", "B0", "B2", "A0", "A2"], and
            Z_order = ["A0", "A2", "B0", "B1", "B2", "A1", ""],
        EXCEPT that the circuit from this method retuces the idling time by staggering the X and Z measurements.
        
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the memory experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model object
                that defines the noise model of the memory experiment.

        Raises:
            ValueError: T is too smol for the mol, make him bigger
            ValueError: a_list and b_list must have length 3
        
        Returns:
            A stim circuit for the memory experiment.
        """

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        if T < 2:
            raise ValueError("T is too smol for the mol, make him bigger")

        if len(a_list) != 3 or len(b_list) != 3:
            raise Exception("a_list and b_list must have length 3")

        n_qubits = 4 * l * m

        coords_list = [
            (label, i, j)
            for label in ["L", "R", "X", "Z"]
            for i in range(l)
            for j in range(m)
        ]

        # round indices line up mod 7 with the rounds in Fig 5 of the IBM paper arxiv:2308.07915
        gate_lists = [[] for _ in range(8)]
        measure_lists = {pauli: [[] for _ in range(8)] for pauli in ["X", "Z"]}
        reset_lists = {pauli: [[] for _ in range(8)] for pauli in ["X", "Z"]}

        for i in range(l):
            for j in range(m):
                x_ancilla = coords_list.index(("X", i, j))
                z_ancilla = coords_list.index(("Z", i, j))

                left_x_qubits = [
                    coords_list.index(("L", (ael[0] + i) % l, (ael[1] + j) % m))
                    for ael in a_list
                ]
                right_x_qubits = [
                    coords_list.index(("R", (bel[0] + i) % l, (bel[1] + j) % m))
                    for bel in b_list
                ]
                left_z_qubits = [
                    coords_list.index(("L", (-bel[0] + i) % l, (-bel[1] + j) % m))
                    for bel in b_list
                ]
                right_z_qubits = [
                    coords_list.index(("R", (-ael[0] + i) % l, (-ael[1] + j) % m))
                    for ael in a_list
                ]

                measure_lists["X"][0].append(str(x_ancilla))
                reset_lists["X"][1].append(str(x_ancilla))
                measure_lists["Z"][7].append(str(z_ancilla))
                reset_lists["Z"][0].append(str(z_ancilla))

                gate_lists[1].append(str(right_z_qubits[0]))
                gate_lists[1].append(str(z_ancilla))
                gate_lists[2].append(str(right_z_qubits[2]))
                gate_lists[2].append(str(z_ancilla))
                gate_lists[3].append(str(left_z_qubits[0]))
                gate_lists[3].append(str(z_ancilla))
                gate_lists[4].append(str(left_z_qubits[1]))
                gate_lists[4].append(str(z_ancilla))
                gate_lists[5].append(str(left_z_qubits[2]))
                gate_lists[5].append(str(z_ancilla))
                gate_lists[6].append(str(right_z_qubits[1]))
                gate_lists[6].append(str(z_ancilla))

                gate_lists[2].append(str(x_ancilla))
                gate_lists[2].append(str(left_x_qubits[1]))
                gate_lists[3].append(str(x_ancilla))
                gate_lists[3].append(str(right_x_qubits[1]))
                gate_lists[4].append(str(x_ancilla))
                gate_lists[4].append(str(right_x_qubits[0]))
                gate_lists[5].append(str(x_ancilla))
                gate_lists[5].append(str(right_x_qubits[2]))
                gate_lists[6].append(str(x_ancilla))
                gate_lists[6].append(str(left_x_qubits[0]))
                gate_lists[7].append(str(x_ancilla))
                gate_lists[7].append(str(left_x_qubits[2]))

        x_matrix = F2(np.diag(np.ones((l - 1)), 1).astype(int))
        x_matrix[l - 1, 0] = 1
        y_matrix = F2(np.diag(np.ones((m - 1)), 1).astype(int))
        y_matrix[m - 1, 0] = 1
        amatrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, ael[0]),
                np.linalg.matrix_power(y_matrix, ael[1]),
            )
            for ael in a_list
        ]
        bmatrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, bel[0]),
                np.linalg.matrix_power(y_matrix, bel[1]),
            )
            for bel in b_list
        ]
        amatrix = F2(np.zeros((l * m, l * m)).astype(int))
        bmatrix = F2(np.zeros((l * m, l * m)).astype(int))
        for i in range(len(a_list)):
            amatrix += amatrices[i]
            bmatrix += bmatrices[i]
        h = {
            "X": np.concatenate((amatrix, bmatrix), axis=1),
            "Z": np.concatenate((bmatrix.T, amatrix.T), axis=1),
        }

        stabilisers = {pauli: [] for pauli in ["X", "Z"]}
        for i in range(l):
            for j in range(m):
                stabilisers["X"].append(np.nonzero(h["X"][m * i + j])[0].tolist())
                stabilisers["Z"].append(np.nonzero(h["Z"][m * i + j])[0].tolist())

        hker = {pauli: h[pauli].null_space() for pauli in ["X", "Z"]}
        stabs_to_logicals = {pauli: h[pauli].row_reduce() for pauli in ["X", "Z"]}
        observables = {pauli: [] for pauli in ["X", "Z"]}
        
        for pauli in ["X", "Z"]:
            other_pauli = "Z" if pauli == "X" else "X"
            for i in range(hker[other_pauli].shape[0]):
                new_vecs = np.concatenate(
                    (stabs_to_logicals[pauli], [hker[other_pauli][i]]), axis=0
                )
                if np.linalg.matrix_rank(new_vecs) != np.linalg.matrix_rank(
                    stabs_to_logicals[pauli]
                ):
                    stabs_to_logicals[pauli] = new_vecs
                    observables[pauli].append(
                        np.nonzero(hker[other_pauli][i])[0].tolist()
                    )
        n_logicals = len(observables["X"])

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)
        noiseless_model = NoiselessModel(all_qubits_dictionary)
        tick = noiseless_model.tick()

        labels = ["L", "R", "X", "Z"]
        qubit_strs = {
            labels[j]: [str(i) for i in range(j * l * m, (j + 1) * l * m)]
            for j in range(4)
        }

        offsets = {"L": 0, "R": 1 + 1j, "X": 1, "Z": 1j}
        coordinates_list = []
        for i in range(n_qubits):
            offset = offsets[coords_list[i][0]]
            coordinates_list.append(
                round(2 * coords_list[i][1] + np.real(offset))
                + 1j * round(2 * coords_list[i][2] + np.imag(offset))
            )

        #copy/paste of the coordinates_circuit function from morphing_specifications.py

        circuit_str = ""

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            min_x = min(round(coord.real), min_x)
            max_x = max(round(coord.real), max_x)
            min_y = min(round(coord.imag), min_y)
            max_y = max(round(coord.imag), max_y)
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            circuit_str += f"QUBIT_COORDS({-min_x + round(coord.real)}, {(max_y - round(coord.imag))}) {i}\n"
            # the minus makes sure y always is pointing up.
            # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)
        for j in range(n_logicals):
            circuit_str += f"QUBIT_COORDS({-min_x + max_x + 1 + (j // (-min_y+max_y+1))}, {-min_y + max_y - (j % (-min_y+max_y+1))}) {len(coordinates_list) + j}\n"

        circ = stim.Circuit(circuit_str)

        # prepare an error-free Bell state between each logical qubit and each perfect dummy qubit

        circ += noiseless_model.reset_x(
            [str(i) for i in range(4 * l * m, 4 * l * m + n_logicals)]
        )
        circ += noiseless_model.reset(qubit_strs["L"] + qubit_strs["R"])
        circ += tick

        for i in range(n_logicals):
            for qubit in observables["X"][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
            circ += tick

        projection_str = ""
        for pauli in ["Z", "X"]:
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

        circ += noise_model.reset(reset_lists["Z"][0])
        circ += noise_model.idle(
            [str(i) for i in range(4 * l * m) if str(i) not in reset_lists["Z"][0]]
        )
        circ += noise_model.tick()

        for i in [1, 2, 3, 4, 5, 6, 7, 0]:
            circ += noise_model.cnot(gate_lists[i])
            circ += noise_model.measure_x(measure_lists["X"][i])
            circ += noise_model.measure(measure_lists["Z"][i])
            circ += noise_model.reset_x(reset_lists["X"][i])
            circ += noise_model.reset(reset_lists["Z"][i])
            circ += noise_model.idle(
                [
                    str(j)
                    for j in range(4 * l * m)
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
        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        if T > 2:
            repeat_circ = stim.Circuit()
            for i in [1, 2, 3, 4, 5, 6, 7, 0]:
                repeat_circ += noise_model.cnot(gate_lists[i])
                repeat_circ += noise_model.measure_x(measure_lists["X"][i])
                repeat_circ += noise_model.measure(measure_lists["Z"][i])
                repeat_circ += noise_model.reset_x(reset_lists["X"][i])
                repeat_circ += noise_model.reset(reset_lists["Z"][i])
                repeat_circ += noise_model.idle(
                    [
                        str(j)
                        for j in range(4 * l * m)
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
            for i in range(l * m):
                repeat_circ.append(
                    "DETECTOR",
                    [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                    (0, 0, 0, 3),
                )
            for i in range(l * m, 2 * l * m):
                repeat_circ.append(
                    "DETECTOR",
                    [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                    (0, 0, 0, 0),
                )
            circ += repeat_circ * (T - 2)

        for i in [1, 2, 3, 4, 5, 6, 7]:
            circ += noise_model.cnot(gate_lists[i])
            circ += noise_model.measure_x(measure_lists["X"][i])
            circ += noise_model.measure(measure_lists["Z"][i])
            circ += noise_model.reset_x(reset_lists["X"][i])
            circ += noise_model.reset(reset_lists["Z"][i])
            circ += noise_model.idle(
                [
                    str(j)
                    for j in range(4 * l * m)
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

        # no idling noise in this round to transition to the noise-free Bell measurements
        circ += noise_model.cnot(gate_lists[0])
        circ += noise_model.measure_x(measure_lists["X"][0])
        circ += noise_model.measure(measure_lists["Z"][0])
        circ += noise_model.reset_x(reset_lists["X"][0])
        circ += noise_model.reset(reset_lists["Z"][0])
        circ += tick

        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        circ += stim.Circuit(projection_str)
        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        # perfect measurement of the observables
        for i in range(n_logicals):
            for qubit in observables["X"][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
            circ += tick

        circ += noiseless_model.measure_x(
            [str(i) for i in range(4 * l * m, 4 * l * m + n_logicals)]
        )
        circ += noiseless_model.measure([str(i) for i in range(2 * l * m)])

        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-2 * l * m - n_logicals + i)], i
            )
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-2 * l * m + qubit) for qubit in observables["Z"][i]],
                n_logicals + i,
            )
        return circ