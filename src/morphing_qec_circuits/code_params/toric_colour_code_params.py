from copy import deepcopy
from .bb_params import BBParams
from ..morphing import MorphingSpecifications

def toric_colour_code(d, geometry="", end_cycle_distance = False):
    '''
    Generates the ToricColourCodeParams for a 6.6.6 toric colour code laid out on a torus with (static) distance d.

    Arguments:
        geometry is a string that specifies the geometry of the colour code when laid out on a torus. See toric_colour_code_boundaries.pdf for a visualisation.
            "standard", or "non-twisted", is the standard, brick layout of the toric colour code. This is the optimised layout for the planar colour code
                but is outperformed by the twisted layout (see below) on a torus.
                [[n,k,d]] = [[3/2 * d**2, 4, d]].
            "twisted" is the optimised layout of the toric colour code. With boundary conditions, this is not as good as the standard layout.
                [[n,k,d]] = [[9/8 * d**2, 4, d]] for d divisible by 4,
                [[n,k,d]] = [[9/8 * d**2 + 3/2, 4, d]] for d congruent to 2 modulo 4.
            "standard-for-half-shifted-hex" gives the mid-cycle layout of the toric colour code such that the end-cycle code under the half-shifted
                hex morphing circuit with the "x" homomorphism has logical structure similar to the standard, non-twisted layout of the colour code.
                That is, it is the optimised layout for the planar half-shifted hex morphing colour code, but is outperformed on the torus by the
                twisted-for-half-shifted-hex layout (see below).
                [[n,k,d]] = [[3   * d**2, 4, d]] in the mid-cycle code.
                [[n,k,d]] = [[3/2 * d**2, 4, d]] in the end-cycle code under the half-shifted hex morphing circuit with the "x" homomorphism.
                I really thought that this would suffer from the error mechanism in Fig 28(b) of arXiv:2604:09797 but somehow it doesn't?
            "twisted-for-half-shifted-hex" gives the mid-cycle layout of the toric colour code such that the end-cycle code under the half-shifted
                hex morphing circuit with the "x" homomorphism has logical structure similar to the twisted layout of the colour code.
                [[n,k,d]] = [[9/4 * d**2, 4, d]] in the mid-cycle code.
                [[n,k,d]] = [[9/8 * d**2, 4, d]] in the end-cycle code under the half-shifted hex morphing circuit with the "x" homomorphism.
                Note that the morphing circuit has d_circ = 3/4 * d, it suffers from the error mechanism in Fig 28(b) of arXiv:2604:09797.
                Doing the "even_more_layers" thing does seem to fix this. It is not natively supported but can be implemented via
                    d = 4 #for example
                    test = toric_colour_code(d, "twisted-for-half-shifted-hex", end_cycle_distance = True)
                    contraction_tree = [[["B2","A1"]],[["A1","B0"]],[["B0","B1"]],[["A0","B1"]],[["A2","B1"]]]
                    homomorphism_image = [2]
                    homomorphism_action = [[1],[0]]
                    b_shift = [1]
                    test_morphing = test.get_morphing_specifications(contraction_tree, homomorphism_image, homomorphism_action, b_shift, swap_AB_rounds = [1])
                    and it gave me d_circ = 4 for d = 4 (couldn't be bothered testing it for higher distances).
            "rotated" gives the mid-cycle layout of the toric colour code such that the end-cycle code under the shifted and centred hex morphing circuit
                with the "x" homomorphism has logical structure similar to two copies of the toric code.
                [[n,k,d]] = [[4/3 * d**2, 4, d]] in the mid-cycle code.
                [[n,k,d]] = [[3/2 * d_end**2, 4, d_end]] in the end-cycle code under the shifted or centred hex morphing circuit with the "x" homomorphism,
                    where d_end = 2/3 * d.
        end_cycle_distance is a bool. When True, the inputted d is interpreted as the end-cycle distance of the morphing circuit for the given geometry.
            When False, it is simply the distance of the returned toric colour code, which is the mid-cycle code of a morphing circuit. This only makes a
            difference when geometry = "rotated".
    '''
    if geometry == "twisted":
        if d % 4 == 0:
            l = 3 * (d // 4)
            m = l
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [0, 1]])
        elif d % 4 == 2:
            l = (9 * d**2 + 12) // 16
            m = 1
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [(3*d + 2)//4, 0]])
        else:
            raise Exception(
                "d must be divisible by 2 for the twisted toric colour code"
            )
    elif geometry in ["", "standard", "non-twisted"]:
        if d % 2 == 0:
            m = d // 2
            l = 3 * m
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [-1, 1]])
        else:
            raise Exception(
                "d must be divisible by 2 for the standard, non-twisted toric colour code"
            )
    elif geometry == "standard-for-half-shifted-hex":
        if d % 2 == 0:
            m = d // 2
            l = 3 * d
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [2, 1]])
        else:
            raise Exception(
                "d must be divisible by 2 for the standard-for-half-shifted-hex toric colour code"
            )
    elif geometry == "twisted-for-half-shifted-hex":
        if d % 4 == 0:
            m = d // 4
            l = 9 * d // 2
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [-4, 1]])
        else:
            raise Exception(
                "d must be divisible by 4 for the twisted-for-half-shifted-hex toric colour code"
            )
    elif geometry == "rotated":
        if end_cycle_distance:
            d = (d * 3) // 2
        if d % 6 == 0:
            m = d // 6
            l = 4 * d
            return ToricColourCodeParams(l, m, [[0, 0], [1, 0], [-4, 1]])
        else:
            raise Exception(
                "d must be divisible by 6 for the rotated toric colour code"
            )
    else:
        raise NotImplementedError(
            "geometry must be standard, twisted, standard-for-half-shifted-hex, twisted-for-half-shifted-hex, or rotated."
        )

class ToricColourCodeParams(BBParams):
    """
    A class for the (hexagonal grid) colour code on a torus.
    """
    def __init__(self, l, m, a_list):
        if len(a_list) != 3:
            raise ValueError("ToricColourCodeParams must be initialised with an a_list of length 3!")
        b_list = [[(-a_el[0]) % l, (-a_el[1]) % m] for a_el in a_list]
        super().__init__(l, m, a_list, b_list)

    def get_colour_code_morphing_specifications(
        self, homomorphism_str, connectivity, circuit_type, **kwargs
    ):
        """
        Returns a morphing specification for a morphing circuit for the toric colour code.
        For example, the contraction circuits in Fig 17 of arXiv:2604.09797 correspond to:
            (a) connectivity = "hex", circuit_type = "shifted"
            (b) connectivity = "hex", circuit_type = "half-shifted"
            (c) connectivity = "hex", circuit_type = "centred"
        
        Arguments:
            homomorphism_str is a Str "x", "y", or "xy" that represents homomorphism.
            connectivity is a Str "hex" or "sq" that corresponds the connectivity requirements of the output morphing circuit.
            circuit_type is a Str "shifted", "twisted", "centred", or "half-shifted" that determines the order of the CNOT gates that are applied.

        Raises:
            ValueError: connectivity should be hex or sq!
            ValueError: circuit_type should be shifted, twisted, centred or half-shifted!
            ValueError: What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!
            ValueError: The inputted midcycle parameters do not form a valid middle-out circuit.
        
        Returns:
            The morphing specifications for the morphing circuit.
        """
        if connectivity not in ["hex", "sq"]:
            raise ValueError("connectivity should be hex or sq!")
        elif circuit_type not in ["shifted", "twisted", "centred", "half-shifted"]:
            raise ValueError(
                "circuit_type should be shifted, twisted, centred or half-shifted!"
            )
        if connectivity == "hex":
            if circuit_type == "shifted":
                return self._get_shifted_hex_morphing_specifications(
                    homomorphism_str, **kwargs
                )
            elif circuit_type == "twisted":
                return self.get_swap2_morphing_specifications(homomorphism_str, **kwargs)
            elif circuit_type == "centred":
                return self.get_standard_morphing_specifications(homomorphism_str, **kwargs)
            elif circuit_type == "half-shifted":
                return self._get_half_shifted_hex_morphing_specifications(
                    homomorphism_str, **kwargs
                )
        elif connectivity == "sq":
            return self._get_sq_morphing_specifications(homomorphism_str, circuit_type, **kwargs)

    def _get_shifted_hex_morphing_specifications(self, homomorphism_str):
        old_b_list = deepcopy(self.b_list)
        self.b_list = [self.b_list[0], self.b_list[2], self.b_list[1]]
        specifications = self.get_swap2_morphing_specifications(homomorphism_str)
        self.b_list = old_b_list
        return specifications

    def _get_half_shifted_hex_morphing_specifications(self, homomorphism_str, simple_detectors = False):
        '''
        simple_detectors determines the direction of the CNOTs in the final layer of the contraction round and the subsequent qubits that are measured.
            When set to False, the CNOTs go in the direction such that each detector is made up of 3 measurement outcomes.
            When set to True, the CNOTs go in the opposite direction such that each detector is made up of 2 measurement outcomes.
        '''
        
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

        gate_lists = [[[] for layer in range(3)] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                round_X = contraction_round_functions["X"](
                    [i, j]
                )  # this is the round in which the X stabiliser labelled by [i,j] is contracting
                other_round = (round_X + 1) % 2
                left_qubits = [
                    coordinates_list.index([0, (a_el[0] + i) % l, (a_el[1] + j) % m])
                    for a_el in a_list
                ]
                right_qubits = [
                    coordinates_list.index([1, (b_el[0] + i) % l, (b_el[1] + j) % m])
                    for b_el in b_list
                ]
                gate_lists[round_X][0] += [
                    [left_qubits[0], right_qubits[2]],
                    [right_qubits[0], left_qubits[2]],
                ]
                if round_X == 0:
                    gate_lists[round_X][1] += [
                        [right_qubits[1], left_qubits[0]],
                        [right_qubits[0], left_qubits[1]],
                    ]
                    if simple_detectors:
                        gate_lists[round_X][2] += [[right_qubits[1], right_qubits[0]]]
                        gate_lists[other_round][2] += [[right_qubits[0], right_qubits[1]]]
                        measurement_lists["X"][round_X].append(right_qubits[1])
                        measurement_lists["Z"][other_round].append(right_qubits[1])
                    else:
                        gate_lists[round_X][2] += [[right_qubits[0], right_qubits[1]]]
                        gate_lists[other_round][2] += [[right_qubits[1], right_qubits[0]]]
                        measurement_lists["X"][round_X].append(right_qubits[0])
                        measurement_lists["Z"][other_round].append(right_qubits[0])
                else:
                    gate_lists[round_X][1] += [
                        [left_qubits[0], right_qubits[1]],
                        [left_qubits[1], right_qubits[0]],
                    ]
                    if simple_detectors:
                        gate_lists[round_X][2] += [[left_qubits[1], left_qubits[0]]]
                        gate_lists[other_round][2] += [[left_qubits[0], left_qubits[1]]]
                        measurement_lists["X"][round_X].append(left_qubits[1])
                        measurement_lists["Z"][other_round].append(left_qubits[1])
                    else:
                        gate_lists[round_X][2] += [[left_qubits[0], left_qubits[1]]]
                        gate_lists[other_round][2] += [[left_qubits[1], left_qubits[0]]]
                        measurement_lists["X"][round_X].append(left_qubits[0])
                        measurement_lists["Z"][other_round].append(left_qubits[0])

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

    def _get_sq_morphing_specifications(self, homomorphism_str, circuit_type):
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

        coordinates_list = self._get_real_coordinates_list()

        contraction_round_functions = {
            "X": lambda el: (homomorphism(el) + homomorphism(a_list[0])) % 2,
            "Z": lambda el: (homomorphism(el) + homomorphism(b_list[0])) % 2,
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self._get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(3)] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                round_X = contraction_round_functions["X"](
                    [i, j]
                )  # this is the round in which the X stabiliser labelled by [i,j] is contracting
                left_qubits = [
                    coordinates_list.index([0, (a_el[0] + i) % l, (a_el[1] + j) % m])
                    for a_el in a_list
                ]
                right_qubits = [
                    coordinates_list.index([1, (b_el[0] + i) % l, (b_el[1] + j) % m])
                    for b_el in b_list
                ]
                gate_lists[round_X][0] += [
                    [left_qubits[0], right_qubits[0]],
                    [right_qubits[1], left_qubits[2]],
                ]
                if circuit_type == "centred":
                    gate_lists[round_X][1] += [
                        [left_qubits[0], right_qubits[1]],
                        [left_qubits[1], right_qubits[0]],
                    ]
                    gate_lists[round_X][2] += [
                        [left_qubits[0], right_qubits[2]],
                        [left_qubits[2], right_qubits[0]],
                    ]
                    measurement_lists["X"][round_X].append(left_qubits[0])
                    measurement_lists["Z"][round_X].append(right_qubits[0])
                elif circuit_type == "shifted":
                    gate_lists[round_X][1] += [
                        [left_qubits[0], right_qubits[1]],
                        [left_qubits[2], right_qubits[0]],
                    ]
                    gate_lists[round_X][2] += [
                        [right_qubits[2], left_qubits[0]],
                        [right_qubits[0], left_qubits[1]],
                    ]
                    measurement_lists["X"][round_X].append(right_qubits[2])
                    measurement_lists["Z"][round_X].append(left_qubits[1])
                elif circuit_type == "twisted":
                    gate_lists[round_X][1] += [
                        [left_qubits[0], right_qubits[1]],
                        [left_qubits[1], right_qubits[0]],
                    ]
                    gate_lists[round_X][2] += [
                        [right_qubits[2], left_qubits[0]],
                        [right_qubits[0], left_qubits[2]],
                    ]
                    measurement_lists["X"][round_X].append(right_qubits[2])
                    measurement_lists["Z"][round_X].append(left_qubits[2])
                elif circuit_type == "half-shifted":
                    gate_lists[round_X][1] += [
                        [left_qubits[0], right_qubits[1]],
                        [left_qubits[2], right_qubits[0]],
                    ]
                    gate_lists[round_X][2] += [
                        [right_qubits[2], left_qubits[0]],
                        [left_qubits[1], right_qubits[0]],
                    ]
                    measurement_lists["X"][round_X].append(right_qubits[2])
                    measurement_lists["Z"][round_X].append(right_qubits[0])
                else:
                    raise ValueError(
                        "circuit_type should be shifted, twisted, centred, or half-shifted!"
                    )

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

