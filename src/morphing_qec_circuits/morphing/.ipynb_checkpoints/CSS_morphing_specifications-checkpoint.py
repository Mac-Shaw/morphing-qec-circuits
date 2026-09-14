import numpy as np
import stim
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from surface_sim.models import CircuitNoiseModel, NoiselessModel
from ..misc import nkd_params, uniform_setup, DistanceNoiseModel, circuit_distance
from collections import Counter, defaultdict

class CSSMorphingSpecifications:
    """
    A class for CSS Morphing circuits.

    Attributes: 
        mid_cycle_stabilisers represents all the stabilisers such that mid_cycle_stabilisers[pauli] contains all mid-cycle stabilisers
        stabiliser_contraction_rounds lists the rounds during which each corresponding mid cycle stabiliser is contracting. MUST BE IN ASCENDING ORDER!!!
        gate_lists represents the CNOT gates used in each contraction circuit, such that gate_lists[c_round][layer] is a list of [ctrl, targ] CNOT gates that occur in the layer = layer of the contraction circuit in round = c_round.
        measurement_lists represents the measurements after each contraction circuit, such that measurement_lists[pauli][c_round] contains a list of measurements that occur after the contraction circuit round = c_round in measurement basis = pauli.
        mid_cycle_logicals represents the logical operators of the mid-cycle code, such that mid_cycle_logicals[pauli][i] lists the qubits in the support of the logical operator \bar{pauli}_{i}.
        meta_checks are redundancies (see below) that should be treated as detectors instead of observables in stability experiments. Specifically, each meta check lists the indices of the mid_cycle_stabilisers that products to the identity.
        redundancies is a dict with keys "X" and "Z", and where the items represent the linear combinations of mid_cycle_stabilisers that product to the identity. Specifically, each redundancy lists the indices of the mid_cycle_stabilisers that products to the identity.meta_checks are redundancies (see below) that should be treated as detectors instead of observables in stability experiments.
        coordinates_list is a list of complex numbers representing coordinates for each qubit.
        end_cycle_stabilisers represents the support of each mid_cycle_stabiliser on the data qubits at the end of each round, such that end_cycle_stabilisers[pauli][c_round][k] lists the support at the end of the round c_round of the mid-cycle stabiliser mid_cycle_stabiliser[pauli][k].
        stabiliser_measurement_indices represents the support of each mid-cycle stabiliser on measurements at the end of each round, such that stabiliser_measurement_indices[pauli][c_round][k] lists the indices i of the measurements (given explicitly by measurement_lists[pauli][c_round][i]) during the round c_round that are supported by the mid-cycle stabiliser mid_cycle_stabiliser[pauli][k].
        end_cycle_logicals represents the support of each mid-cycle logical on the data qubits at the end of each round, such that end_cycle_logicals[pauli][c_round][k] lists the support of at the end of the round c_round of the mid-cycle logical mid_cycle_logical[pauli][k].
        logical_measurement_indices represents the support of each mid-cycle logical on measurements at the end of each round, such that logical_measurement_indices[pauli][c_round][k] lists the indices i of the measurements (given explicitly by measurement_lists[pauli][c_round][i]) during the round c_round that are supported by the mid-cycle logical mid_cycle_logical[pauli][k].
    """

    def __init__(
        self,
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        mid_cycle_logicals={},
        meta_checks={},
        redundancies={},
        coordinates_list=[],
        end_cycle_stabilisers={},
        stabiliser_measurement_indices={},
        end_cycle_logicals={},
        logical_measurement_indices={},
        stabiliser_annotations={},
        mid_cycle_nkd = (None, None, None),
        end_cycle_nkd = [],
        circuit_distance = None,
    ):
        self.mid_cycle_stabilisers = mid_cycle_stabilisers
        self.stabiliser_contraction_rounds = stabiliser_contraction_rounds
        self.gate_lists = gate_lists
        self.measurement_lists = measurement_lists
        self.mid_cycle_logicals = mid_cycle_logicals
        self.meta_checks = meta_checks
        self.redundancies = redundancies
        self.coordinates_list = coordinates_list
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_indices = stabiliser_measurement_indices
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_indices = logical_measurement_indices
        self.stabiliser_annotations = stabiliser_annotations
        self.mid_cycle_nkd = (self.n_qubits(), mid_cycle_nkd[1], mid_cycle_nkd[2])
        if len(end_cycle_nkd) == self.n_rounds():
            self.end_cycle_nkd = end_cycle_nkd
        else:
            self.end_cycle_nkd = [(None, None, None) for _ in range(self.n_rounds())]
        self.circuit_distance = circuit_distance

    def to_morphing_specifications(self):
        from .morphing_specifications import MorphingSpecifications
        qubit_dictionary = {str(i): i for i in range(self.n_qubits())}
        noiseless_model = NoiselessModel(qubit_dictionary)

        mid_cycle_stabilisers = []
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                to_append = stim.PauliString(self.n_qubits())
                for j in self.mid_cycle_stabilisers[pauli][i]:
                    to_append[j] = pauli
                mid_cycle_stabilisers.append(to_append)

        stabiliser_contraction_rounds = self.stabiliser_contraction_rounds["X"] + self.stabiliser_contraction_rounds["Z"]
        
        contraction_circuits = []
        for i in range(self.n_rounds()):
            contraction_circuits.append(self.contraction_circuit(noiseless_model, i))

        if self.mid_cycle_logicals == {}:
            mid_cycle_logicals = {}
        else:
            mid_cycle_logicals = {"X": [], "Z": []}
            for pauli in ["X", "Z"]:
                for i in range(len(self.mid_cycle_logicals[pauli])):
                    to_append = stim.PauliString(self.n_qubits())
                    for j in self.mid_cycle_logicals[pauli][i]:
                        to_append[j] = pauli
                    mid_cycle_logicals[pauli].append(to_append)

        if self.meta_checks == {}:
            meta_checks = []
        else:
            meta_checks = deepcopy(self.meta_checks["X"])
            meta_checks += [np.array([i + len(self.stabiliser_contraction_rounds["X"]) for i in meta_check]) for meta_check in self.meta_checks["Z"]]
        
        if self.redundancies == {}:
            redundancies = []
        else:
            redundancies = deepcopy(self.redundancies["X"])
            redundancies += [np.array([i + len(self.stabiliser_contraction_rounds["X"]) for i in redundancy]) for redundancy in self.redundancies["Z"]]

        coordinates_list = self.coordinates_list

        if self.end_cycle_stabilisers == {}:
            end_cycle_stabilisers = []
        else:
            end_cycle_stabilisers = [[] for _ in range(self.n_rounds())]
            for c_round in range(self.n_rounds()):
                for pauli in ["X", "Z"]:
                    for i in range(len(self.end_cycle_stabilisers[pauli][c_round])):
                        to_append = stim.PauliString(self.n_qubits())
                        for j in self.end_cycle_stabilisers[pauli][c_round][i]:
                            to_append[j] = pauli
                        end_cycle_stabilisers[c_round].append(to_append)

        if self.stabiliser_measurement_indices == {}:
            stabiliser_measurement_target_recs = []
        else:
            stabiliser_measurement_target_recs = []
            for c_round in range(self.n_rounds()):
                n_X_meas = len(self.measurement_lists["X"][c_round])
                n_Z_meas = len(self.measurement_lists["Z"][c_round])
                stabiliser_measurement_target_recs.append([
                    [stim.target_rec(i - n_X_meas - n_Z_meas) for i in stabiliser_indices]
                    for stabiliser_indices in self.stabiliser_measurement_indices["X"][c_round]
                ])
                stabiliser_measurement_target_recs[-1] += [
                    [stim.target_rec(i - n_Z_meas) for i in stabiliser_indices]
                    for stabiliser_indices in self.stabiliser_measurement_indices["Z"][c_round]
                ]
            
        if self.end_cycle_logicals == {}:
            end_cycle_logicals = {}
        else:
            end_cycle_logicals = {pauli: [[] for _ in range(self.n_rounds())] for pauli in ["X", "Z"]}
            for c_round in range(self.n_rounds()):
                for pauli in ["X", "Z"]:
                    for i in range(len(self.end_cycle_logicals[pauli][c_round])):
                        to_append = stim.PauliString(self.n_qubits())
                        for j in self.end_cycle_logicals[pauli][c_round][i]:
                            to_append[j] = pauli
                        end_cycle_logicals[pauli][c_round].append(to_append)

        if self.logical_measurement_indices == {}:
            logical_measurement_target_recs = {}
        else:
            logical_measurement_target_recs = {"X": [], "Z": []}
            for c_round in range(self.n_rounds()):
                n_X_meas = len(self.measurement_lists["X"][c_round])
                n_Z_meas = len(self.measurement_lists["Z"][c_round])
                logical_measurement_target_recs["X"].append([
                    [stim.target_rec(i - n_X_meas - n_Z_meas) for i in logical_indices]
                    for logical_indices in self.logical_measurement_indices["X"][c_round]
                ])
                logical_measurement_target_recs["Z"].append([
                    [stim.target_rec(i - n_Z_meas) for i in logical_indices]
                    for logical_indices in self.logical_measurement_indices["Z"][c_round]
                ])

        if self.stabiliser_annotations == {}:
            stabiliser_annotations = []
        else:
            stabiliser_annotations = self.stabiliser_annotations["X"] + self.stabiliser_annotations["Z"]
            
        mid_cycle_nkd = self.mid_cycle_nkd
        end_cycle_nkd = deepcopy(self.end_cycle_nkd)
        circuit_distance = self.circuit_distance

        return MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            contraction_circuits,
            mid_cycle_logicals = mid_cycle_logicals,
            meta_checks = meta_checks,
            redundancies = redundancies,
            coordinates_list = coordinates_list,
            end_cycle_stabilisers = end_cycle_stabilisers,
            stabiliser_measurement_target_recs = stabiliser_measurement_target_recs,
            end_cycle_logicals = end_cycle_logicals,
            logical_measurement_target_recs = logical_measurement_target_recs,
            stabiliser_annotations = stabiliser_annotations,
            mid_cycle_nkd = mid_cycle_nkd,
            end_cycle_nkd = end_cycle_nkd,
            circuit_distance = circuit_distance,
        )

    def __repr__(self):
        """
        Returns a string that specifies the morphing specifications.
        """
        cls = self.__class__.__name__
        mid_cycle_nkd_str = f'[[{self.mid_cycle_nkd[0]},{"?" if self.mid_cycle_nkd[1] == None else self.mid_cycle_nkd[1]},{"?" if self.mid_cycle_nkd[2] == None else self.mid_cycle_nkd[2]}]]'
        end_cycle_nkd_strs = [f'[[{self.end_cycle_nkd[c_round][0]},{"?" if self.end_cycle_nkd[c_round][1] == None else self.end_cycle_nkd[c_round][1]},{"?" if self.end_cycle_nkd[c_round][2] == None else self.end_cycle_nkd[c_round][2]}]]' for c_round in range(self.n_rounds())]
        return f'{cls}(\nmid_cycle_stabilisers={self.mid_cycle_stabilisers},\nstabiliser_contraction_rounds={self.stabiliser_contraction_rounds},\ngate_lists={self.gate_lists},\nmeasurement_lists={self.measurement_lists},\nmid-cycle [[n,k,d]]={mid_cycle_nkd_str}\nend-cycle [[n,k,d]]={end_cycle_nkd_strs}\ncircuit distance={self.circuit_distance}\n)'
        
    def n_qubits(self):
        """
        Returns the number of qubits in the mid-cycle code.
        """
        return 1 + max(
            [
                qubit
                for pauli in ["X", "Z"]
                for stabiliser in self.mid_cycle_stabilisers[pauli]
                for qubit in stabiliser
            ]
        )

    def n_rounds(self):
        """
        Returns the number of contraction rounds of the morphing circuit.

        Raises:
            ValueError: The number of rounds is inconsistent between the gate and measurement lists.

        Returns:
            The number of contraction rounds of the morphing circuit.
        """
        num_rounds = [
            len(rounds_list)
            for rounds_list in [
                self.gate_lists,
                self.measurement_lists["X"],
                self.measurement_lists["Z"],
            ]
        ]
        if num_rounds.count(num_rounds[0]) == len(num_rounds):
            return num_rounds[0]
        else:
            raise ValueError(
                """The number of rounds is inconsistent between the gate and measurement lists."""
            )

    def n_redundancies(self):
        """
        Returns dictionary the number of redundancies between the X and Z stabilisers.
        """
        h = self.parity_check_matrices()
        n_redundancies = {pauli: len(h[pauli].T.null_space()) for pauli in ["X", "Z"]}
        return n_redundancies

    def n_stabilisers(self):
        """
        Returns a dictionary with keys "X" and "Z" and items the number of X (respectively, Z) type stabilisers.
        """
        return {pauli: len(self.mid_cycle_stabilisers[pauli]) for pauli in ["X", "Z"]}

    def n_meta_checks(self):
        """
        Returns a dictionary with keys "X" and "Z" and items the number of X and Z type meta-checks.
        """
        return {pauli: len(self.meta_checks[pauli]) for pauli in ["X", "Z"]}

    def n_logicals(self, recalculate = False):
        """
        Returns the number of logical qubits in the circuit.

        Arguments:
            recalculate is a bool that if set to True will mean that the number of logical qubits is recalculated, rather than using a stored value.

        Returns:
            The number of logical qubits.
        """
        if recalculate or self.mid_cycle_nkd[1] == None:
            n_redundancies = self.n_redundancies()
            n_total_stabilisers = self.n_stabilisers()
            to_return = (
                self.n_qubits()
                - sum(n_total_stabilisers.values())
                + sum(n_redundancies.values())
            )
            self.mid_cycle_nkd = (self.mid_cycle_nkd[0], to_return, self.mid_cycle_nkd[2])
            return to_return
        else:
            return self.mid_cycle_nkd[1]
        

    def check_consistent_gates(self):
        """
        Checks whether the gates in the morphing circuit are simultaneously-executable.

        Raises:
            Exception: gate_lists do not describe a simultaneously-executable set of gates!
        """
        for c_round_gates in self.gate_lists:
            for gate_list in c_round_gates:
                test_list = [qubit for gate in gate_list for qubit in gate]
                test_set = set(test_list)
                if len(test_list) != len(test_set):
                    raise Exception("gate_lists do not describe a simultaneously-executable set of gates!")

    def mid_cycle_nkd_params(
        self, recalculate = False, **kwargs,
    ):
        """
        Returns the number of physical qubits (n), number of logical qubits (k), and distance (d) of the mid-cycle code, and sets self.mid_cycle_nkd to the returned value.

        Arguments:
            recalculate is a bool that if set to True will mean that k and d are recalculated, rather than using stored values.
            kwargs are passed into distance_functions.nkd_params().

        Returns:
            A length-3 tuple of Ints (n,k,d) representing the mid-cycle code parameters.
        """
        if recalculate or None in self.mid_cycle_nkd:
            hs = self.parity_check_matrices()
            to_return = nkd_params(
                hs,
                **kwargs,
            )
            self.mid_cycle_nkd = to_return
            return to_return
        else:
            return self.mid_cycle_nkd

    def end_cycle_nkd_params(
        self,
        recalculate = False,
        check_all_end_cycle_codes=True,
        **kwargs,
    ):
        """
        Returns the number of physical qubits (n), number of logical qubits (k), and distance (d) of each end-cycle code code, and sets self.end_cycle_nkd to the returned value.

        Arguments:
            recalculate is a bool that if set to True will mean that k and d are recalculated, rather than using stored values.
            check_all_end_cycle_codes is a bool that if set to False will assume that all end-cycle codes have the same n,k,d values. This is the case, for example, for homomorphism-based 
                morphing circuits for Abelian 2BGA codes.
            kwargs are passed into distance_functions.nkd_params().

        Returns:
            A list of length-3 tuple of Ints (n,k,d) representing each of the end-cycle code parameters.
        """
        need_to_calculate = any([None in params for params in self.end_cycle_nkd])
        if recalculate or need_to_calculate:
            hs = self.end_cycle_parity_check_matrices()
            if check_all_end_cycle_codes:
                nkd_params_list = []
                for i in range(self.n_rounds()):
                    nkd_params_list.append(
                        nkd_params(
                            hs[i],
                            **kwargs,
                        )
                    )
                self.end_cycle_nkd = nkd_params_list
                return nkd_params_list
            else:
                first_nkd_params = nkd_params(
                    hs[0],
                    **kwargs,
                )
                to_return = [first_nkd_params for _ in range(self.n_rounds())]
                self.end_cycle_nkd = to_return
                return to_return
        else:
            return self.end_cycle_nkd

    def parity_check_matrices(self):
        """
        Returns a dictionary h with keys "X" and "Z" and items corresponding to the parity-check matrix for the X or Z type stabilisers of the mid-cycle code.
        """
        h_int = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            for stabiliser in self.mid_cycle_stabilisers[pauli]:
                new_row = np.zeros((self.n_qubits())).astype(int)
                for qubit in stabiliser:
                    new_row[qubit] = 1
                h_int[pauli].append(new_row)
                
        hs = {pauli: galois.GF(2)(h_int[pauli]) for pauli in ["X", "Z"]}
        comm_rel = hs["X"] @ hs["Z"].T
        if np.any(comm_rel):
            raise Exception("mid-cycle stabilisers do not commute!")

        return hs

    def end_cycle_parity_check_matrices(self):
        """
        Returns a list of dictionaries with keys "X" and "Z" and items corresponding to the parity-check matrix for the X or Z type stabilisers of each end-cycle code.
        """
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        hs = [{pauli: [] for pauli in ["X", "Z"]} for i in range(n_rounds)]
        if self.end_cycle_stabilisers == {}:
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_indices()
        for i in range(n_rounds):
            left_over_qubits = [
                j
                for j in range(n_qubits)
                if j
                not in self.measurement_lists["X"][i] + self.measurement_lists["Z"][i]
            ]
            for pauli in ["X", "Z"]:
                h_int = []
                for stabiliser in self.end_cycle_stabilisers[pauli][i]:
                    new_row = np.zeros((len(left_over_qubits))).astype(int)
                    for qubit in stabiliser:
                        if qubit in left_over_qubits:
                            new_row[left_over_qubits.index(qubit)] = 1
                    if new_row.any():
                        h_int.append(new_row)
                if len(h_int) == 0:
                    hs[i][pauli] = galois.GF(2)(np.zeros((1,len(left_over_qubits)),dtype = int))
                else:
                    hs[i][pauli] = galois.GF(2)(h_int)
        
            comm_rel = hs[i]["X"] @ hs[i]["Z"].T
            if np.any(comm_rel):
                raise Exception("end-cycle stabilisers do not commute!")
            
        return hs

    def compile_mid_cycle_logicals(self):
        """
        Calculates the mid-cycle logical operators of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_logicals.
        """
        h = self.parity_check_matrices()
        h_kernel = {pauli: h[pauli].null_space() for pauli in ["X", "Z"]}
        stabilisers_to_logicals = {pauli: h[pauli].row_reduce() for pauli in ["X", "Z"]}

        mid_cycle_logicals = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            other_pauli = "Z" if pauli == "X" else "X"
            for i in range(h_kernel[other_pauli].shape[0]):
                new_vectors = np.concatenate(
                    (stabilisers_to_logicals[pauli], [h_kernel[other_pauli][i]]), axis=0
                )
                if np.linalg.matrix_rank(new_vectors) != np.linalg.matrix_rank(
                    stabilisers_to_logicals[pauli]
                ):
                    stabilisers_to_logicals[pauli] = new_vectors
                    mid_cycle_logicals[pauli].append(
                        np.nonzero(h_kernel[other_pauli][i])[0].tolist()
                    )

        self.mid_cycle_logicals = mid_cycle_logicals

    def compile_redundancies(self):
        """
        Calculates the redundancies between the stabilisers of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_redundancies.
        (Note that these redundancies do not change throughout the circuit since the identity operator is preserved by all unitary gates)
        """
        h = self.parity_check_matrices()
        hT_kernel = {pauli: h[pauli].T.null_space() for pauli in ["X", "Z"]}
        n_redundancies = {pauli: len(hT_kernel[pauli]) for pauli in ["X", "Z"]}

        mid_cycle_redundancies = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            for redundancy in hT_kernel[pauli]:
                redundant_stabilisers = np.nonzero(redundancy)[0]
                mid_cycle_redundancies[pauli].append(redundant_stabilisers)

        self.redundancies = mid_cycle_redundancies

    def _mid_cycle_to_pre_measurement(self, mid_cycle_operator, pauli, c_round):
        """
        Manually propagates a multi-qubit Pauli operator (either a product of X's or Z's) from the mid-cycle code to the pre-measurement code in the round c_round. In principle this could be done using for example stim.PauliString.before() but is kept for backwards compatibility.
        """
        if pauli not in ["X", "Z"]:
            raise ValueError("""pauli should be either the string 'X' or 'Z'. """)

        pre_measurement_operator = deepcopy(mid_cycle_operator)
        for gate_layer in self.gate_lists[c_round]:
            for gate in gate_layer:
                # gate[0] is the control of the CNOT and gate[1] is the target.
                # When propagating an X operator through a CNOT gate, nothing happens if the ctrl of the CNOT (the "check_qubit") does not overlap with
                #    the operator. If it does overlap, then the operator is multiplied by X on the targ of the CNOT (the "expansion_qubit").
                # Vice-versa for if the basis is Z.
                if pauli == "X":
                    check_qubit = gate[0]
                    expansion_qubit = gate[1]
                else:
                    check_qubit = gate[1]
                    expansion_qubit = gate[0]
                if check_qubit in pre_measurement_operator:
                    if expansion_qubit in pre_measurement_operator:
                        pre_measurement_operator.remove(expansion_qubit)
                    else:
                        pre_measurement_operator.append(expansion_qubit)
        return pre_measurement_operator

    def _pre_measurement_to_end_cycle_and_measurement_indices(
        self, pre_measurement_operator, pauli, c_round
    ):
        """
        Manually propagates a multi-qubit Pauli operator (either a product of X's or Z's) from the pre-measurement code in the round c_round to the end-cycle code, including a list of indices representing the measurements in measurement_lists[c_round] that are included in the pre-measurement support of the operator. In principle this could be done using for example stim.PauliString.before() but is kept for backwards compatibility.
        """
        end_cycle_operator = deepcopy(pre_measurement_operator)
        measurement_indices = []
        this_measurement_list = self.measurement_lists[pauli][c_round]
        for i in range(len(this_measurement_list)):
            # checks the measurements in the expanded c_round that the expanded detector has support over and adds their measurement index to the
            #   list of measurements in the detector.
            measured_qubit = this_measurement_list[i]
            if measured_qubit in pre_measurement_operator:
                measurement_indices.append(i)
                end_cycle_operator.remove(measured_qubit)
                
        other_pauli = {"X": "Z", "Z": "X"}[pauli]
        other_measurement_list = self.measurement_lists[other_pauli][c_round]
        for qubit in end_cycle_operator:
            if qubit in other_measurement_list:
                raise Exception("A stabiliser anticommuted with a measurement!")
        return end_cycle_operator, measurement_indices

    def _mid_cycle_to_end_cycle_and_measurement_indices(
        self, mid_cycle_operator, pauli, c_round
    ):
        """
        Manually propagates a multi-qubit Pauli operator (either a product of X's or Z's) from the pre-measurement code in the round c_round to the end-cycle code, including a list of indices representing the measurements in measurement_lists[c_round] that are included in the pre-measurement support of the operator. In principle this could be done using for example stim.PauliString.before() but is kept for backwards compatibility.
        """
        pre_measurement_operator = self._mid_cycle_to_pre_measurement(
            mid_cycle_operator, pauli, c_round
        )
        return self._pre_measurement_to_end_cycle_and_measurement_indices(
            pre_measurement_operator, pauli, c_round
        )

    def compile_end_cycle_stabilisers_and_stabiliser_measurement_indices(self):
        """
        Calculates the end-cycle stabilisers and measurement indices and sets them to self.end_cycle_stabilisers and self.stabiliser_measurement_indices (respectively).
        """

        self.check_consistent_gates()
        n_rounds = self.n_rounds()
        end_cycle_stabilisers = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        stabiliser_measurement_indices = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        for pauli in ["X", "Z"]:
            for mid_cycle_stabiliser in self.mid_cycle_stabilisers[pauli]:
                for i in range(n_rounds):
                    end_cycle_stabiliser, measurement_indices = (
                        self._mid_cycle_to_end_cycle_and_measurement_indices(
                            mid_cycle_stabiliser, pauli, i
                        )
                    )
                    end_cycle_stabilisers[pauli][i].append(end_cycle_stabiliser)
                    stabiliser_measurement_indices[pauli][i].append(measurement_indices)
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_indices = stabiliser_measurement_indices

    def compile_end_cycle_logicals_and_logical_measurement_indices(self):
        """Calculates the end-cycle logical operators and measurement indices and sets them to self.end_cycle_logicals and self.logical_measurement_indices (respectively)."""

        self.check_consistent_gates()
        n_rounds = self.n_rounds()
        end_cycle_logicals = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        logical_measurement_indices = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        for pauli in ["X", "Z"]:
            for mid_cycle_logical in self.mid_cycle_logicals[pauli]:
                for i in range(n_rounds):
                    end_cycle_logical, measurement_indices = (
                        self._mid_cycle_to_end_cycle_and_measurement_indices(
                            mid_cycle_logical, pauli, i
                        )
                    )
                    end_cycle_logicals[pauli][i].append(end_cycle_logical)
                    logical_measurement_indices[pauli][i].append(measurement_indices)
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_indices = logical_measurement_indices

    def compile_all(self, check_compiled=False):
        """Compiles the mid-cycle logical operators, mid-cycle redundancies, end-cycle stabilisers, stabiliser measurement indices, end-cycle logical operators, and logical measurement indices, and assigns them to the corresponding attributes.
        
        Args:
            check_compiled is a bool that if True only compiles the attributes that have not yet been computed.
        """
        if (not check_compiled) or self.mid_cycle_logicals == {}:
            self.compile_mid_cycle_logicals()
        if (not check_compiled) or self.redundancies == {}:
            self.compile_redundancies()
        if (
            (not check_compiled)
            or self.end_cycle_stabilisers == {}
            or self.stabiliser_measurement_indices == {}
        ):
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_indices()
        if (
            (not check_compiled)
            or self.end_cycle_logicals == {}
            or self.logical_measurement_indices == {}
        ):
            self.compile_end_cycle_logicals_and_logical_measurement_indices()

    def get_morphing_incoming_XZ_memory_circuit(
        self,
        p,
        which_round = 0,
        round_coords = False,
    ):
        """
        Returns a stim circuit for a memory experiment under incoming noise.
        The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with n_rounds rounds
            of ideal contraction circuits, with one layer of data-qubit errors in between during the end-cycle code specified by which_round.
        
        Arguments:
            p is a float representing the probability of a Pauli error occurring in a single-qubit depolarising noise channel that is applied to each data qubit.
            which_round is an Int specifying which end-cycle code to add the incoming noise to.
            round_coords is a bool that is passed to self.coordinate_circuit.
        
        Returns:
            A stim circuit for the memory experiment.
        """
        T = 1

        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()

        self.compile_all(check_compiled=True)

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
        noiseless_model = NoiselessModel(all_qubits_dictionary)
        tick = noiseless_model.tick()

        detector_circuits = self._mid_experiment_detector_circuits()
        observable_circuits = self._mid_experiment_logical_observable_circuits()

        circ = self.coordinate_circuit(n_extra_qubits=n_logicals, round_coords = round_coords)
        
        # The noiseless part of the circuit starts in the end-cycle code \tilde{C}_{which_round+1} and finishes in the end-cycle code \tilde{C}_{which_round}
        # The circuit starts with the expanding gates of c_round 0, then the contracting gates of c_round 1.
        # The contraction gates of final_round will just have been completed at the end of the circuit.
        
        circ += noiseless_model.reset_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.reset_z([str(i) for i in range(n_qubits)])

        first_round = (which_round + 1) % n_rounds

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][first_round][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        circ += tick
        
        for i in range(which_round + 1, which_round + n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, (i + 1 / 2) % n_rounds)
            circ += observable_circuits[i % n_rounds]

        # perfect Bell state prepared, now add the incoming noise.

        circ.append("DEPOLARIZE1", range(n_qubits), p)

        # perfect Bell measurements now

        for i in range(which_round, which_round - 1 + n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, (i + 1 / 2) % n_rounds)
            circ += detector_circuits[i % n_rounds]
            circ += observable_circuits[i % n_rounds]

        final_round = (which_round - 1) % n_rounds

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][final_round][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        circ += tick
        circ += noiseless_model.measure_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.measure_z([str(i) for i in range(n_qubits)])
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(i - n_qubits - n_logicals)], i
            )
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(j - n_qubits)
                    for j in self.end_cycle_logicals["Z"][final_round][i]
                ],
                i + n_logicals,
            )
        return circ

    def get_morphing_XZ_memory_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
        round_coords = False,
    ):
        """
        Returns a stim circuit for a memory experiment under incoming noise.
        The memory experiment contains observables for all the logical operators of the code (both X and Z). It begins and ends with one round of ideal QEC, with T rounds of noisy QEC in between.
        
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the memory experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model
                object that defines the noise model of the memory experiment.
            which_round is an Int specifying which end-cycle code to add the incoming noise to.
            round_coords is a bool that is passed to self.coordinate_circuit.

        Raises:
            ValueError: T is too smol for the mol, make him bigger
        
        Returns:
            A stim circuit for the memory experiment.
        """
        if T < 1:
            raise ValueError("T is too smol for the mol, make him bigger")

        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()

        self.compile_all(check_compiled=True)

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)
        noiseless_model = NoiselessModel(all_qubits_dictionary)
        tick = noiseless_model.tick()

        detector_circuits = self._mid_experiment_detector_circuits()
        observable_circuits = self._mid_experiment_logical_observable_circuits()

        circ = self.coordinate_circuit(n_extra_qubits=n_logicals, round_coords = round_coords)
        
        # The noiseless part of the circuit starts in the end-cycle code \tilde{C}_{0} and finishes in the end-cycle code \tilde{C}_{-1}
        # The circuit starts with the expanding gates of c_round 0, then the contracting gates of c_round 1.
        # The contraction gates of final_round will just have been completed at the end of the circuit.
        
        circ += noiseless_model.reset_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.reset_z([str(i) for i in range(n_qubits)])

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][0][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        circ += tick
        
        for i in range(1, n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, i - 1 / 2)
            circ += observable_circuits[i]

        # perfect Bell state prepared, now run T cycles

        # The noisy part of the circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        n_repeats = T // n_rounds
        n_left_over_rounds = T % n_rounds

        repeated_circ = stim.Circuit()
        for i in range(n_rounds):
            repeated_circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            repeated_circ += detector_circuits[i]
            repeated_circ += observable_circuits[i]

        circ += repeated_circ * n_repeats

        for i in range(n_left_over_rounds):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits[i]
            circ += observable_circuits[i]

        # perfect Bell measurements now

        for i in range(n_left_over_rounds, n_left_over_rounds + n_rounds - 1):
            circ += self.morphing_QEC_round(noiseless_model, i - 1 / 2)
            circ += detector_circuits[i % n_rounds]
            circ += observable_circuits[i % n_rounds]

        final_round = (n_left_over_rounds - 2) % n_rounds

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][final_round][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        circ += tick
        circ += noiseless_model.measure_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.measure_z([str(i) for i in range(n_qubits)])
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(i - n_qubits - n_logicals)], i
            )
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(j - n_qubits)
                    for j in self.end_cycle_logicals["Z"][final_round][i]
                ],
                i + n_logicals,
            )
        return circ

    def calculate_circuit_distance(
        self,
        recalculate = False,
        cutoff = np.inf,
        T_override = None,
    ):
        """
        Returns the exact (spatial) circuit distance of a morphing circuit and sets it to self.circuit_distance.
        The end-cycle distances are automatically pre-calculated here (or taken from stored values) to provide an upper-bound on the
            circuit-distance.

        Arguments:
            recalculate is a bool that if set to True will mean that the circuit distance is recalculated, rather than using a stored value.
            cutoff is an Int (or np.inf) setting an upper-bound on the calculated circuit distance. If the circuit distance is greater than or
                equal to cutoff, the algorithm will simply return cutoff.
            T_override is an Int (or None) that if set to an integer sets the number of measurement rounds to check the circuit-distance for. If
                set to None, then the default value of T is chosen to be large enough to guarantee that the circuit distance is exact (and not
                simply an upper-bound).

        Returns:
            The (spatial) circuit distance of the morphing circuit.
        """
        if recalculate or self.circuit_distance == None:
            noise_setup = uniform_setup(0.001)
            self.end_cycle_nkd_params()
            calculated_distances = [nkd[2] for nkd in ([self.mid_cycle_nkd] + self.end_cycle_nkd) if nkd[2] != None]
            cutoff = min(calculated_distances + [cutoff])
            
            if T_override != None:
                T = T_override
            else:
                if self.n_rounds() == 2:
                    #See Proposition 1 of arxiv:2604.09797
                    T = 1
                else:
                    #the maximum span of a detecting region is self.n_rounds() measurement rounds, so a single error can trigger detectors no further
                    # away than self.n_rounds() - 1.
                    # Any circuit-level error with weight < cutoff can therefore span at most (cutoff-1) * (self.n_rounds()-1) measurement rounds.
                    T = (cutoff - 1) * (self.n_rounds() - 1)
            
            noise_model_generator = lambda qubit_dictionary, noise_setup: DistanceNoiseModel(qubit_dictionary, noise_setup, "X")
            
            stim_circuit_X_errors = self.get_morphing_XZ_memory_circuit(
                noise_setup,
                T,
                noise_model_generator = noise_model_generator,
            )
    
            X_circuit_distance = circuit_distance(stim_circuit_X_errors, error_basis = "X", cutoff = cutoff)
            
            noise_model_generator = lambda qubit_dictionary, noise_setup: DistanceNoiseModel(qubit_dictionary, noise_setup, "Z")
            
            stim_circuit_Z_errors = self.get_morphing_XZ_memory_circuit(
                noise_setup,
                T,
                noise_model_generator = noise_model_generator,
            )
    
            to_return = circuit_distance(stim_circuit_Z_errors, error_basis = "Z", cutoff = X_circuit_distance)
            self.circuit_distance = to_return
            return to_return
        else:
            return self.circuit_distance

    def get_morphing_memory_circuit(
        self,
        noise_setup,
        T,
        basis = "Z",
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
        round_coords = False,
    ):
        """
        Returns a stim circuit for a memory experiment in a given basis (if basis = 'both', then get_morphing_XZ_memory_circuit is called instead).
        The memory experiment contains observables for just the logical operators of type given by basis. It begins by initialising all data qubits (noisily) in the basis of the experiment and ends by measuring these data qubits (noisily) in the same basis, with T rounds of noisy QEC in between.
        
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the memory experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model
                object that defines the noise model of the memory experiment.
            basis is a Str either 'X', 'Z' or 'both', determining the basis of the memory experiment. If basis = 'both' then self.get_morphing_XZ_memory_circuit() is called.
            round_coords is a bool that is passed to self.coordinate_circuit.

        Raises:
            ValueError: T is too smol for the mol, make him bigger
            ValueError: basis needs to be X, Z, or both!
        
        Returns:
            A stim circuit for the memory experiment.
        """

        if basis == "both":
            return self.get_morphing_XZ_memory_circuit(noise_setup, T, noise_model_generator = noise_model_generator, round_coords = round_coords)
            
        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()
        
        if T <= n_rounds:
            raise ValueError("T is too smol for the mol, make him bigger")

        self.compile_all(check_compiled=True)

        # The circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)

        detector_circuits_i = self._initial_memory_detector_circuits(basis)
        detector_circuits = self._mid_experiment_detector_circuits()
        observable_circuits = self._mid_experiment_logical_observable_circuits(basis = basis)

        circ = self.coordinate_circuit(n_extra_qubits=0, round_coords = round_coords)

        if basis not in ["X", "Z"]:
            raise ValueError("basis needs to be X, Z, or both!")
        
        # Run T cycles

        circ += self.morphing_QEC_round(noise_model, -1/2, data_qubit_reset_basis = basis)
        circ += detector_circuits_i[0]
        circ += observable_circuits[0]

        #special detectors for the first (n_rounds-1) rounds
        for i in range(1,n_rounds - 1):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits_i[i]
            circ += observable_circuits[i]

        circ += self.morphing_QEC_round(noise_model, n_rounds - 1 - 1 / 2)
        circ += detector_circuits[-1]
        circ += observable_circuits[-1]

        n_repeats = ((T-1) // n_rounds) - 1
        n_left_over_rounds = ((T-1) % n_rounds) + 1# makes sure n_left_over_rounds > 0, so that I can add the data qubit measurements in the last round.
        
        repeated_circ = stim.Circuit()
        for i in range(n_rounds):
            repeated_circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            repeated_circ += detector_circuits[i]
            repeated_circ += observable_circuits[i]

        circ += repeated_circ * n_repeats
        
        for i in range(n_left_over_rounds-1):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits[i]
            circ += observable_circuits[i]

        final_round = (n_left_over_rounds - 1) % n_rounds
        
        circ += self.morphing_QEC_round(noise_model, final_round - 1/2, data_qubit_measurement_basis = basis)
        final_round_offset = self.n_qubits() - len(self.measurement_lists["X"][final_round]) - len(self.measurement_lists["Z"][final_round])
        final_round_detector_circuit = stim.Circuit()
        for instruction in detector_circuits[final_round]:
            final_round_detector_circuit.append(instruction.name, [stim.target_rec(i.value - final_round_offset) for i in instruction.targets_copy()], instruction.gate_args_copy())
        circ += final_round_detector_circuit
        final_round_observable_circuits = stim.Circuit()
        for instruction in observable_circuits[final_round]:
            final_round_observable_circuits.append(instruction.name, [stim.target_rec(i.value - final_round_offset) for i in instruction.targets_copy()], instruction.gate_args_copy())
        circ += final_round_observable_circuits

        circ += self._final_memory_extra_detector_circuits(final_round, basis)

        final_data_qubit_list = [i for i in range(self.n_qubits()) if i not in (self.measurement_lists["X"][final_round] + self.measurement_lists["Z"][final_round])]
            
        offset = len(final_data_qubit_list)
        
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(final_data_qubit_list.index(j) - offset)
                    for j in self.end_cycle_logicals[basis][final_round][i]
                ],
                i,
            )
        return circ

    def get_morphing_stability_circuit(self, *args, **kwargs):
        """
        An alias for get_morphing_XZ_stability_circuit.
        """
        return self.get_morphing_XZ_stability_circuit(*args, **kwargs)
    
    def get_morphing_XZ_stability_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
        round_coords = False,
    ):
        """
        Returns a stim circuit for a stability experiment.
        The stability experiment contains observables for each redundancy in the (mid-cycle) stabilisers. It begins by initialising all data qubits (noisily) in the Y-basis and ends by measuring these data qubits (noisily) in the Y-basis, with T rounds of noisy QEC in between.
        Only supported for a number of rounds T that is at least as large as n_rounds.
        
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the stability experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model
                object that defines the noise model of the memory experiment.
            round_coords is a bool that is passed to self.coordinate_circuit.

        Raises:
            ValueError: T is too smol for the mol, make him bigger
        
        Returns:
            A stim circuit for the stability experiment.
        """

        n_qubits = self.n_qubits()
        n_redundancies = self.n_redundancies()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()

        if T < n_rounds:
            raise ValueError("T is too smol for the mol, make him bigger")

        self.compile_all(check_compiled=True)

        final_round = (T - 1) % n_rounds

        # The circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)

        detector_circuits_i = self._initial_stability_detector_circuits()
        detector_circuits = self._mid_experiment_detector_circuits()
        observable_circuits = self._initial_redundancy_observable_circuits()
        #for convenience, when observable_redundancies is not trivial, observable_circuits contains also the detectors derived from non-observable redundancies.

        circ = self.coordinate_circuit(n_extra_qubits=0, round_coords = round_coords)

        circ += noise_model.reset_y([str(i) for i in range(n_qubits)])

        circ += noise_model.tick()

        # first (n_rounds - 1) rounds need to use initial detectors (not mid-circuit detectors)
        # first (n_rounds - 1) rounds have observables

        for i in range(n_rounds - 1):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits_i[i]
            circ += observable_circuits[i]

        circ += self.morphing_QEC_round(noise_model, n_rounds - 1 - 1 / 2)
        circ += detector_circuits[-1]

        n_repeats = (T // n_rounds) - 1
        n_left_over_rounds = T % n_rounds

        repeated_circ = stim.Circuit()
        for i in range(n_rounds):
            repeated_circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            repeated_circ += detector_circuits[i]

        circ += repeated_circ * n_repeats

        for i in range(n_left_over_rounds):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits[i]
        
        circ += noise_model.measure_y([str(i) for i in range(n_qubits)])
        return circ

    def get_morphing_XZ_stability_circuit_XYZ_permutation(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
        round_coords = False,
    ):
        """
        Returns a stim circuit for a stability experiment where each time the code is in the mid-cycle code, a transversal HS^dag gate is performed.
        This method therefore only works for specfications with the following properties:
            - the mid-cycle code is self-dual (i.e. the X and Z stabilisers coincide), so that (HS^dag)^n is a logical gate. (It does not need to logically implement HS^dag).
            - the morphing circuit has two contraction rounds.
            - for a given pair of X and Z stabilisers that are identified under the self-duality, one of them contracts in the first round and the other contracts in the second round.
        This is primarily intended for use on any of the hex-grid colour code morphing circuits (not square grid), but could be used also for other self-dual codes satisfying the above properties such as self-dual BB codes with higher weight checks.
        The stability experiment contains observables for each redundancy in the (mid-cycle) stabilisers. It begins by initialising all data qubits (noisily) in the Y-basis and ends by measuring these data qubits (noisily) in the Y-basis, with T rounds of noisy QEC in between. At each mid-cycle time-step an (HS^dag) gate is applied on each qubit. Because this gate interchanges contracting and expanding stabilisers, the same contraction circuit is used every QEC round.
        Only supported for a number of rounds T that is at least as large as n_rounds.
        
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            T is the number of noisy QEC rounds in the stability experiment.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model
                object that defines the noise model of the memory experiment.
            round_coords is a bool that is passed to self.coordinate_circuit.

        Raises:
            ValueError: T is too smol for the mol, make him bigger
        
        Returns:
            A stim circuit for the stability experiment.
        """

        n_qubits = self.n_qubits()
        n_redundancies = self.n_redundancies()
        #n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()

        if T < 1:
            raise ValueError("T is too smol for the mol, make him bigger")

        self.compile_all(check_compiled=True)

        # The circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)

        detector_circuit = self._detector_circuit_XYZ_permutation()
        observable_circuits = self._initial_redundancy_observable_circuits()# redundancies are still redundancies even if there is a logical gate!

        circ = self.coordinate_circuit(n_extra_qubits=0, round_coords = round_coords)

        circ += noise_model.reset_y([str(i) for i in range(n_qubits)])

        circ += noise_model.tick()
        
        circ += self.morphing_QEC_round_XYZ_permutation(noise_model)
        circ += observable_circuits[0]

        n_repeats = T - 1

        repeated_circ = stim.Circuit()
        repeated_circ += self.morphing_QEC_round_XYZ_permutation(noise_model)
        repeated_circ += detector_circuit

        circ += repeated_circ * n_repeats
        
        circ += noise_model.measure_y([str(i) for i in range(n_qubits)])
        return circ

    def coordinate_circuit(self, n_extra_qubits=0, round_coords = False):
        """
        Returns a stim circuit that defines qubit coordinates for a memory or stability experiment using the self.coordinates_list attribute.

        Arguments:
            n_extra_qubits is an Int that adds extra qubits alongside the qubits in the morphing circuit. Can be used, for example, to define "dummy" logical qubits for memory experiments conducted in both the X and Z basis.
            round_coords is a bool, if True it rounds all of the coordinates to the nearest integer. Can be useful for editing the resulting circuit in crumble.

        Returns:
            A stim circuit defining the coordinates of the qubits.
        """
        circuit_str = ""
        coordinates_list = self.coordinates_list

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            if round_coords:
                min_x = min(round(coord.real), min_x)
                max_x = max(round(coord.real), max_x)
                min_y = min(round(coord.imag), min_y)
                max_y = max(round(coord.imag), max_y)
            else:
                min_x = min(coord.real, min_x)
                max_x = max(coord.real, max_x)
                min_y = min(coord.imag, min_y)
                max_y = max(coord.imag, max_y)
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            if round_coords:
                circuit_str += f"QUBIT_COORDS({-min_x + round(coord.real)}, {(max_y - round(coord.imag))}) {i}\n"
            else:
                circuit_str += f"QUBIT_COORDS({-min_x + coord.real}, {(max_y - coord.imag)}) {i}\n"
            # the minus makes sure y always is pointing up.
            # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)
        for j in range(n_extra_qubits):
            circuit_str += f"QUBIT_COORDS({-min_x + max_x + 1 + (j // (-min_y+max_y+1))}, {-min_y + max_y - (j % (-min_y+max_y+1))}) {len(coordinates_list) + j}\n"

        return stim.Circuit(circuit_str)

    def contraction_circuit(self, noise_model, contraction_round):
        """
        Returns a stim circuit for a contraction circuit, from the mid-cycle code to the end-cycle code.

        Arguments:
            noise_model is a surface_sim.Model object that defines the noise model to be used in the circuit.
            contraction_round is an Int that determines which contraction circuits to use.

        eturns:
            A stim circuit for the contraction circuit.
        """
        n_qubits = self.n_qubits()
        circ = stim.Circuit()
        contracting_gates = self.gate_lists[contraction_round]
        for j in range(len(contracting_gates)):
            cnot_list = [i for gate in contracting_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += noise_model.tick()
        meas_lists = defaultdict(list)
        for pauli in ["X", "Z"]:
            meas_lists[pauli] += self.measurement_lists[pauli][contraction_round]
        meas_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in meas_lists["X"]) and (not i in meas_lists["Z"])
        ]
        circ += noise_model.measure_x([str(i) for i in meas_lists["X"]])
        circ += noise_model.measure_z([str(i) for i in meas_lists["Z"]])
        circ += noise_model.idle([str(i) for i in meas_lists["idle"]])
        circ += noise_model.tick()
        return circ

    def morphing_QEC_round(self, noise_model, QEC_round, data_qubit_reset_basis = "None", data_qubit_measurement_basis = "None"):
        """
        Returns a stim circuit for one "measurement round" of QEC, from one end-cycle code to the next. QEC_round should be a half-integer, so that the returned circuit starts in the end-cycle code QEC_round - 1/2, then applies the inverse of the (QEC_round - 1/2)th contraction circuit, then the (QEC_round + 1/2)th contraction, then finishes in the end-cycle code QEC_round + 1/2.

        Arguments:
            noise_model is a surface_sim.Model object that defines the noise model to be used in the circuit.
            QEC_round is a float that should be equal to k+1/2 for some integer k. It defines which contraction circuits to implement in the circuit.
            data_qubit_reset_basis is a Str. When set to "X" or "Z", then the data qubits at the start of the circuit are reset in the corresponding basis; this can be used for example at the start of a memory experiment. When set to "None" (default) then the data qubits are not reset.
            data_qubit_measurement_basis is a Str. When set to "X" or "Z", then the data qubits at the end of the circuit are measured in the corresponding basis; this can be used for example at the end of a memory experiment. When set to "None" (default) then the data qubits are not measured.

        Raises:
            ValueError: Uh oh, data_qubit_reset_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!
            ValueError: Uh oh, data_qubit_measurement_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!

        Returns:
            A stim circuit for the QEC round.
        """
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        previous_round = round(QEC_round - 1 / 2) % n_rounds
        following_round = round(QEC_round + 1 / 2) % n_rounds
        circ = stim.Circuit()
        reset_lists = defaultdict(list)
        for pauli in ["X", "Z"]:
            reset_lists[pauli] += self.measurement_lists[pauli][previous_round]
        reset_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in reset_lists["X"]) and (not i in reset_lists["Z"])
        ]
        circ += noise_model.reset_x([str(i) for i in reset_lists["X"]])
        circ += noise_model.reset_z([str(i) for i in reset_lists["Z"]])
        if data_qubit_reset_basis == "X":
            circ += noise_model.reset_x([str(i) for i in reset_lists["idle"]])
        elif data_qubit_reset_basis == "Z":
            circ += noise_model.reset_z([str(i) for i in reset_lists["idle"]])
        elif data_qubit_reset_basis == "None":
            circ += noise_model.idle([str(i) for i in reset_lists["idle"]])
        else:
            raise ValueError("Uh oh, data_qubit_reset_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!")
        circ += noise_model.tick()
        expanding_gates = self.gate_lists[previous_round]
        for j in range(len(expanding_gates) - 1, -1, -1):
            cnot_list = [i for gate in expanding_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += noise_model.tick()
        contracting_gates = self.gate_lists[following_round]
        for j in range(len(contracting_gates)):
            cnot_list = [i for gate in contracting_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += noise_model.tick()
        meas_lists = defaultdict(list)
        for pauli in ["X", "Z"]:
            meas_lists[pauli] += self.measurement_lists[pauli][following_round]
        meas_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in meas_lists["X"]) and (not i in meas_lists["Z"])
        ]
        circ += noise_model.measure_x([str(i) for i in meas_lists["X"]])
        circ += noise_model.measure_z([str(i) for i in meas_lists["Z"]])
        if data_qubit_measurement_basis == "X":
            circ += noise_model.measure_x([str(i) for i in meas_lists["idle"]])
        elif data_qubit_measurement_basis == "Z":
            circ += noise_model.measure_z([str(i) for i in meas_lists["idle"]])
        elif data_qubit_measurement_basis == "None":
            circ += noise_model.idle([str(i) for i in meas_lists["idle"]])
        else:
            raise ValueError("Uh oh, data_qubit_measurement_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!")
        circ += noise_model.tick()
        return circ

    def morphing_QEC_round_XYZ_permutation(self, noise_model, data_qubit_reset_basis = "None", data_qubit_measurement_basis = "None"):
        """
        The equivalent of self.morphing_QEC_round except that an HS^dag gate is applied to each qubit in the mid-cycle state, and the same contraction circuit is applied both at the start and the end of the circuit. Intended for use in the get_morphing_XZ_stability_circuit_XYZ_permutation method.

        Arguments:
            noise_model is a surface_sim.Model object that defines the noise model to be used in the circuit.
            data_qubit_reset_basis is a Str. When set to "X" or "Z", then the data qubits at the start of the circuit are reset in the corresponding basis; this can be used for example at the start of a memory experiment. When set to "None" (default) then the data qubits are not reset.
            data_qubit_measurement_basis is a Str. When set to "X" or "Z", then the data qubits at the end of the circuit are measured in the corresponding basis; this can be used for example at the end of a memory experiment. When set to "None" (default) then the data qubits are not measured.

        Raises:
            ValueError: Uh oh, data_qubit_reset_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!
            ValueError: Uh oh, data_qubit_measurement_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!

        
        Returns:
            A stim circuit for the QEC round.
        """
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        previous_round = 0
        following_round = 0
        circ = stim.Circuit()
        reset_lists = {}
        for pauli in ["X", "Z"]:
            reset_lists[pauli] += self.measurement_lists[pauli][previous_round]
        reset_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in reset_lists["X"]) and (not i in reset_lists["Z"])
        ]
        circ += noise_model.reset_x([str(i) for i in reset_lists["X"]])
        circ += noise_model.reset_z([str(i) for i in reset_lists["Z"]])
        if data_qubit_reset_basis == "X":
            circ += noise_model.reset_x([str(i) for i in reset_lists["idle"]])
        elif data_qubit_reset_basis == "Z":
            circ += noise_model.reset_z([str(i) for i in reset_lists["idle"]])
        elif data_qubit_reset_basis == "None":
            circ += noise_model.idle([str(i) for i in reset_lists["idle"]])
        else:
            raise ValueError("Uh oh, data_qubit_reset_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!")
        circ += noise_model.tick()
        expanding_gates = self.gate_lists[previous_round]
        for j in range(len(expanding_gates) - 1, -1, -1):
            cnot_list = [i for gate in expanding_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += noise_model.tick()
        
        circ += noise_model.s_dag_gate([str(i) for i in range(n_qubits)])
        circ += noise_model.tick()
        circ += noise_model.hadamard([str(i) for i in range(n_qubits)])
        circ += noise_model.tick()
        
        contracting_gates = self.gate_lists[following_round]
        for j in range(len(contracting_gates)):
            cnot_list = [i for gate in contracting_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += noise_model.tick()
        meas_lists = {}
        for pauli in ["X", "Z"]:
            meas_lists[pauli] += self.measurement_lists[pauli][following_round]
        meas_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in meas_lists["X"]) and (not i in meas_lists["Z"])
        ]
        circ += noise_model.measure_x([str(i) for i in meas_lists["X"]])
        circ += noise_model.measure_z([str(i) for i in meas_lists["Z"]])
        if data_qubit_measurement_basis == "X":
            circ += noise_model.measure_x([str(i) for i in meas_lists["idle"]])
        elif data_qubit_measurement_basis == "Z":
            circ += noise_model.measure_z([str(i) for i in meas_lists["idle"]])
        elif data_qubit_measurement_basis == "None":
            circ += noise_model.idle([str(i) for i in meas_lists["idle"]])
        else:
            raise ValueError("Uh oh, data_qubit_measurement_basis needs to be 'X', 'Z', or 'None' for me to proceed, whoops!!!")
        circ += noise_model.tick()
        return circ

    def _mid_experiment_detector_circuits(self, additional_offset = 0):
        
        #assumes you are in the "middle" of the experiment; that is, more than n_rounds after the initialisation.
        
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        #goes through each mid-cycle stabiliser. Each stabiliser may contract multiple times through the circuit, given by stabiliser_contraction_rounds.
        #Each contraction round of each stabiliser gives rise to a detector. The measurements in the detector are the final measurement in the contraction round,
        #   and all the measurements that the stabiliser has support on while it is expanded, these are added to the detector round by round to give the
        #   detector instructions.
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                contraction_rounds = self.stabiliser_contraction_rounds[pauli][i]
                for j in range(len(contraction_rounds)):
                    rec_targets = []
                    current_contraction_round = contraction_rounds[j]
                    past_contraction_round = contraction_rounds[j - 1]
                    n_included_rounds = (
                        (current_contraction_round - past_contraction_round - 1)
                        % n_rounds
                    ) + 1
                    offset = len(self.measurement_lists["Z"][current_contraction_round])
                    offset += {
                        "X": len(
                            self.measurement_lists["X"][current_contraction_round]
                        ),
                        "Z": 0,
                    }[pauli]
                    for k in range(n_included_rounds):
                        this_round = (current_contraction_round - k) % n_rounds
                        rec_targets += [
                            stim.target_rec(l - offset - additional_offset)
                            for l in self.stabiliser_measurement_indices[pauli][
                                this_round
                            ][i]
                        ]
                        offset += len(self.measurement_lists["Z"][this_round - 1])
                        offset += len(
                            self.measurement_lists["X"][
                                this_round - {"X": 1, "Z": 0}[pauli]
                            ]
                        )
                    if self.stabiliser_annotations == {}:
                        detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[pauli])
                    else:
                        detector_annotation = self.stabiliser_annotations[pauli][i]
                    detector_circuits[current_contraction_round].append(
                        "DETECTOR", rec_targets, detector_annotation
                    )
        return detector_circuits

    def _detector_circuit_XYZ_permutation(self):
        #For use in the stability experiments with the XYZ permutation. There's only one circuit, since the same contraction circuit is applied each time.
        n_rounds = self.n_rounds()
        detector_circuit = stim.Circuit()
        #The assumptions of the permutation stability experiment are stringent enough that this function is quite simple compared to the general mid-experiment detector circuit.
        #Things are most convenient if we first construct the self-duality of the code.
        X_to_Z = {}
        Z_to_X = {}
        for i in range(len(self.mid_cycle_stabilisers["X"])):
            for j in range(len(self.mid_cycle_stabilisers["Z"])):
                if np.all([el in self.mid_cycle_stabilisers["Z"][j] for el in self.mid_cycle_stabilisers["X"][i]]):
                    if np.all([el in self.mid_cycle_stabilisers["X"][i] for el in self.mid_cycle_stabilisers["Z"][j]]):
                        X_to_Z[i] = j
                        Z_to_X[j] = i
        #See iPad notes 15/06/2025
        for i in range(len(self.mid_cycle_stabilisers["X"])):
            if self.stabiliser_contraction_rounds["X"][i] == [0]:
                rec_targets = []
                offset = len(self.measurement_lists["Z"][0]) + len(self.measurement_lists["X"][0])
                rec_targets += [stim.target_rec(l - offset) for l in self.stabiliser_measurement_indices["X"][0][i]]
                offset += len(self.measurement_lists["Z"][0])
                rec_targets += [stim.target_rec(l - offset) for l in self.stabiliser_measurement_indices["Z"][0][X_to_Z[i]]]
                offset += len(self.measurement_lists["X"][0])
                rec_targets += [stim.target_rec(l - offset) for l in self.stabiliser_measurement_indices["X"][0][i]]
                if self.stabiliser_annotations == {}:
                    detector_annotation = (0, 0, 0, 0)
                else:
                    detector_annotation = self.stabiliser_annotations["X"][i]
                detector_circuit.append(
                    "DETECTOR", rec_targets, detector_annotation
                )

        
        for i in range(len(self.mid_cycle_stabilisers["Z"])):
            if self.stabiliser_contraction_rounds["Z"][i] == [0]:
                rec_targets = []
                offset = len(self.measurement_lists["Z"][0])
                rec_targets += [stim.target_rec(l - offset) for l in self.stabiliser_measurement_indices["Z"][0][i]]
                offset += len(2*self.measurement_lists["X"][0] + self.measurement_lists["Z"][0])
                rec_targets += [stim.target_rec(l - offset) for l in self.stabiliser_measurement_indices["X"][0][Z_to_X[i]]]
                if self.stabiliser_annotations == {}:
                    detector_annotation = (0, 0, 0, 3)
                else:
                    detector_annotation = self.stabiliser_annotations["Z"][i]
                detector_circuit.append(
                    "DETECTOR", rec_targets, detector_annotation
                )

        return detector_circuit

    def _initial_memory_detector_circuits(self, basis):
        
        #Defines the detector circuits for the first n_rounds after the initialisation.
        #This is only needed for memory experiments performed in a fixed pauli basis ("X" or "Z"); for the XZ memory experiment because it begins with perfect QEC rounds the initial time boundary detectors are the same as the mid-experiment detectors.
        
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        if not basis in ["X", "Z"]:
            raise Exception("basis should be X or Z!")
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                contraction_rounds = self.stabiliser_contraction_rounds[pauli][i]
                for j in range(len(contraction_rounds)):
                    rec_targets = []
                    current_contraction_round = contraction_rounds[j]
                    if pauli != basis and j == 0:
                        continue
                    elif pauli == basis and j == 0:
                        n_included_rounds = current_contraction_round + 1
                    else:
                        past_contraction_round = contraction_rounds[j - 1]
                        n_included_rounds = (
                            (current_contraction_round - past_contraction_round - 1)
                            % n_rounds
                        ) + 1
                    offset = len(self.measurement_lists["Z"][current_contraction_round])
                    offset += {
                        "X": len(
                            self.measurement_lists["X"][current_contraction_round]
                        ),
                        "Z": 0,
                    }[pauli]
                    for k in range(n_included_rounds):
                        this_round = (current_contraction_round - k) % n_rounds
                        rec_targets += [
                            stim.target_rec(l - offset)
                            for l in self.stabiliser_measurement_indices[pauli][
                                this_round
                            ][i]
                        ]
                        offset += len(self.measurement_lists["Z"][this_round - 1])
                        offset += len(
                            self.measurement_lists["X"][
                                this_round - {"X": 1, "Z": 0}[pauli]
                            ]
                        )
                    if self.stabiliser_annotations == {}:
                        detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[pauli])
                    else:
                        detector_annotation = self.stabiliser_annotations[pauli][i]
                    detector_circuits[current_contraction_round].append(
                        "DETECTOR", rec_targets, detector_annotation
                    )
        return detector_circuits

    def _final_memory_extra_detector_circuits(self, final_round, basis):
        
        #Defines the "extra" detector circuits for the final round of measurements; that is, those that are expanding in the final round and therefore include final data qubit measurements.
        #This is only needed for memory experiments performed in a fixed pauli basis ("X" or "Z"); for the XZ memory experiment because it ends with perfect QEC rounds the final time boundary detectors are the same as the mid-experiment detectors.
        #Importantly, this doesn't include detectors that are contracting in the final round as these are already included in the "mid-experiment" detector circuits.
        
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        detector_circuit = stim.Circuit()
        if not basis in ["X", "Z"]:
            raise Exception("basis should be X or Z!")

        for i in range(len(self.mid_cycle_stabilisers[basis])):
            rec_targets = []
            contraction_rounds = self.stabiliser_contraction_rounds[basis][i]
            if final_round in contraction_rounds:
                continue
            temp_list = [c_round for c_round in contraction_rounds if c_round < final_round]
            if len(temp_list) > 0:
                past_contraction_round = temp_list[-1]
            else:
                past_contraction_round = contraction_rounds[-1]
            n_included_rounds = (
                (final_round - past_contraction_round - 1)
                % n_rounds
            ) + 1
            
            final_data_qubit_list = [i for i in range(self.n_qubits()) if not i in (self.measurement_lists["X"][final_round] + self.measurement_lists["Z"][final_round])]
            
            offset = len(final_data_qubit_list)
            rec_targets += [stim.target_rec(final_data_qubit_list.index(l) - offset) for l in self.end_cycle_stabilisers[basis][final_round][i]]
            
            offset += len(self.measurement_lists["Z"][final_round])
            offset += {
                "X": len(
                    self.measurement_lists["X"][final_round]
                ),
                "Z": 0,
            }[basis]
            for k in range(n_included_rounds):
                this_round = (final_round - k) % n_rounds
                rec_targets += [
                    stim.target_rec(l - offset)
                    for l in self.stabiliser_measurement_indices[basis][
                        this_round
                    ][i]
                ]
                offset += len(self.measurement_lists["Z"][this_round - 1])
                offset += len(
                    self.measurement_lists["X"][
                        this_round - {"X": 1, "Z": 0}[basis]
                    ]
                )
            if self.stabiliser_annotations == {}:
                detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[basis])
            else:
                detector_annotation = self.stabiliser_annotations[basis][i]
            detector_circuit.append(
                "DETECTOR", rec_targets, detector_annotation
            )
        return detector_circuit
    
    def _initial_stability_detector_circuits(self):
        
        #defines the detector circuits for the BULK detectors (NOT boundary detectors) in the first n_rounds after the initialisation.
        #This is used in the stability experiment because in general there may be some bulk detectors in the first n_rounds because some stabilisers may be contracting in more than one round.
        
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                contraction_rounds = self.stabiliser_contraction_rounds[pauli][i]
                if contraction_rounds[-1] == n_rounds - 1:
                    contraction_rounds = [contraction_rounds[-1]] + contraction_rounds[:-1]
                if len(contraction_rounds) > 1:
                    for j in range(1, len(contraction_rounds)):
                        rec_targets = []
                        current_contraction_round = contraction_rounds[j]
                        past_contraction_round = contraction_rounds[j - 1]
                        n_included_rounds = (
                            (current_contraction_round - past_contraction_round - 1)
                            % n_rounds
                        ) + 1
                        offset = len(
                            self.measurement_lists["Z"][current_contraction_round]
                        )
                        offset += {
                            "X": len(
                                self.measurement_lists["X"][current_contraction_round]
                            ),
                            "Z": 0,
                        }[pauli]
                        for k in range(n_included_rounds):
                            this_round = (current_contraction_round - k) % n_rounds
                            rec_targets += [
                                stim.target_rec(l - offset)
                                for l in self.stabiliser_measurement_indices[pauli][
                                    this_round
                                ][i]
                            ]
                            offset += len(self.measurement_lists["Z"][this_round - 1])
                            offset += len(
                                self.measurement_lists["X"][
                                    this_round - {"X": 1, "Z": 0}[pauli]
                                ]
                            )
                        if self.stabiliser_annotations == {}:
                            detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[pauli])
                        else:
                            detector_annotation = self.stabiliser_annotations[pauli][i]
                        detector_circuits[current_contraction_round].append(
                            "DETECTOR", rec_targets, detector_annotation
                        )
        return detector_circuits

    def _mid_experiment_logical_observable_circuits(self, basis = "both"):
        
        #assumes you are in the "middle" of the experiment; that is, more than n_rounds after the initialisation.
        #this is for when your observable is a logical operator (as opposed to a stabiliser redundancy, in which case use _initial_redundancy_observable_circuits below)
        #basis should be "both" if both X and Z logicals are being included as observables, "X" if it's just "X" operators (as in an X-basis memory experiment), or "Z" if it's just "Z" operators (as in a Z-basis memory experiment).
        
        n_rounds = self.n_rounds()
        n_logicals = self.n_logicals()
        observable_circuits = [stim.Circuit() for _ in range(n_rounds)]
        if basis == "X":
            basis = ["X"]
        elif basis == "Z":
            basis = ["Z"]
        elif basis == "both":
            basis = ["X", "Z"]
        for pauli in basis:
            for i in range(n_logicals):
                observable_number = i
                if len(basis) == 2 and pauli == "Z":
                    observable_number += n_logicals
                for current_contraction_round in range(n_rounds):
                    offset = len(self.measurement_lists["Z"][current_contraction_round])
                    offset += {
                        "X": len(
                            self.measurement_lists["X"][current_contraction_round]
                        ),
                        "Z": 0,
                    }[pauli]
                    indices_to_include = self.logical_measurement_indices[pauli][
                        current_contraction_round
                    ][i]
                    targets_to_include = [
                        stim.target_rec(indices_to_include[j] - offset)
                        for j in range(len(indices_to_include))
                    ]
                    observable_circuits[current_contraction_round].append(
                        "OBSERVABLE_INCLUDE",
                        targets_to_include,
                        observable_number,
                    )
        return observable_circuits

    def _initial_redundancy_observable_circuits(self):
        
        #uses the initial (-1) end-cycle code of the experiment to define the redundancies that are included in each observable.
        #(or, equivalently, the pre-initial (-1.5) mid-cycle code)
        #slightly confusingly, this also includes detectors for any meta-check redundancies.
        
        n_rounds = self.n_rounds()
        n_redundancies = self.n_redundancies()
        targets_to_include = {pauli: [
            [[] for _ in range(n_redundancies["X"] + n_redundancies["Z"])]
            for _ in range(n_rounds - 1)
        ] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            redundancy_index = 0
            for i in range(n_redundancies[pauli]):
                redundancy = self.redundancies[pauli][i]
                for j in range(len(redundancy)):
                    stabiliser_index = redundancy[j]
                    if (
                        n_rounds - 1
                        not in self.stabiliser_contraction_rounds[pauli][
                            stabiliser_index
                        ]
                    ):
                        for current_contraction_round in range(
                            self.stabiliser_contraction_rounds[pauli][stabiliser_index][
                                0
                            ]
                            + 1
                        ):
                            indices_to_include = self.stabiliser_measurement_indices[
                                pauli
                            ][current_contraction_round][stabiliser_index]
                            offset = len(
                                self.measurement_lists["Z"][current_contraction_round]
                            )
                            offset += {
                                "X": len(
                                    self.measurement_lists["X"][
                                        current_contraction_round
                                    ]
                                ),
                                "Z": 0,
                            }[pauli]
                            targets_to_include[pauli][current_contraction_round][
                                redundancy_index
                            ] += [
                                index - offset
                                for index in indices_to_include
                            ]
                redundancy_index += 1
        observable_indices = {"X": [], "Z": []}
        detector_indices = {"X": [], "Z": []}
        for pauli in ["X", "Z"]:
            if len(self.meta_checks[pauli]) == 0:
                observable_indices[pauli] = np.eye(n_redundancies[pauli])
                detector_indices[pauli] = []
            else:
                h = self.parity_check_matrices()
                hT_kernel = galois.GF(2)(np.zeros((len(self.redundancies[pauli]), h[pauli].shape[0]), dtype = int))
                for i in range(len(self.redundancies[pauli])):
                    for j in self.redundancies[pauli][i]:
                        hT_kernel[i][j] = 1

                #hT_kernel[pauli] has as its rows the redundancies and as its columns the stabilisers that go into the redundancy.

                meta_check_vectors = galois.GF(2)(np.zeros((len(self.meta_checks[pauli]),h[pauli].shape[0]), dtype = int))
                for i in range(len(self.meta_checks[pauli])):
                    for j in self.meta_checks[pauli][i]:
                        meta_check_vectors[i][j] = 1

                #meta_check_vectors has the redundancies that are a detector as its rows and the stabilisers that go into the redundancy as its columns
                row_reduced = meta_check_vectors.row_reduce()
                null_space_2 = galois.GF(2)(np.eye(row_reduced.shape[1], dtype = int))
                for i in range(len(row_reduced)):
                    if len(np.nonzero(row_reduced[i])[0]) > 0:
                        null_space_2 = np.delete(null_space_2, np.nonzero(row_reduced[i])[0][0] - i, 0)
                #null_space_2 has as its rows the observable redundancies and as its columns the stabilisers that go into the redundancy
                observable_indices[pauli] = galois.GF(2)(np.block([null_space_2.T, hT_kernel.T])).null_space()[:,null_space_2.shape[0]:]
                #detector_indices has as its rows the detectors and as its columns the redundancies that go into the detector
                detector_indices[pauli] = galois.GF(2)(np.block([meta_check_vectors.T, hT_kernel.T])).null_space()[:,meta_check_vectors.shape[0]:]
                #observable_indices has as its rows the observables and as its columns the redundancies that go into the observable
        observable_circuits = [stim.Circuit() for _ in range(n_rounds - 1)]
        observable_argument = 0
        for pauli in ["X", "Z"]:
            for observable in observable_indices[pauli]:
                for current_contraction_round in range(n_rounds - 1):
                    total_targets_to_include = []
                    redundancies_to_include = np.nonzero(observable)[0]
                    for redundancy_index in redundancies_to_include:
                        total_targets_to_include += targets_to_include[pauli][current_contraction_round][redundancy_index]
                    target_occurrences = Counter(total_targets_to_include)
                    targets_to_include_here = [target for target in target_occurrences.keys() if target_occurrences[target] % 2 == 1]
                    observable_circuits[current_contraction_round].append(
                        "OBSERVABLE_INCLUDE",
                        [stim.target_rec(target) for target in targets_to_include_here],
                        observable_argument,
                    )
                observable_argument += 1
            for detector in detector_indices[pauli]:
                targets_to_include_here = []
                for current_contraction_round in range(n_rounds - 1):
                    offset = len(self.measurement_lists["X"][current_contraction_round]) + len(self.measurement_lists["Z"][current_contraction_round])
                    targets_to_include_here = [target - offset for target in targets_to_include_here]
                    total_targets_to_include = []
                    redundancies_to_include = np.nonzero(detector)[0]
                    for redundancy_index in redundancies_to_include:
                        total_targets_to_include += targets_to_include[pauli][current_contraction_round][redundancy_index]
                    target_occurrences = Counter(total_targets_to_include)
                    targets_to_include_here += [target for target in target_occurrences.keys() if target_occurrences[target] % 2 == 1]
                observable_circuits[n_rounds - 2].append(
                    "DETECTOR",
                    [stim.target_rec(target) for target in targets_to_include_here],
                    (0,0,0,{"X": 0, "Z": 3}[pauli]),
                )
        return observable_circuits
