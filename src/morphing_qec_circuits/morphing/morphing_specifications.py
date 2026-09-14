import numpy as np
import stim
import galois  # for matrix calculations over a finite field
from surface_sim.models import CircuitNoiseModel, NoiselessModel
from surface_sim.setups import SQ_GATES, SQ_MEASUREMENTS, SQ_RESETS, TQ_GATES
from .CSS_morphing_specifications import CSSMorphingSpecifications
from ..misc import nkd_params, get_symplectic_logical_basis, uniform_setup, circuit_distance, forward_propagate_pauli
from collections import Counter

#for some reason this isn't loaded by surface_sim.setups
ANNOTATIONS = {"tick": "TICK", "qubit_coords": "QUBIT_COORDS"}

class MorphingSpecifications:
    """
    A class for Morphing circuits.

    Attributes: 
        mid_cycle_stabilisers represents all the stabilisers such that mid_cycle_stabilisers[i] is a stim.PauliString describing the ith stabiliser.
        stabiliser_contraction_rounds lists the rounds during which each corresponding mid cycle stabiliser is contracting. MUST BE IN ASCENDING ORDER!!!
        contraction_circuits lists the contraction circuits in the morphing circuit. It's intentended that each contraction circuit is:
            - noiseless (noise is added when constructing the circuits using a surface_sim.NoiseModel)
            - has internal TICK annotations (TICKs at the start/end of the contraction circuit will be added/removed automatically as necessary)
            During initialisation, identity gates are added to qubits that are not participating in an operation each TICK. If this is not desired, you can set add_idling = False during initialisation.
        mid_cycle_logicals represents the logical operators of the mid-cycle code, such that mid_cycle_logicals[pauli][i] lists the qubits in the support of the logical operator \bar{pauli}_{i}.
        meta_checks are redundancies (see below) that should be treated as detectors instead of observables in stability experiments. Specifically, each meta check lists the indices of
            the mid_cycle_stabilisers that products to the identity.
        redundancies represents the linear combinations of mid_cycle_stabilisers that product to the identity. Specifically, each redundancy lists the indices of
            the mid_cycle_stabilisers that products to the identity.
        coordinates_list is a list of complex numbers representing coordinates for each qubit.
        end_cycle_stabilisers represents the support of each mid_cycle_stabiliser on the data qubits at the end of each round, such that end_cycle_stabilisers[c_round][k] is a stim.PauliString describing the support at the end of the round c_round of the mid-cycle stabiliser mid_cycle_stabiliser[k].
        stabiliser_measurement_target_recs represents the support of each mid-cycle stabiliser on measurements at the end of each round, such that stabiliser_measurement_target_recs[c_round][k] lists the indices i of the measurements (given explicitly by the ordering in contraction_circuits[c_round]) during the round c_round that are supported by the mid-cycle stabiliser mid_cycle_stabiliser[k].
        end_cycle_logicals represents the support of each mid-cycle logical on the data qubits at the end of each round, such that end_cycle_logicals[pauli][c_round][k] is a stim.PauliString representing the support of at the end of the round c_round of the mid-cycle logical mid_cycle_logical[pauli][k].
        logical_measurement_target_recs represents the support of each mid-cycle logical on measurements at the end of each round, such that logical_measurement_target_recs[pauli][c_round][k] lists the indices i of the measurements (given explicitly by the ordering in contraction_circuits[c_round]) during the round c_round that are supported by the mid-cycle logical mid_cycle_logical[pauli][k].
    """

    def __new__(
        cls,
        *args,
        **kwargs,
    ):
        if len(args) == 4:
            return CSSMorphingSpecifications(*args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(
        self,
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        contraction_circuits,
        mid_cycle_logicals={},
        meta_checks=[],
        redundancies=[],
        coordinates_list=[],
        end_cycle_stabilisers=[],
        stabiliser_measurement_target_recs=[],
        end_cycle_logicals={},
        logical_measurement_target_recs={},
        stabiliser_annotations=[],
        mid_cycle_nkd = (None, None, None),
        end_cycle_nkd = [],
        circuit_distance = None,
        residual_distance = None,
        add_idling = True,
    ):
        """
        By default, the contraction circuits are checked and idling is added to qubits that do not take part in an operation in each TICK. To disable this, set add_idling to False.
        """
        self.mid_cycle_stabilisers = mid_cycle_stabilisers
        self.stabiliser_contraction_rounds = stabiliser_contraction_rounds
        if add_idling:
            self.contraction_circuits = []
            for contraction_circuit in contraction_circuits:
                to_append = stim.Circuit()
                idling_qubits = list(range(contraction_circuit.num_qubits))
                for i in range(len(contraction_circuit)):
                    instruction = contraction_circuit[i]
                    if instruction.name == "TICK" and i != 0:
                        to_append.append("I", idling_qubits)
                        to_append.append(instruction)
                        idling_qubits = list(range(contraction_circuit.num_qubits))
                    else:
                        to_append.append(instruction)
                        for target in instruction.targets_copy():
                            if target.value in idling_qubits:
                                idling_qubits.remove(target.value)
                        if i == len(contraction_circuit) - 1:
                            to_append.append("I", idling_qubits)
                self.contraction_circuits.append(to_append)
        else:
            self.contraction_circuits = contraction_circuits
        self.mid_cycle_logicals = mid_cycle_logicals
        self.meta_checks = meta_checks
        self.redundancies = redundancies
        self.coordinates_list = coordinates_list
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_target_recs = stabiliser_measurement_target_recs
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_target_recs = logical_measurement_target_recs
        self.stabiliser_annotations = stabiliser_annotations
        self.mid_cycle_nkd = (self.n_qubits(), mid_cycle_nkd[1], mid_cycle_nkd[2])
        if len(end_cycle_nkd) == self.n_rounds():
            self.end_cycle_nkd = end_cycle_nkd
        else:
            self.end_cycle_nkd = [(None, None, None) for _ in range(self.n_rounds())]
        self.circuit_distance = circuit_distance
        self.residual_distance = residual_distance
    
    def __repr__(self):
        """
        Returns a string that specifies the morphing specifications.
        """
        cls = self.__class__.__name__
        mid_cycle_nkd_str = f'[[{self.mid_cycle_nkd[0]},{"?" if self.mid_cycle_nkd[1] == None else self.mid_cycle_nkd[1]},{"?" if self.mid_cycle_nkd[2] == None else self.mid_cycle_nkd[2]}]]'
        end_cycle_nkd_strs = [f'[[{self.end_cycle_nkd[c_round][0]},{"?" if self.end_cycle_nkd[c_round][1] == None else self.end_cycle_nkd[c_round][1]},{"?" if self.end_cycle_nkd[c_round][2] == None else self.end_cycle_nkd[c_round][2]}]]' for c_round in range(self.n_rounds())]
        return f'{cls}(\nmid_cycle_stabilisers={self.mid_cycle_stabilisers},\nstabiliser_contraction_rounds={self.stabiliser_contraction_rounds},\ncontraction circuits={self.contraction_circuits},\nmid-cycle [[n,k,d]]={mid_cycle_nkd_str}\nend-cycle [[n,k,d]]={end_cycle_nkd_strs}\ncircuit distance={self.circuit_distance}\nresidual distance={self.residual_distance}\n)'
        
    def n_qubits(self):
        """
        Returns the number of qubits in the mid-cycle code.
        """
        return max(
            [
                len(stabiliser)
                for stabiliser in self.mid_cycle_stabilisers
            ]
        )

    def n_rounds(self):
        """
        Returns the number of contraction rounds of the morphing circuit.

        Returns:
            The number of contraction rounds of the morphing circuit.
        """
        return len(self.contraction_circuits)

    def n_redundancies(self):
        """
        Returns dictionary the number of redundancies between the stabilisers.
        """
        return len(self.parity_check_matrix().T.null_space())

    def n_stabilisers(self):
        """
        Returns a dictionary with keys "X" and "Z" and items the number of X (respectively, Z) type stabilisers.
        """
        return len(self.mid_cycle_stabilisers)

    def n_meta_checks(self):
        """
        Returns a dictionary with keys "X" and "Z" and items the number of X and Z type meta-checks.
        """
        return len(self.meta_checks)

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
                - n_total_stabilisers
                + n_redundancies
            )
            self.mid_cycle_nkd = (self.mid_cycle_nkd[0], to_return, self.mid_cycle_nkd[2])
            return to_return
        else:
            return self.mid_cycle_nkd[1]

    def check_consistent_gates(self):
        for c_round in range(self.n_rounds()):
            measured_qubits = []
            for instruction in self.contraction_circuits[c_round]:
                if instruction.name in ["I", "TICK"]:
                    continue
                for i in instruction.targets_copy():
                    if i.value in measured_qubits:
                        raise ValueError(f"contraction circuits should not involve qubits after they are measured, but in contraction circuit {c_round}, qubit {i.value} is used in instruction {instruction}")
                if instruction.name in ["R", "RX", "RY", "RZ"]:
                    raise ValueError(f"contraction circuits should not have resets, but in contraction circuit {c_round} I found a reset instruction {instruction}")
                elif instruction.name in ["MPP"]:
                    raise ValueError(f"contraction circuits should not have MPP instructions, but in contraction circuit {c_round} I found an MPP instruction {instruction}")

                elif instruction.name in ["M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"]:
                    measured_qubits += [i.value for i in instruction.targets_copy()]

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
            h = self.parity_check_matrix()
            to_return = nkd_params(
                h,
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
            h = self.end_cycle_parity_check_matrices()
            if check_all_end_cycle_codes:
                nkd_params_list = []
                for i in range(self.n_rounds()):
                    nkd_params_list.append(
                        nkd_params(
                            h[i],
                            **kwargs,
                        )
                    )
                self.end_cycle_nkd = nkd_params_list
                return nkd_params_list
            else:
                first_nkd_params = nkd_params(
                    h[0],
                    **kwargs,
                )
                to_return = [first_nkd_params for _ in range(self.n_rounds())]
                self.end_cycle_nkd = to_return
                return to_return
        else:
            return self.end_cycle_nkd

    def parity_check_matrix(self):
        """
        Returns a parity check matrix h for the stabilisers of the mid-cycle code. The first n_qubits columns represent the X-type support of each
            stabiliser, and the last n_qubits columns represent the Z-type support.
        """
        h_int = []
        for stabiliser in self.mid_cycle_stabilisers:
            h_int.append(np.concatenate(stabiliser.to_numpy()))
                
        h = galois.GF(2)(h_int)
        
        n_qubits = h.shape[1]//2
        zero_block = np.zeros((n_qubits, n_qubits)).astype(int)
        identity_block = np.eye(n_qubits).astype(int)
        omega = galois.GF(2)(np.block([[zero_block, identity_block], [identity_block, zero_block]]))
        comm_rel = h @ omega @ h.T
        if np.any(comm_rel):
            raise Exception("mid-cycle stabilisers do not commute!")
        return h

    def left_over_qubits(self, c_round):
        qubit_indices = list(range(self.n_qubits()))
        for instruction in self.contraction_circuits[c_round]:
            if instruction.num_measurements > 0:
                for gate_target in instruction.targets_copy():
                    qubit_indices.remove(gate_target.qubit_value)
        return qubit_indices

    def end_cycle_parity_check_matrices(self):
        """
        Returns a list of parity check matrices h for the stabilisers of each end-cycle code.
        """
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        hs = [[] for i in range(n_rounds)]
        if self.end_cycle_stabilisers == []:
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_target_recs()
        for i in range(n_rounds):
            left_over_qubits_list = self.left_over_qubits(i)
            h_int = []
            for stabiliser in self.end_cycle_stabilisers[i]:
                xs, zs = stabiliser.to_numpy()
                if np.sum(xs) + np.sum(zs) > 0:
                    h_int.append(np.concatenate((xs[left_over_qubits_list], zs[left_over_qubits_list])))
            if len(h_int) == 0:
                hs[i] = galois.GF(2)(np.zeros((1,len(left_over_qubits_list)), dtype = int))
            else:
                hs[i] = galois.GF(2)(h_int)

            n_qubits = hs[i].shape[1]//2
            zero_block = np.zeros((n_qubits, n_qubits)).astype(int)
            identity_block = np.eye(n_qubits).astype(int)
            omega = galois.GF(2)(np.block([[zero_block, identity_block], [identity_block, zero_block]]))
        
            comm_rel = hs[i] @ omega @ hs[i].T
            if np.any(comm_rel):
                raise Exception(f"end-cycle stabilisers in contraction round {i} do not commute!")
            
        return hs

    def compile_mid_cycle_logicals(self):
        """
        Calculates the mid-cycle logical operators of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_logicals.
        """
        h = self.parity_check_matrix()
        n_qubits = self.n_qubits()
        symplectic_logical_basis = get_symplectic_logical_basis(h)
        mid_cycle_logicals = {"X": [], "Z": []}
        for pauli in ["X", "Z"]:
            for i in range(symplectic_logical_basis[pauli].shape[0]):
                mid_cycle_logicals[pauli].append(stim.PauliString.from_numpy(
                    xs = np.array(symplectic_logical_basis[pauli][i,:n_qubits], dtype = np.bool_),
                    zs = np.array(symplectic_logical_basis[pauli][i,n_qubits:], dtype = np.bool_),
                ))

        self.mid_cycle_logicals = mid_cycle_logicals

    def compile_redundancies(self):
        """
        Calculates the redundancies between the stabilisers of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_redundancies.
        (Note that these redundancies do not change throughout the circuit since the identity operator is preserved by all unitary gates)
        """
        h = self.parity_check_matrix()
        hT_kernel = h.T.null_space()
        n_redundancies = len(hT_kernel)

        mid_cycle_redundancies = []
        for redundancy in hT_kernel:
            redundant_stabilisers = np.nonzero(redundancy)[0]
            mid_cycle_redundancies.append(redundant_stabilisers)

        self.redundancies = mid_cycle_redundancies

    def compile_end_cycle_stabilisers_and_stabiliser_measurement_target_recs(self):
        """
        Calculates the end-cycle stabilisers and measurement indices and sets them to self.end_cycle_stabilisers and self.stabiliser_measurement_target_recs (respectively).
        """

        self.check_consistent_gates()
        n_rounds = self.n_rounds()
        end_cycle_stabilisers = [[] for _ in range(n_rounds)]
        stabiliser_measurement_target_recs = [[] for _ in range(n_rounds)]
        
        for mid_cycle_stabiliser in self.mid_cycle_stabilisers:
            for i in range(n_rounds):
                end_cycle_stabiliser, measurement_target_recs = forward_propagate_pauli(self.contraction_circuits[i], mid_cycle_stabiliser)
                end_cycle_stabilisers[i].append(end_cycle_stabiliser)
                stabiliser_measurement_target_recs[i].append(measurement_target_recs)
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_target_recs = stabiliser_measurement_target_recs
    
    def compile_end_cycle_logicals_and_logical_measurement_target_recs(self):
        """Calculates the end-cycle logical operators and measurement target recs and sets them to self.end_cycle_logicals and self.logical_measurement_target_recs (respectively)."""

        self.check_consistent_gates()
        n_rounds = self.n_rounds()
        end_cycle_logicals = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        logical_measurement_target_recs = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        for pauli in ["X", "Z"]:
            for mid_cycle_logical in self.mid_cycle_logicals[pauli]:
                for i in range(n_rounds):
                    end_cycle_logical, measurement_target_recs = forward_propagate_pauli(self.contraction_circuits[i], mid_cycle_logical)
                    end_cycle_logicals[pauli][i].append(end_cycle_logical)
                    logical_measurement_target_recs[pauli][i].append(measurement_target_recs)
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_target_recs = logical_measurement_target_recs

    def compile_all(self, check_compiled=False):
        """Compiles the mid-cycle logical operators, mid-cycle redundancies, end-cycle stabilisers, stabiliser measurement target recs, end-cycle logical operators, and logical measurement target recs, and assigns them to the corresponding attributes.
        
        Args:
            check_compiled is a bool that if True only compiles the attributes that have not yet been computed.
        """
        if (not check_compiled) or self.mid_cycle_logicals == {}:
            self.compile_mid_cycle_logicals()
        if (not check_compiled) or self.redundancies == []:
            self.compile_redundancies()
        if (
            (not check_compiled)
            or self.end_cycle_stabilisers == []
            or self.stabiliser_measurement_target_recs == []
        ):
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_target_recs()
        if (
            (not check_compiled)
            or self.end_cycle_logicals == {}
            or self.logical_measurement_target_recs == {}
        ):
            self.compile_end_cycle_logicals_and_logical_measurement_target_recs()
    
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

        circ += self._logical_bell_state_measurements_and_observables((which_round+1) % n_rounds)
        
        for i in range(which_round + 1, which_round + n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, (i + 1 / 2) % n_rounds)
            circ += observable_circuits[(i + 1) % n_rounds]

        # perfect Bell state prepared, now add the incoming noise.

        circ.append("DEPOLARIZE1", range(n_qubits), p)

        # perfect Bell measurements now

        for i in range(which_round, which_round - 1 + n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, (i + 1 / 2) % n_rounds)
            circ += detector_circuits[(i + 1) % n_rounds]
            circ += observable_circuits[(i + 1) % n_rounds]

        circ += self._logical_bell_state_measurements_and_observables((which_round-1) % n_rounds)
        
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
        
        circ += self._logical_bell_state_measurements_and_observables(0)
        
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

        circ += self._logical_bell_state_measurements_and_observables(final_round)
        
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
        if (not recalculate) and self.n_rounds() == 2 and self.residual_distance != None:
            self.circuit_distance = self.residual_distance
            return self.residual_distance
            
        if recalculate or self.circuit_distance == None:
            noise_setup = uniform_setup(0.001)
            self.end_cycle_nkd_params()
            calculated_distances = [nkd[2] for nkd in ([self.mid_cycle_nkd] + self.end_cycle_nkd) if nkd[2] != None]
            cutoff = min(calculated_distances + [cutoff])
            if cutoff <= 1:
                self.circuit_distance = cutoff
                return cutoff
            if (not recalculate) and self.residual_distance != None:
                if cutoff == self.residual_distance:
                    self.circuit_distance = cutoff
                    return cutoff
            
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
            
            noise_model_generator = lambda qubit_dictionary, noise_setup: CircuitNoiseModel(qubit_dictionary, noise_setup)
            
            stim_circuit_errors = self.get_morphing_XZ_memory_circuit(
                noise_setup,
                T,
                noise_model_generator = noise_model_generator,
            )
    
            to_return = circuit_distance(stim_circuit_errors, error_basis = "both", cutoff = cutoff)
            self.circuit_distance = to_return
            return to_return
        else:
            return self.circuit_distance
            
    def get_residual_error_circuit(
        self,
        noise_setup,
        noise_model_generator = lambda qubit_dictionary, noise_setup: CircuitNoiseModel(
            qubit_dictionary, noise_setup
        ),
        round_coords = False,
        preserve_probabilities = True,
        extra_ticks = False,
    ):
        """
        Returns a stim circuit for the static mid-cycle code, with a correlated noise model that corresponds to the
            "residual" errors obtained by propagating the circuit-level noise from each contraction circuit back to
            the mid-cycle code. The circuit-level distance of this residual error circuit should be a lower-bound
            to the actual circuit-level distance (I haven't formally proved this but it seems right intuitively),
            and, stronger still, if n_rounds = 2 then the circuit-level distance of the residual error circuit 
            exactly equals the circuit-level distance of the morphing circuit (from Proposition 1 of arxiv:2604.09797).
            
        Arguments:
            noise_setup is a surface_sim.Setup object that sets up the noise model to be used in the circuit.
            noise_model_generator is a function that takes two arguments: a qubit dictionary, and a noise setup, and returns a surface_sim.Model
                object that defines the noise model of the memory experiment.
            round_coords is a bool that is passed to self.coordinate_circuit.
            preserve_probabilities is a bool that if set to False, merges all equivalent residual errors and sets their probability to 0.01.
            extra_ticks is a bool that if set to True, adds a TICK instruction after each residual error. Can be useful for visualising the
                residual errors in relatively small codes/circuits.
        Raises:
            NotImplementedError: residual_error_circuit does not support this instruction.
        
        Returns:
            A stim circuit for the residual errors in the mid-cycle code.
        """
        n_qubits = self.n_qubits()
        n_stabilisers = len(self.mid_cycle_stabilisers)
        n_rounds = self.n_rounds()
        self.compile_all(check_compiled=True)
        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        noise_model = noise_model_generator(qubit_dictionary, noise_setup)
        residual_errors = []
        residual_error_probabilities = []
        gate_stim_names = list(SQ_GATES.values()) + list(TQ_GATES.values())
        for c_round in range(n_rounds):
            noisy_contraction_circuit = stim.Circuit()
            noisy_contraction_circuit += noise_model.idle([str(i) for i in range(n_qubits)])
            noisy_contraction_circuit += noise_model.tick()
            noisy_contraction_circuit += self.contraction_circuit(c_round, noise_model = noise_model)
            unitary_circuit = stim.Circuit()
            already_measured_qubits = []
            for instruction in noisy_contraction_circuit:
                if instruction.name in gate_stim_names:
                    #don't need to check things like whether the gates involved already measured qubits,
                    # because that's already checked by another method.
                    unitary_circuit.append(instruction)
                elif instruction.name in list(SQ_MEASUREMENTS.values()):
                    already_measured_qubits += [target.value for target in instruction.targets_copy()]
                elif instruction.name in ["X_ERROR", "Y_ERROR", "Z_ERROR"]:
                    for target in instruction.targets_copy():
                        if target.value in already_measured_qubits:
                            continue
                        residual_error = stim.PauliString(n_qubits)
                        residual_error[target.value] = instruction.name[0]
                        residual_error = residual_error.before(unitary_circuit)
                        residual_errors.append(residual_error)
                        residual_error_probabilities.append(instruction.gate_args_copy()[0])
                elif instruction.name in ["DEPOLARIZE1"]:
                    for target in instruction.targets_copy():
                        if target.value in already_measured_qubits:
                            continue
                        for pauli in ["X", "Y", "Z"]:
                            residual_error = stim.PauliString(n_qubits)
                            residual_error[target.value] = pauli
                            residual_error = residual_error.before(unitary_circuit)
                            residual_errors.append(residual_error)
                            p_individual = 0.5 - np.sqrt(0.25 - instruction.gate_args_copy()[0]/3)
                            residual_error_probabilities.append(p_individual)
                elif instruction.name in ["DEPOLARIZE2"]:
                    targets_list = instruction.targets_copy()
                    for i in range(len(targets_list)//2):
                        targets = [targets_list[2*i].value, targets_list[2*i+1].value]
                        if (targets[0] in already_measured_qubits) or (targets[1] in already_measured_qubits):
                            continue
                        for pauli in ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]:
                            residual_error = stim.PauliString(n_qubits)
                            residual_error[targets[0]] = pauli[0]
                            residual_error[targets[1]] = pauli[1]
                            residual_error = residual_error.before(unitary_circuit)
                            residual_errors.append(residual_error)
                            p_individual = 0.5 - 0.5 * (1 - 16*instruction.gate_args_copy()[0]/15)**(1/8)
                            residual_error_probabilities.append(p_individual)
                elif instruction.name in [
                    "CORRELATED_ERROR",
                    "E",
                    "ELSE_CORRELATED_ERROR",
                    "HERALDED_ERASE",
                    "HERALDED_PAULI_CHANNEL_1",
                    "PAULI_CHANNEL_1",
                    "PAULI_CHANNEL_2",
                ]:
                    raise NotImplementedError(f"residual_error_circuit does not support {instruction.name} instructions.")
        circuit = self.coordinate_circuit(n_extra_qubits = self.n_logicals(), round_coords = round_coords)
        circuit += self._logical_mid_cycle_bell_state_measurements_and_observables()
        circuit.append("MPP", self.mid_cycle_stabilisers)
        circuit.append("TICK")
        if preserve_probabilities:
            for i in range(len(residual_errors)):
                circuit.append("E", [residual_errors[i]], residual_error_probabilities[i])
                if extra_ticks:
                    circuit.append("TICK")
        else:
            already_added = []
            for i in range(len(residual_errors)):
                to_add = residual_errors[i] / residual_errors[i].sign
                if to_add in already_added:
                    continue
                circuit.append("E", [to_add], 0.001)
                already_added.append(to_add)
                if extra_ticks:
                    circuit.append("TICK")
        if not extra_ticks:
            circuit.append("TICK")
        circuit.append("MPP", self.mid_cycle_stabilisers)
        circuit.append("TICK")
        for i in range(n_stabilisers):
            circuit.append("DETECTOR", [stim.target_rec(i - 2*n_stabilisers), stim.target_rec(i - n_stabilisers)])
        circuit += self._logical_mid_cycle_bell_state_measurements_and_observables()
        return circuit
        
    def calculate_residual_distance(
        self,
        recalculate = False,
        cutoff = np.inf,
    ):
        """
        Returns the circuit distance of the residual error circuit and sets it to self.residual_distance.
        The end-cycle distances are automatically pre-calculated here (or taken from stored values) to provide an upper-bound on the
            circuit-distance.

        Arguments:
            recalculate is a bool that if set to True will mean that the residual distance is recalculated, rather than using a stored value.
            cutoff is an Int (or np.inf) setting an upper-bound on the calculated residual distance. If the residual distance is greater than or
                equal to cutoff, the algorithm will simply return cutoff.

        Returns:
            The residual distance of the morphing circuit.
        """
        
        if recalculate or self.residual_distance == None:
            calculated_distances = [nkd[2] for nkd in ([self.mid_cycle_nkd] + self.end_cycle_nkd) if nkd[2] != None]
            cutoff = min(calculated_distances + [cutoff])
            if cutoff <= 1:
                self.residual_distance = cutoff
                return cutoff
            noise_setup = uniform_setup(0.001)
            
            noise_model_generator = lambda qubit_dictionary, noise_setup: CircuitNoiseModel(qubit_dictionary, noise_setup)
            
            stim_circuit_errors = self.get_residual_error_circuit(
                noise_setup,
                noise_model_generator = noise_model_generator,
                preserve_probabilities = False,
            )
    
            to_return = circuit_distance(stim_circuit_errors, error_basis = "both", cutoff = cutoff)
            self.residual_distance = to_return
            return to_return
        else:
            return self.residual_distance

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
            NotImplementedError: fixed basis memory experiments are not implemented for non-CSS morphing circuits.
        
        Returns:
            A stim circuit for the memory experiment.
        """

        if basis == "both":
            return self.get_morphing_XZ_memory_circuit(noise_setup, T, noise_model_generator = noise_model_generator, round_coords = round_coords)
        else:
            raise NotImplementedError("fixed basis memory experiments are not implemented for non-CSS morphing circuits.")

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

    def _logical_bell_state_measurements_and_observables(self, c_round):
        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        
        circ = stim.Circuit()
        
        to_MPP = [stim.PauliString() for _ in range(2*n_logicals)]
        for i in range(n_logicals):
            zero_array = np.zeros((n_qubits + n_logicals), dtype = np.bool_)
            non_trivial_array = np.zeros((n_qubits + n_logicals), dtype = np.bool_)
            non_trivial_array[n_qubits + i] = True
            to_MPP[i] = self.end_cycle_logicals["X"][c_round][i] * stim.PauliString.from_numpy(xs = non_trivial_array, zs = zero_array)
            to_MPP[i + n_logicals] = self.end_cycle_logicals["Z"][c_round][i] * stim.PauliString.from_numpy(xs = zero_array, zs = non_trivial_array)
            
        circ.append("MPP", to_MPP)

        for i in range(2*n_logicals):
            circ.append("OBSERVABLE_INCLUDE", stim.target_rec(i - 2*n_logicals), i)

        circ.append("TICK")

        return circ

    def _logical_mid_cycle_bell_state_measurements_and_observables(self):
        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        
        circ = stim.Circuit()
        
        to_MPP = [stim.PauliString() for _ in range(2*n_logicals)]
        for i in range(n_logicals):
            zero_array = np.zeros((n_qubits + n_logicals), dtype = np.bool_)
            non_trivial_array = np.zeros((n_qubits + n_logicals), dtype = np.bool_)
            non_trivial_array[n_qubits + i] = True
            to_MPP[i] = self.mid_cycle_logicals["X"][i] * stim.PauliString.from_numpy(xs = non_trivial_array, zs = zero_array)
            to_MPP[i + n_logicals] = self.mid_cycle_logicals["Z"][i] * stim.PauliString.from_numpy(xs = zero_array, zs = non_trivial_array)
            
        circ.append("MPP", to_MPP)

        for i in range(2*n_logicals):
            circ.append("OBSERVABLE_INCLUDE", stim.target_rec(i - 2*n_logicals), i)

        circ.append("TICK")

        return circ
        
    def contraction_circuit(self, contraction_round, noise_model = "noiseless_model"):
        """
        Returns a stim circuit for a contraction circuit, from the mid-cycle code to the end-cycle code.

        Arguments:
            contraction_round is an Int that determines which contraction circuits to use.
            noise_model is a surface_sim.Model object that defines the noise model to be used in the circuit. Or, it is the string "noiseless_model", in which case the noise model NoiselessModel is automatically created and used.

        eturns:
            A stim circuit for the contraction circuit.
        """
        n_qubits = self.n_qubits()
        circ = stim.Circuit()
        if noise_model == "noiseless_model":
            qubit_dictionary = {str(i): i for i in range(self.n_qubits())}
            noise_model = NoiselessModel(qubit_dictionary)
        stim_names = list(SQ_GATES.values()) + list(SQ_MEASUREMENTS.values()) + list(SQ_RESETS.values()) + list(TQ_GATES.values()) + list(ANNOTATIONS.values())
        surface_sim_names = list(SQ_GATES.keys()) + list(SQ_MEASUREMENTS.keys()) + list(SQ_RESETS.keys()) + list(TQ_GATES.keys()) + list(ANNOTATIONS.keys())
        
        contraction_circuit = self.contraction_circuits[contraction_round].flattened()

        for i in range(len(contraction_circuit)):
            instruction = contraction_circuit[i]
            
            if instruction.name in ["TICK"] and i != 0:
                circ += noise_model.tick()
            else:
                circ += getattr(noise_model, surface_sim_names[stim_names.index(instruction.name)])([str(j.value) for j in instruction.targets_copy()])
            
            if i == len(contraction_circuit) - 1 and instruction.name != "TICK":
                circ += noise_model.tick()
        return circ

    def morphing_QEC_round(self, noise_model, QEC_round, data_qubit_reset_basis = "None", data_qubit_measurement_basis = "None"):
        """
        Returns a stim circuit for one "measurement round" of QEC, from one end-cycle state to the next. QEC_round should be a half-integer, so that the returned circuit starts in the end-cycle code QEC_round - 1/2, then applies the inverse of the (QEC_round - 1/2)th contraction circuit, then the (QEC_round + 1/2)th contraction, then finishes in the end-cycle code QEC_round + 1/2.

        Arguments:
            noise_model is a surface_sim.Model object that defines the noise model to be used in the circuit.
            QEC_round is a float that should be equal to k+1/2 for some integer k. It defines which contraction circuits to implement in the circuit.
            data_qubit_reset_basis is a Str. When set to "X" or "Z", then the data qubits at the start of the circuit are reset in the corresponding basis; this can be used for example at the start of a memory experiment. When set to "None" (default) then the data qubits are not reset.
            data_qubit_measurement_basis is a Str. When set to "X" or "Z", then the data qubits at the end of the circuit are measured in the corresponding basis; this can be used for example at the end of a memory experiment. When set to "None" (default) then the data qubits are not measured.

        Raises:
            NotImplementedError: data_qubit_reset_basis = 'X' or 'Z' is not supported for non-CSS morphing circuits.
            NotImplementedError: data_qubit_measurement_basis = 'X' or 'Z' is not supported for non-CSS morphing circuits.

        
        Returns:
            A stim circuit for the QEC round.
        """
        if data_qubit_reset_basis in ["X", "Z"]:
            raise NotImplementedError("data_qubit_reset_basis = 'X' or 'Z' is not supported for non-CSS morphing circuits.")
        if data_qubit_measurement_basis in ["X", "Z"]:
            raise NotImplementedError("data_qubit_measurement_basis = 'X' or 'Z' is not supported for non-CSS morphing circuits.")
        
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        previous_round = round(QEC_round - 1 / 2) % n_rounds
        following_round = round(QEC_round + 1 / 2) % n_rounds
        circ = stim.Circuit()
        stim_names = list(SQ_GATES.values()) + list(SQ_MEASUREMENTS.values()) + list(SQ_RESETS.values()) + list(TQ_GATES.values()) + list(ANNOTATIONS.values())
        surface_sim_names = list(SQ_GATES.keys()) + list(SQ_MEASUREMENTS.keys()) + list(SQ_RESETS.keys()) + list(TQ_GATES.keys()) + list(ANNOTATIONS.keys())
        contraction_circuit = self.contraction_circuits[previous_round].flattened()
        
        for i in reversed(range(len(contraction_circuit))):
            instruction = contraction_circuit[i]
            if i in [0, len(contraction_circuit) - 1] and instruction.name == "TICK":
                continue
            
            if instruction.name in ["M", "MX", "MY", "MZ"]:
                new_name = {"M": "reset", "MX": "reset_x", "MY": "reset_y", "MZ": "reset_z"}[instruction.name]
                circ += getattr(noise_model, new_name)([str(j.value) for j in instruction.targets_copy()])
            elif instruction.name in ["TICK"]:
                circ += noise_model.tick()
            else:
                temp = stim.Circuit()
                temp.append(instruction)
                inverted_name = temp.inverse()[0].name
                circ += getattr(noise_model, surface_sim_names[stim_names.index(inverted_name)])([str(j.value) for j in instruction.targets_copy()])
        
        contraction_circuit = self.contraction_circuits[following_round].flattened()

        for i in range(len(contraction_circuit)):
            instruction = contraction_circuit[i]
            if i == 0 and instruction.name != "TICK":
                circ += noise_model.tick()

            
            if instruction.name in ["TICK"]:
                circ += noise_model.tick()
            else:
                circ += getattr(noise_model, surface_sim_names[stim_names.index(instruction.name)])([str(j.value) for j in instruction.targets_copy()])
            
            if i == len(contraction_circuit) - 1 and instruction.name != "TICK":
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
        for i in range(len(self.mid_cycle_stabilisers)):
            contraction_rounds = self.stabiliser_contraction_rounds[i]
            for j in range(len(contraction_rounds)):
                rec_targets = []
                current_contraction_round = contraction_rounds[j]
                past_contraction_round = contraction_rounds[j - 1]
                n_included_rounds = (
                    (current_contraction_round - past_contraction_round - 1)
                    % n_rounds
                ) + 1
                offset = 0
                for k in range(n_included_rounds):
                    this_round = (current_contraction_round - k) % n_rounds
                    rec_targets += [
                        stim.target_rec(l.value - offset - additional_offset)
                        for l in self.stabiliser_measurement_target_recs[this_round][i]
                    ]
                    offset += self.contraction_circuits[this_round].num_measurements
                if self.stabiliser_annotations == []:
                    detector_annotation = (0, 0, 0, 0)
                else:
                    detector_annotation = self.stabiliser_annotations[i]
                detector_circuits[current_contraction_round].append(
                    "DETECTOR", rec_targets, detector_annotation
                )
        return detector_circuits
    
    def _initial_stability_detector_circuits(self):
        
        #defines the detector circuits for the BULK detectors (NOT boundary detectors) in the first n_rounds after the initialisation.
        #This is used in the stability experiment because in general there may be some bulk detectors in the first n_rounds because some stabilisers may be contracting in more than one round.
        
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        for i in range(len(self.mid_cycle_stabilisers)):
            contraction_rounds = self.stabiliser_contraction_rounds[i]
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
                    offset = 0
                    for k in range(n_included_rounds):
                        this_round = (current_contraction_round - k) % n_rounds
                        rec_targets += [
                            stim.target_rec(l.value - offset)
                            for l in self.stabiliser_measurement_target_recs[this_round][i]
                        ]
                        offset += self.contraction_circuits[this_round].num_measurements
                    if self.stabiliser_annotations == []:
                        detector_annotation = (0, 0, 0, 0)
                    else:
                        detector_annotation = self.stabiliser_annotations[i]
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
                    targets_to_include = self.logical_measurement_target_recs[pauli][
                        current_contraction_round
                    ][i]
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
        targets_to_include = [
            [[] for _ in range(n_redundancies)]
            for _ in range(n_rounds - 1)
        ]
        for i in range(n_redundancies):
            redundancy = self.redundancies[i]
            for j in range(len(redundancy)):
                stabiliser_index = redundancy[j]
                if (
                    n_rounds - 1
                    not in self.stabiliser_contraction_rounds[stabiliser_index]
                ):
                    for current_contraction_round in range(
                        self.stabiliser_contraction_rounds[stabiliser_index][0] + 1
                    ):
                        targets_to_include[current_contraction_round][
                            i
                        ] += self.stabiliser_measurement_target_recs[current_contraction_round][stabiliser_index]
        observable_indices = []
        detector_indices = []
        if len(self.meta_checks) == 0:
            observable_indices = np.eye(n_redundancies)
            detector_indices = []
        else:
            h = self.parity_check_matrix()
            hT_kernel = galois.GF(2)(np.zeros((len(self.redundancies), h.shape[0]), dtype = int))
            for i in range(len(self.redundancies)):
                for j in self.redundancies[i]:
                    hT_kernel[i][j] = 1

            #hT_kernel has as its rows the redundancies and as its columns the stabilisers that go into the redundancy.

            meta_check_vectors = galois.GF(2)(np.zeros((len(self.meta_checks),h.shape[0]), dtype = int))
            for i in range(len(self.meta_checks)):
                for j in self.meta_checks[i]:
                    meta_check_vectors[i][j] = 1

            #meta_check_vectors has the redundancies that are a detector as its rows and the stabilisers that go into the redundancy as its columns
            row_reduced = meta_check_vectors.row_reduce()
            null_space_2 = galois.GF(2)(np.eye(row_reduced.shape[1], dtype = int))
            for i in range(len(row_reduced)):
                if len(np.nonzero(row_reduced[i])[0]) > 0:
                    null_space_2 = np.delete(null_space_2, np.nonzero(row_reduced[i])[0][0] - i, 0)
            #null_space_2 has as its rows the observable redundancies and as its columns the stabilisers that go into the redundancy
            observable_indices = galois.GF(2)(np.block([null_space_2.T, hT_kernel.T])).null_space()[:,null_space_2.shape[0]:]
            #detector_indices has as its rows the detectors and as its columns the redundancies that go into the detector
            detector_indices = galois.GF(2)(np.block([meta_check_vectors.T, hT_kernel.T])).null_space()[:,meta_check_vectors.shape[0]:]
            #observable_indices has as its rows the observables and as its columns the redundancies that go into the observable
            
        observable_circuits = [stim.Circuit() for _ in range(n_rounds - 1)]
        observable_argument = 0
        for observable in observable_indices:
            for current_contraction_round in range(n_rounds - 1):
                total_targets_to_include = []
                redundancies_to_include = np.nonzero(observable)[0]
                for redundancy_index in redundancies_to_include:
                    total_targets_to_include += targets_to_include[current_contraction_round][redundancy_index]
                target_occurrences = Counter(total_targets_to_include)
                targets_to_include_here = [target for target in target_occurrences.keys() if target_occurrences[target] % 2 == 1]
                observable_circuits[current_contraction_round].append(
                    "OBSERVABLE_INCLUDE",
                    targets_to_include_here,
                    observable_argument,
                )
            observable_argument += 1
        for detector in detector_indices:
            targets_to_include_here = []
            for current_contraction_round in range(n_rounds - 1):
                offset = self.contraction_circuits[current_contraction_round].num_measurements
                targets_to_include_here = [stim.target_rec(target.value - offset) for target in targets_to_include_here]
                total_targets_to_include = []
                redundancies_to_include = np.nonzero(detector)[0]
                for redundancy_index in redundancies_to_include:
                    total_targets_to_include += targets_to_include[current_contraction_round][redundancy_index]
                target_occurrences = Counter(total_targets_to_include)
                targets_to_include_here += [target for target in target_occurrences.keys() if target_occurrences[target] % 2 == 1]
            observable_circuits[n_rounds - 2].append(
                "DETECTOR",
                targets_to_include_here,
                (0,0,0,0),
            )
        return observable_circuits
