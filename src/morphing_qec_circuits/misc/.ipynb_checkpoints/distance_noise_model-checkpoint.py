import stim
from surface_sim.models import NoiselessModel
from surface_sim.setups import Setup


class DistanceNoiseModel(NoiselessModel):
    """
    A noise model suited to calculating the circuit-level distance of a CSS circuit. It inserts only single-qubit X and/or Z errors at each time-step.
        For CSS circuits the circuit-level distance of the circuit calculated using this noise model is equal to the circuit-level distance under a full
        circuit-level noise model (which contains, for example, full single- and two-qubit depolarising channels after single- and two-qubit gates).
        If your circuit is not CSS, then one should use a different noise model to calculate the cicuit-level distance (surface_sim.CircuitNoiseModel is an
        appropriate choice here).
    """
    def __init__(
        self, qubit_inds: dict[str, int], setup: Setup, noise_basis: str
    ) -> None:
        """Creates a DistanceNoiseModel.

        Args:
            qubit_inds: a dictionary with keys that are strings corresponding to the names of qubits, and items that are integers corresponding to the qubit
                indices in a stim.Circuit.
            setup: a surface_sim.Setup object from which to define the DistanceNoiseModel. morphing_qec_circuits.UniformSetup is an appropriate choice here.
            noise_basis: a Str that is "X", "Z", or "both". If set to "X" or "Z", then only X or Z errors will be inserted (respectively), if "both", then
                both X and Z errors are inserted.

        Raises:
            ValueError: Noise_basis should be X or Z or both. (If your code/circuit is not CSS, then you should use a different noise model when calculating the circuit-level distance)
        """
        super().__init__(qubit_inds)
        self._setup = setup
        if noise_basis in ["X", "Z"]:
            self._noise_bases = [noise_basis]
        elif noise_basis == "both":
            self._noise_bases = ["X", "Z"]
        else:
            raise ValueError(
                '''Noise_basis should be X or Z or both.
                (If your code/circuit is not CSS, then you should use a different noise model when calculating the circuit-level distance)'''
            )

    def tick(self) -> stim.Circuit:
        circ = stim.Circuit()
        for qubit, ind in self._qubit_inds.items():
            prob = self.param("sq_error_prob", qubit)
            for noise_basis in self._noise_bases:
                circ.append(noise_basis + "_ERROR", [ind], [prob])
        circ.append("TICK")
        return circ
