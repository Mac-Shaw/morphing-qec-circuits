from surface_sim.models import SI1000NoiseModel
from surface_sim.setups.setup import SQ_GATES, TQ_GATES, SQ_MEASUREMENTS, SQ_RESETS

class ArbitraryGatesSI1000NoiseModel(SI1000NoiseModel):
    """
    A subclass of SI1000NoiseModel that allows arbitrary single-qubit gates, measurements, and resets, and arbitrary two-qubit gates.
    """
    _supported_operations: list[str] = list(SQ_GATES) + list(TQ_GATES) + list(SQ_MEASUREMENTS) + list(SQ_RESETS)