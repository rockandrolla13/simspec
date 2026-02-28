"""Protocol definitions for stochastic processes in the RFQ simulator."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StochasticProcess(Protocol):
    """Unified interface for all stochastic processes in the simulator.

    This protocol defines the common interface for:
    - HawkesProcess (self-exciting arrivals)
    - ImbalanceProcess (AR(1) buy/sell flow)
    - RegimeProcess (Markov chain)
    - StreetLeanProcess (OU process)
    """

    def step(self, dt: float = 1.0) -> float:
        """Advance the process by dt time units and return new state value."""
        ...

    def reset(self, initial_value: float | None = None) -> None:
        """Reset process to initial state or specified value."""
        ...

    @property
    def value(self) -> float:
        """Current state value of the process."""
        ...
