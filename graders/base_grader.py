from abc import ABC, abstractmethod


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """Return a float between 0.0 and 1.0. Must never return a constant value."""
        pass
