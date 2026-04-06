"""
Abstract base class that every grader must inherit from.

Defines the single interface all graders share: a grade() method that takes
the episode trajectory, the ground-truth diagnosis, and the full patient dict,
and returns a normalised score in [0.0, 1.0].

New graders (e.g. CommunicationGrader, CostGrader) must subclass BaseGrader
and implement grade(). CompositeGrader picks them up by calling grade() directly.
"""

from abc import ABC, abstractmethod


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Score one completed episode.

        Args:
            trajectory:     Ordered list of action strings taken during the episode
                            (e.g. ["ask_symptom:pallor", "diagnose:severe_anaemia"]).
            true_diagnosis: The ground-truth disease_id for this episode
                            (from env.get_state()["true_diagnosis"]).
            patient:        Full patient dict including demographics, true symptoms,
                            history, comorbidities, etc. (from env.get_state()["patient"]).

        Returns:
            A float in [0.0, 1.0]. Must never return a constant value — the score
            must vary meaningfully based on the quality of the trajectory.
        """
        pass
