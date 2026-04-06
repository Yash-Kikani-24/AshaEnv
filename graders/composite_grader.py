"""
CompositeGrader — combines all sub-graders into a single episode score.

This is the grader you should call from agents and evaluation scripts.
It runs all four sub-graders independently and merges their scores using
fixed weights that reflect clinical priority:

    Diagnosis   40% — identifying the disease is the primary task.
    Safety      25% — dangerous mistakes (missed emergencies, harmful drugs) are costly.
    Efficiency  20% — good clinical reasoning uses the fewest necessary steps.
    Referral    15% — sending the patient to the right facility matters.

The final score is clamped to [0.0, 1.0] and rounded to 4 decimal places.
"""

from .base_grader import BaseGrader
from .diagnosis_grader import DiagnosisGrader
from .safety_grader import SafetyGrader
from .efficiency_grader import EfficiencyGrader
from .referral_grader import ReferralGrader


class CompositeGrader(BaseGrader):
    def __init__(self):
        # Instantiate each sub-grader once and reuse across many grade() calls.
        self.diagnosis_grader = DiagnosisGrader()
        self.safety_grader = SafetyGrader()
        self.efficiency_grader = EfficiencyGrader()
        self.referral_grader = ReferralGrader()

    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Run all four sub-graders and return the weighted composite score.

        Each sub-grader independently inspects the same trajectory and returns
        a score in [0.0, 1.0]. This method combines them with fixed weights.

        Args:
            trajectory:     Ordered list of action strings from the episode.
            true_diagnosis: Ground-truth disease_id for this episode.
            patient:        Full patient dict (from env.get_state()["patient"]).

        Returns:
            Weighted composite score in [0.0, 1.0], rounded to 4 decimal places.
        """
        d = self.diagnosis_grader.grade(trajectory, true_diagnosis, patient)
        s = self.safety_grader.grade(trajectory, true_diagnosis, patient)
        e = self.efficiency_grader.grade(trajectory, true_diagnosis, patient)
        r = self.referral_grader.grade(trajectory, true_diagnosis, patient)

        # Weighted sum — weights sum to 1.0.
        score = 0.40 * d + 0.25 * s + 0.20 * e + 0.15 * r
        return round(max(0.0, min(1.0, score)), 4)
