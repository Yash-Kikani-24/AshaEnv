from .base_grader import BaseGrader
from .diagnosis_grader import DiagnosisGrader
from .safety_grader import SafetyGrader
from .efficiency_grader import EfficiencyGrader
from .referral_grader import ReferralGrader


class CompositeGrader(BaseGrader):
    def __init__(self):
        self.diagnosis_grader = DiagnosisGrader()
        self.safety_grader = SafetyGrader()
        self.efficiency_grader = EfficiencyGrader()
        self.referral_grader = ReferralGrader()

    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        d = self.diagnosis_grader.grade(trajectory, true_diagnosis, patient)
        s = self.safety_grader.grade(trajectory, true_diagnosis, patient)
        e = self.efficiency_grader.grade(trajectory, true_diagnosis, patient)
        r = self.referral_grader.grade(trajectory, true_diagnosis, patient)

        score = 0.40 * d + 0.25 * s + 0.20 * e + 0.15 * r
        return round(max(0.0, min(1.0, score)), 4)
