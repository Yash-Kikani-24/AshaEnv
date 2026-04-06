# graders/ — scoring components that evaluate a completed episode trajectory.
#
# Each grader is an independent BaseGrader subclass that scores one specific
# aspect of the agent's behaviour. CompositeGrader combines them with weights:
#
#   DiagnosisGrader   (40%) — did the agent identify the correct disease?
#   SafetyGrader      (25%) — did the agent avoid dangerous mistakes?
#   EfficiencyGrader  (20%) — did the agent reach its conclusion without wasted steps?
#   ReferralGrader    (15%) — did the agent send the patient to the right facility?
#
# All graders return a float in [0.0, 1.0]. The composite score is also in [0.0, 1.0].
# Graders are called after an episode ends via: grader.grade(trajectory, true_diagnosis, patient)
