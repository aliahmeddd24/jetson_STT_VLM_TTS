"""g1action: local-only, allowlisted gesture/step server for the Unitree G1.

Safety model (see docs/ARCHITECTURE.md): dry-run by default; the LLM can only name
allowlisted gestures; arm_sdk weight is always ramped from the measured pose; steps are
lateral-only, small, time-limited, non-continuous and require an operator flag; a
heartbeat watchdog stops everything; no damping/zero-torque calls exist in this package.
"""
__version__ = "0.1.0"

# G1 29-DoF motor index -> URDF joint name (unitree_sdk2 G1JointIndex order).
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEFT_ARM = list(range(15, 22))
RIGHT_ARM = list(range(22, 29))
ARMS = LEFT_ARM + RIGHT_ARM
