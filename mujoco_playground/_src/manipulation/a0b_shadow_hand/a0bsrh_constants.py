# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for the A0B arm with a Shadow Hand (a0bsrh)."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "a0b_shadow_hand"
CUBE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube.xml"
PICK_XML = ROOT_PATH / "xmls" / "scene_mjx_pick_reorient.xml"
TWO_ARMS_XML = ROOT_PATH / "xmls" / "scene_mjx_two_arms.xml"
LEAP_XML_PATH = mjx_env.ROOT_PATH / "manipulation" / "leap_hand" / "xmls"

ARM_JOINT_NAMES = [f"a0b_joint{i}" for i in range(6)]

# All hand joints, in qpos order.
HAND_JOINT_NAMES = [
    "wrist_joint1",
    "wrist_joint0",
    # index
    "ind_joint3",
    "ind_joint2",
    "ind_joint1",
    "ind_joint0",
    # middle
    "mid_joint3",
    "mid_joint2",
    "mid_joint1",
    "mid_joint0",
    # ring
    "rin_joint3",
    "rin_joint2",
    "rin_joint1",
    "rin_joint0",
    # little
    "lit_joint4",
    "lit_joint3",
    "lit_joint2",
    "lit_joint1",
    "lit_joint0",
    # thumb
    "thu_joint4",
    "thu_joint3",
    "thu_joint2",
    "thu_joint1",
    "thu_joint0",
]

# Hand actuators (named after the joint they drive). The distal *_joint0 of
# the four fingers are not actuated; fixed tendons couple them to *_joint1.
HAND_ACTUATOR_NAMES = [
    "wrist_joint1",
    "wrist_joint0",
    "ind_joint3",
    "ind_joint2",
    "ind_joint1",
    "mid_joint3",
    "mid_joint2",
    "mid_joint1",
    "rin_joint3",
    "rin_joint2",
    "rin_joint1",
    "lit_joint4",
    "lit_joint3",
    "lit_joint2",
    "lit_joint1",
    "thu_joint4",
    "thu_joint3",
    "thu_joint2",
    "thu_joint1",
    "thu_joint0",
]
ARM_ACTUATOR_NAMES = ARM_JOINT_NAMES

NQ_HAND = len(HAND_JOINT_NAMES)
NU_HAND = len(HAND_ACTUATOR_NAMES)

FINGERTIP_NAMES = ["th_tip", "ff_tip", "mf_tip", "rf_tip", "lf_tip"]

# Home pose of the pick tasks: palm down with the grasp site 0.72 m in front
# of the arm base and 0.2 m above the table (IK solution; the arm folds in the
# horizontal plane, the only way this arm keeps the palm down at that reach).
ARM_HOME_QPOS = [-1.183, -0.178, 2.288, -0.183, -1.117, 0.149]
# Fingers slightly bent and thumb opposed, in HAND_JOINT_NAMES order.
HAND_HOME_QPOS = [
    0.0,
    0.0,  # wrist
    0.0,
    0.1,
    0.0096,
    0.01,  # index
    0.0,
    0.1,
    0.0096,
    0.01,  # middle
    0.0,
    0.1,
    0.0096,
    0.01,  # ring
    0.0097,
    0.0,
    0.1,
    0.0096,
    0.01,  # little
    0.3,
    0.3,
    0.0,
    0.0,
    0.0,  # thumb
]
