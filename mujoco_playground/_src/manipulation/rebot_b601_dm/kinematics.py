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

"""Pure-JAX kinematics of a site on the reBot B601-DM arm."""

from typing import Sequence, Tuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, quat)
  return mat.reshape(3, 3)


class SiteKinematics:
  """Forward kinematics and damped least-squares IK for a site on a serial arm.

  Body offsets and hinge axes are read once from the MjModel, so both functions
  are plain JAX and work with any MJX backend. Only hinges located at their
  body origin are supported, which holds for the B601 arm joints.
  """

  def __init__(
      self,
      mj_model: mujoco.MjModel,
      joint_names: Sequence[str],
      site_name: str,
  ):
    jids = [mj_model.joint(name).id for name in joint_names]
    joint_of_body = {}
    for i, jid in enumerate(jids):
      if mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or np.any(
          mj_model.jnt_pos[jid]
      ):
        raise ValueError(
            f"{joint_names[i]} must be a hinge at its body origin."
        )
      joint_of_body[int(mj_model.jnt_bodyid[jid])] = (i, mj_model.jnt_axis[jid])

    site_id = mj_model.site(site_name).id
    chain = []
    body = int(mj_model.site_bodyid[site_id])
    while body != 0:
      chain.append(body)
      body = int(mj_model.body_parentid[body])
    if not set(joint_of_body).issubset(chain):
      raise ValueError(f"All joints must lie on the path to {site_name}.")

    # (body offset, body rotation, joint index, joint axis, axis skew matrix).
    self._links = []
    for body in reversed(chain):
      pos = mj_model.body_pos[body].copy()
      rot = _quat_to_mat(mj_model.body_quat[body])
      if body in joint_of_body:
        idx, axis = joint_of_body[body]
        x, y, z = axis
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        self._links.append((pos, rot, idx, axis.copy(), skew))
      else:
        self._links.append((pos, rot, None, None, None))
    self._site_pos = mj_model.site_pos[site_id].copy()
    self._site_rot = _quat_to_mat(mj_model.site_quat[site_id])
    self._num_joints = len(jids)
    self.lower, self.upper = mj_model.jnt_range[jids].T

  def _forward_with_joints(
      self, q: jax.Array
  ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns site position and rotation, and world joint axes and origins."""
    pos, rot = jp.zeros(3), jp.eye(3)
    axes = [None] * self._num_joints
    origins = [None] * self._num_joints
    for body_pos, body_rot, idx, axis, skew in self._links:
      pos = pos + rot @ body_pos
      rot = rot @ body_rot
      if idx is not None:
        axes[idx] = rot @ axis
        origins[idx] = pos
        # Rodrigues' rotation about the joint axis.
        rot = rot @ (
            jp.eye(3)
            + jp.sin(q[idx]) * skew
            + (1 - jp.cos(q[idx])) * (skew @ skew)
        )
    site_pos = pos + rot @ self._site_pos
    site_rot = rot @ self._site_rot
    return site_pos, site_rot, jp.stack(axes), jp.stack(origins)

  def forward(self, q: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Returns the site position and rotation matrix for joint angles q."""
    site_pos, site_rot, _, _ = self._forward_with_joints(q)
    return site_pos, site_rot

  def inverse(
      self,
      q: jax.Array,
      target_pos: jax.Array,
      target_rot: jax.Array,
      iterations: int = 3,
      damping: float = 1e-3,
  ) -> Tuple[jax.Array, jax.Array]:
    """Damped least-squares IK starting from q, within joint limits.

    Returns:
      The joint angles and the remaining site position error.
    """

    def step(_, q):
      pos, rot, axes, origins = self._forward_with_joints(q)
      err = jp.concatenate([
          target_pos - pos,
          0.5 * jp.sum(jp.cross(rot.T, target_rot.T), axis=0),
      ])
      jac = jp.concatenate([jp.cross(axes, pos - origins).T, axes.T])
      dq = jac.T @ jp.linalg.solve(jac @ jac.T + damping * jp.eye(6), err)
      return jp.clip(q + dq, self.lower, self.upper)

    q = jax.lax.fori_loop(0, iterations, step, q)
    pos, _ = self.forward(q)
    return q, jp.linalg.norm(target_pos - pos)
