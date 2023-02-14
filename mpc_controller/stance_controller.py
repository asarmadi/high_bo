"""The swing leg controller class."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import mpc_osqp as convex_mpc
from mpc_controller import com_velocity_estimator

class StanceController():
  """Controls the swing leg position using Raibert's formula.

  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.

  """
  def __init__(self,robot):
    self._robot = robot
    self._cpp_mpc = convex_mpc.ConvexMpc(
        50,
        np.array((0.183142146, -0.001379002, -0.027956055, -0.001379002, 0.756327752, 0.000193774, 0, 0.000193774, 0.783777558)),
        4,
        10,
        0.025,
        (0.5, 0.5, 0.02, 0, 0, 1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0),
        (1e-5,)*12,
        convex_mpc.QPOASES
    )
    self.estimator = com_velocity_estimator.COMVelocityEstimator(robot, window_size=20)


  def get_action(self):
      desired_com_position         = np.array((0., 0., 0.45), dtype=np.float64)
      desired_com_velocity         = np.array((0., 0., 0.), dtype=np.float64)
      desired_com_roll_pitch_yaw   = np.array((0., 0., 0.), dtype=np.float64)
      desired_com_angular_velocity = np.array((0., 0., 0.), dtype=np.float64)

      foot_contact_state = np.array([2 for e in range(4)], dtype=np.int32)

      com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw(), dtype=np.float64)
      com_roll_pitch_yaw[2] = 0

      predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
        [0],  #com_position
        np.asarray(self.estimator.com_velocity_body_frame, dtype=np.float64),  #com_velocity
        np.array(com_roll_pitch_yaw, dtype=np.float64),  #com_roll_pitch_yaw
        np.asarray(self._robot.GetBaseRollPitchYawRate(), dtype=np.float64),  #com_angular_velocity
        foot_contact_state,  #foot_contact_states
        np.array(self._robot.GetFootPositionsInBaseFrame().flatten(), dtype=np.float64),  #foot_positions_base_frame
        (0.45, 0.45, 0.45, 0.45),  #foot_friction_coeffs
        desired_com_position,  #desired_com_position
        desired_com_velocity,  #desired_com_velocity
        desired_com_roll_pitch_yaw,  #desired_com_roll_pitch_yaw
        desired_com_angular_velocity  #desired_com_angular_velocity
      )
      contact_forces = {}
      for i in range(4):
        contact_forces[i] = np.array(predicted_contact_forces[i * 3:(i + 1) * 3])
#      contact_forces = {}
 #     for i in range(self._robot.num_legs):
  #        contact_forces[i] = np.array([0.0, 0.0, 9.8]) * 20

      action = []
      for leg_id, force in contact_forces.items():
          motor_torques = self._robot.MapContactForceToJointTorques(leg_id, force)
          for joint_id, torque in motor_torques.items():
              action.extend((0, 0, 0, 0, torque))
      return action
