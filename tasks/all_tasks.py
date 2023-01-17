import gc
import numpy as np
import random
import torch
import pybullet
import time
import scipy.interpolate

from models.vae_models import Decoder
from examples.config_file import config
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
#from mpc_controller import torque_stance_leg_controller_quadprog as torque_stance_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller.gait_params_class import gait_params_class
import mpc_osqp
from perlin_noise import PerlinNoise
import pybullet_data

from pybullet_utils import bullet_client
from robots import a1, b1
from robots import robot_config
from robots.gamepad import gamepad_reader

class unitree_cost():
    def __init__(self, seed):
        self.bounds=np.array([[0,1]])
        self.ig = config()
        self.gait_params = gait_params_class(self.ig.obj.upper(),self.ig.motion)
        if self.ig.ea_type == "vae" or self.ig.ea_type == "cvae":
           self.lb    = -6 * np.ones(self.ig.low_dim)
           self.ub    =  6 * np.ones(self.ig.low_dim)
        elif self.ig.ea_type == 'ae':
           self.lb    = 0 * np.ones(self.ig.low_dim)
           self.ub    = 1 * np.ones(self.ig.low_dim)
        else:
           self.lb       = 0    * np.ones(self.ig.high_dim)
           self.ub       = 1    * np.ones(self.ig.high_dim)
           self.ub[-12:] = 1e-5
        for i in range(self.ig.high_dim-1):
            self.bounds = np.append(self.bounds, [[0,1]], axis=0)

        self.seed = seed
        self._STANCE_DURATION_SECONDS = [0.13] * 4
        if self.ig.motion == 'trot':
           self.v_des = (1.6,0.0,0.0,0.0)
           self._DUTY_FACTOR = [0.6] * 4
           self._INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

           self._INIT_LEG_STATE = (
               gait_generator_lib.LegState.SWING,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.SWING,
           )
        elif self.ig.motion == 'tripod':
           self.v_des = (1.6,0.0,0.0,0.0)
           self._DUTY_FACTOR = [.8] * 4
           self._INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]
           self._INIT_LEG_STATE = (
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.SWING,
               )
        elif self.ig.motion == 'stand':
           self.v_des = (0.0,0.0,0.0,0.0)
           self._DUTY_FACTOR = [1.0] * 4
           self._STANCE_DURATION_SECONDS = [self.ig.max_time_secs] * 4
           self._INIT_PHASE_FULL_CYCLE = [0., 0., 1., 1.]
           self._INIT_LEG_STATE = (
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
               gait_generator_lib.LegState.STANCE,
           )

        elif self.ig.motion == 'jump':
           if self.ig.obj == 'a1':
              self.v_des = (0.5,0.0,0.0,0.0)
              self._STANCE_DURATION_SECONDS = [0.3] * 4
              self._DUTY_FACTOR = [0.7] * 4
              self._INIT_PHASE_FULL_CYCLE = [0.5, 0.5, 0.5, 0.5]
              self._INIT_LEG_STATE = (
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
              )
           elif self.ig.obj == 'b1':
              self.v_des = (0.1,0.0,0.0,0.0)
              self._STANCE_DURATION_SECONDS = [0.3] * 4
              self._DUTY_FACTOR = [0.7] * 4
              self._INIT_PHASE_FULL_CYCLE = [0.5, 0.5, 0.5, 0.5]
              self._INIT_LEG_STATE = (
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
                  gait_generator_lib.LegState.SWING,
              )


    def gen_speed(self, t):
        time_points = (0, 5, 10, 15, 20, 25, 30)
        speed_points = (self.v_des,)* len(time_points)
        speed = scipy.interpolate.interp1d(time_points,
                                           speed_points,
                                           kind="previous",
                                           fill_value="extrapolate",
                                           axis=0)(t)
        return speed[0:3], speed[3], False

    def _setup_controller(self, mpc_weights):
        desired_speed = (0, 0)
        desired_twisting_speed = 0
        gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self.robot,
            stance_duration=self._STANCE_DURATION_SECONDS,
            duty_factor=self._DUTY_FACTOR,
            initial_leg_phase=self._INIT_PHASE_FULL_CYCLE,
            initial_leg_state=self._INIT_LEG_STATE)
        window_size = 20 if not self.ig.use_real_robot else 1
        state_estimator = com_velocity_estimator.COMVelocityEstimator(
            self.robot, window_size=window_size)
        sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self.robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_height=self.robot.MPC_BODY_HEIGHT,
            foot_clearance=0.01)

        st_controller = torque_stance_leg_controller.TorqueStanceLegController(
            self.robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_body_height=self.robot.MPC_BODY_HEIGHT,
            mpc_weights=mpc_weights,
            qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
            )

        self.controller = locomotion_controller.LocomotionController(
            robot=self.robot,
            gait_generator=gait_generator,
            state_estimator=state_estimator,
            swing_leg_controller=sw_controller,
            stance_leg_controller=st_controller,
            clock=self.robot.GetTimeSinceReset)

    def _update_controller_params(self, lin_speed, ang_speed):
        self.controller.swing_leg_controller.desired_speed = lin_speed
        self.controller.swing_leg_controller.desired_twisting_speed = ang_speed
        self.controller.stance_leg_controller.desired_speed = lin_speed
        self.controller.stance_leg_controller.desired_twisting_speed = ang_speed
  
    def f(self, x, noisy=False, fulldim=False, record_file_name='', eval=False):
#        print(x)
      #  self.ig.show_gui = False
        if x == []:
           x = self.gait_params.init_X

        if self.ig.show_gui and not self.ig.use_real_robot:
          p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
          if self.ig.video_dir:
             log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.ig.video_dir+str(np.random.randint(100, size=1)[0])+'.mp4')
        else:
          p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        p.setPhysicsEngineParameter(numSolverIterations=30)
        p.setTimeStep(0.001)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=10, cameraPitch=-20, cameraTargetPosition=[0, 0, 1.0])
#        p.setRealTimeSimulation(1)
        
        height_field_terrain_shape = None
        if self.ig.uneven_terrain:
           pybullet.removeBody(0)
           length_per_index = 0.05 # fixed as only then the contact forces are simulated properly
           patch_length_x = 5 # subject to the trajectory length
           patch_length_y = 5 # subject to the trajectory length
           numHeightfieldRows = int(patch_length_x / length_per_index)
           numHeightfieldColumns = int(patch_length_y / length_per_index)
           terrainMap = np.zeros((numHeightfieldRows,numHeightfieldColumns))
           noise = PerlinNoise(octaves=10, seed=46)
           for i in range(numHeightfieldRows):
               for j in range(numHeightfieldColumns):
                   h = 0.02 * noise([i/numHeightfieldRows, j/numHeightfieldColumns])
                   terrainMap[i][j] = h
           heightfieldData = terrainMap.T.flatten()
           if height_field_terrain_shape == None:
              height_field_terrain_shape = pybullet.createCollisionShape(shapeType = pybullet.GEOM_HEIGHTFIELD,
                                                                            meshScale=[ length_per_index , length_per_index ,1],
                                                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                                            heightfieldData=heightfieldData,
                                                                            numHeightfieldRows=numHeightfieldRows,
                                                                            numHeightfieldColumns=numHeightfieldColumns)
           else:
              height_field_terrain_shape = pybullet.createCollisionShape(shapeType = pybullet.GEOM_HEIGHTFIELD,
                                                                            meshScale=[ length_per_index , length_per_index ,1],
                                                                            heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                                            heightfieldData=heightfieldData,
                                                                            numHeightfieldRows=numHeightfieldRows,
                                                                            numHeightfieldColumns=numHeightfieldColumns,
                                                                            replaceHeightfieldIndex=height_field_terrain_shape)


           terrain_id  = pybullet.createMultiBody(baseMass = 0, baseCollisionShapeIndex = height_field_terrain_shape)
           pybullet.changeVisualShape(terrain_id, -1, rgbaColor=[.101, .67, .33, 1.0])


        # Construct robot class:
        if self.ig.use_real_robot:
          if self.ig.obj == 'a1':
             from motion_imitation.robots import a1_robot
             self.robot = a1_robot.A1Robot(
                 pybullet_client=p,
                 motor_control_mode=robot_config.MotorControlMode.HYBRID,
                 enable_action_interpolation=False,
                 time_step=0.002,
                 action_repeat=1)
          elif self.ig.obj == 'b1':
             from motion_imitation.robots import b1_robot
             self.robot = b1_robot.B1Robot(
                 pybullet_client=p,
                 motor_control_mode=robot_config.MotorControlMode.HYBRID,
                 enable_action_interpolation=False,
                 time_step=0.002,
                 action_repeat=1)
        else:
          if self.ig.obj == 'a1':
             self.robot = a1.A1(p,
                           motor_control_mode=robot_config.MotorControlMode.HYBRID,
                           enable_action_interpolation=False,
                           reset_time=2,
                           time_step=0.002,
                           action_repeat=1)
          elif self.ig.obj == 'b1':
             self.robot = b1.B1(p,
                           motor_control_mode=robot_config.MotorControlMode.HYBRID,
                           enable_action_interpolation=False,
                           reset_time=2,
                           time_step=0.002,
                           action_repeat=1)

        if self.ig.ea_type != 'no':
           if self.ig.ea_type == 'cvae':
              decoder = Decoder(in_dim=self.ig.high_dim, latent_dims=self.ig.low_dim+1)
           else:
              decoder = Decoder(in_dim=self.ig.high_dim, latent_dims=self.ig.low_dim)
           path_n = self.ig.model_checkpoint+self.ig.ea_type+ '_decoder'+'_in'+str(self.ig.high_dim)+'_'+'proj'+str(self.ig.low_dim)+'.pth'
           decoder.load_state_dict(torch.load(path_n))

           X = torch.Tensor(x)
           if self.ig.ea_type == 'cvae':
              Y = torch.Tensor([7.])
              Z = torch.cat([X, Y], 0)
           else:
              Z = X
           weights = decoder(Z).cpu().detach().numpy()
        else:
           weights = x
#        print(weights)
#        input('enter')
        self._setup_controller(weights)
        self.controller.reset()

        start_time = self.robot.GetTimeSinceReset()
        current_time = start_time
        com_vels, imu_rates, actions = [], [], []
        err = 0
        if self.ig.use_gamepad:
           gamepad = gamepad_reader.Gamepad()
           command_function = gamepad.get_command
        else:
           command_function = self.gen_speed
  #      z_com = []
#        p.setRealTimeSimulation(1)
        while current_time - start_time < self.ig.max_time_secs:
#          time.sleep(0.02) #on some fast computer, works better with sleep on real A1?
          start_time_robot = current_time
          start_time_wall = time.time()
          # Updates the controller behavior parameters.
          lin_speed, ang_speed, e_stop = command_function(current_time)
          if e_stop:
             print("E-stop kicked, exiting...")
             break
          self._update_controller_params(lin_speed, ang_speed)
          self.controller.update()
          hybrid_action, _ = self.controller.get_action()
          pcom = np.array(self.robot.GetBasePosition()).copy()
          x_height = pcom[0]
          z_com    = pcom[2]
       #   if np.isnan(z_com):
        #     print(hybrid_action)
         #    break
#          print(z_com)
          time_end = time.time()
#          print(time_end-start_time_wall)
          vcom = np.array(self.robot.GetBaseVelocity()).copy()
          self.robot.Step(hybrid_action)
 #         print(hybrid_action)
          current_time = self.robot.GetTimeSinceReset()
      #    print(current_time)
          if not self.ig.use_real_robot:
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if actual_duration < expected_duration:
              time.sleep(expected_duration - actual_duration)
          err += (np.sum((vcom-self.v_des[0:3])**2)).reshape(-1,1)
          FALL = False
          if (self.ig.motion == 'trot')  and (z_com < 0.17 or np.isnan(z_com) or z_com > 0.42):
             FALL = True
             break
          if (self.ig.motion == 'jump')  and (z_com < 0.1  or np.isnan(z_com) or z_com > 0.52):
             FALL = True
             break
          if (self.ig.motion == 'stand') and (z_com < 0.17 or np.isnan(z_com) or z_com > 0.36):
             FALL = True
             break
        if self.ig.use_gamepad:
           gamepad.stop()
        o = current_time-start_time
        p.removeAllUserDebugItems()
        p.resetSimulation()
        p.disconnect()
#        print(np.isnan(err), FALL, np.sqrt(err), current_time-start_time)
 #       print(np.sqrt(err))
        if np.isnan(x_height):
           x_height = 0
        if np.isnan(err) or FALL:
           if eval:
              return x_height, None, o
           else:
              return -1*x_height
        if eval:
           return x_height, np.sqrt(err), o
        else:
           return np.sqrt(err)

    def __call__(self, weights):
        _, fs, o = self.f(weights, eval=True)
        if fs == None:
           if self.ig.ea_type != 'no':
              return 100.
           else:
              return 100.
        else:
           return fs[0][0]
