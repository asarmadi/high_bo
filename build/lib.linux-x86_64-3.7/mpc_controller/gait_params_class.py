import numpy as np
class gait_params_class:
    def __init__(self, robot_name, motion_name):

        self.robot_name = robot_name
        self.motion_name = motion_name
        if robot_name == 'A1':
           self.q_weights = (0.5, 0.5, 0.02, 0, 0, 1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0)
           self.u_weights = (1e-5,)*12
        elif robot_name == 'B1':
           '''
           self.q_weights = (0.5, 0.5, 0.02, 0, 0, 1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0)
           self.u_weights = (1e-5,)*12
           '''
           self.q_weights = (2.3425317e-01, 3.6345094e-02, 1.7508334e-01, 2.1901321e-01, 1.3102666e-01,
                             9.1430712e-01, 7.2001421e-01, 7.8695370e-03, 5.0355709e-01, 1.8775916e-02,
                             4.7876135e-02, 1.9686129e-03, 6.4953208e-02)
           self.u_weights = (1.9793036e-07, 1.0266554e-07, 9.2998413e-08, 3.4948318e-08, 5.3663339e-08,
                             7.7232620e-08, 1.8162412e-08, 7.1768952e-08, 6.8000297e-08, 3.8215934e-08,
                             4.6250541e-08, 4.6046690e-08)
        self.init_X = np.array(self.q_weights+self.u_weights)
#        self.init_X = np.array(self.q_weights)

