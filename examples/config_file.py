class config():
    def __init__(self):
        self.n_init             = 10             					# Number of initial bounds from an Symmetric Latin hypercube design0
        self.max_evals          = 10001                                                 # Maximum number of evaluations
        self.n_trust_regions    = 10                                                    # Number of trust regions
        self.batch_size         = 20                                                    # How large batch size TuRBO uses
        self.verbose            = True                                                  # Print information from each batch
        self.use_ard            = True                                                  # Set to true if you want to use ARD for the GP kernel
        self.max_cholesky_size  = 6000                                                  # When we switch from Cholesky to Lanczos
        self.n_training_steps   = 500                                                   # Number of steps of ADAM to learn the hypers
        self.min_cuda           = 1024         						# Run on the CPU for small datasets
        self.device             = "cpu"         					# "cpu" or "cuda"
        self.dtype              = "float64"     					# float64 or float32
        self.turbo              = "TurboM"      					# Type of turbo (e.g., TurboM, Turbo1)

        self.low_dim            = 10            					# The projection space dimension
        self.high_dim           = 25            					# The original parameters space dimension
        self.method             = "pre"         					# The sammple-efficient learning method (e.g., turbo, hesbo, pure, and pre)
        self.motion             = "trot"        					# The desired motion of the robot (e.g., trot, bound, and jump)
        self.obj                = "a1"   						# The BO cost function of a robot (e.g., a1, b1)
        self.opt                = "add_bo"     						# Optimization method for the BayesOpt paper
        self.loss               = "Neg_ei"      					# Loss function for the BayesOpt paper
        self.ea_type            = "no"          					# Autoencoder type (e.g., ea, vae, cvae, no)
        _path                   = '/data/alireza/high_bo/BayesOpt/tests/results/'+self.obj
        self.data_samples_dir   = _path+'/turbo_training_points_'+str(self.high_dim)+'_'+self.motion+'/'
        self.model_checkpoint   = _path+'/checkpoint_'+self.motion+'/'
        self.figs_dir           = _path+'/Figs/'
        self.path_to_tasks      = 'tasks/all_tasks'
        self.path_to_turbo      = 'examples/turbo'
        self.uneven_terrain     = False                                                # Whether the environment is uneven or no
        self.uncertain_actuator = False                                               # Whether there is uncertainity in the actuation or not
        self.var                = 0                                                   # Variance of the noise added to the output of obj
        self.n_epochs           = 2000                                                # Number of epochs for training the model
        self.bs                 = 16                                                   # How large batch size EA training uses
        self.lr                 = 0.001                                               # Learning rate of EA training

        self.logdir             = ""                                                  # Where to log the trajectories
        self.use_real_robot     = True                                               # Whether to use real robot or simulator
        self.show_gui           = False                                                # Whether to show GUI
        self.max_time_secs      = 25                                                   # Maximum time to run the robot
        self.video_dir          = "/data/alireza/high_bo/BayesOpt/video/out"          # Where to save the video
        self.use_gamepad        = False                                               # This flag is used for real robot
