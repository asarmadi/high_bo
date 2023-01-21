LD_LIBRARY_PATH=/data/alireza/high_bo/BayesOpt/lib/
export PYTHONPATH="./lib/:$PYTHONPATH"

SEED=12500

#python examples/plots.py --seed=${SEED}

xvfb-run -a python examples/find_statistics.py --seed=${SEED}


#for N_S in 1 2 3 4 5 6 7 8; do
#python examples/turbo_ex.py --seed=${SEED}
#done

#python examples/train_vae.py


#python examples/stability.py --seed=${SEED}


###### Test Real Robot ###############
#python motion_imitation/examples/test_robot_interface.py
