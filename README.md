# Tuning MPC using BO
This repository contains the code for [High-Dimensional Controller Tuning through Latent Representations](https://arxiv.org/pdf/2309.12487).

## Dependencies

To install the dependencies 

```bash
pip install -r requirements.txt
```

`python3 setup.py install --user`


## Credits

We thank authors of the following repos for their contributions to our codebase:
* The Bicon implementation is derived from [Bicon](https://github.com/machines-in-motion/biconvex_mpc.git) with modifications.

* The original convex MPC implementation is derived from [motion_imitation](https://github.com/google-research/motion_imitation) with modifications.

* The underlying simulator is [PyBullet](https://pybullet.org/wordpress/).

* The TuRBO implementation is derived from [TuRBO](https://github.com/uber-research/TuRBO) with modifications.
