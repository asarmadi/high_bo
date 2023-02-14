# Tuning MPC using BO

## Dependencies

To install the dependencies 

```bash
pip install -r requirements.txt
```

`python3 setup.py install --user`


## MPC code Description
The first step is to generate a gait. The class `OpenLoopGaitGenerator` that generates each leg states:

`
stance_duration:   The desired stance duration
duty_factor:       The ratio stance_duration / total_gait_cycle
initial_leg_phase: The desired initial phase [0, 1] of the legs within the full swing + stance cycle.
`


## Credits

We thank authors of the following repos for their contributions to our codebase:

* The original convex MPC implementation is derived from [motion_imitation](https://github.com/google-research/motion_imitation) with modifications.

* The underlying simulator is [PyBullet](https://pybullet.org/wordpress/).

* The TuRBO implementation is derived from [TuRBO](https://github.com/uber-research/TuRBO) with modifications.
