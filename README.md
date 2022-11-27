# MDPO-Auto: Enabling Hyperparameter Auto-tuning in Off-policy Mirror Descent Policy Optimization

## Prerequisites
All dependencies are provided in a python virtual-env requirements.txt file. Majorly, you would need to install stable-baselines, tensorflow, and mujoco_py.
The python version is 3.7

## Installation

1. Install stable-baselines
~~~
pip install stable-baselines[mpi]==2.7.0
~~~

2. [Download](https://www.roboti.us/index.html) and copy MuJoCo library and license files into a `.mujoco/` directory. We use `mujoco200` for this project.

3. Clone MDPO and copy the `mdpo-on` and `mdpo-off` directories inside [this directory](https://github.com/hill-a/stable-baselines/tree/master/stable_baselines).


4. Activate `virtual-env` using the `requirements.txt` file provided.
~~~
source <virtual env path>/bin/activate
~~~

# Example

Off-policy MDPO
~~~
python3 run_mujoco.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1000 --klcoeff=1.0 --lam=0.2 --tsallis_coeff=1.0
~~~

SAC
~~~
python3 sac/run_mujoco.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1;
~~~

MDPO-Auto
~~~
python3.7 mdpo_off/run_mujoco_auto.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1 --klcoeff=0.4 --tsallis_coeff=2.0; 

~~~
