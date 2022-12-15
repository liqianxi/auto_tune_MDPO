# MDPO-Auto: Enabling Hyperparameter Auto-tuning in Off-policy Mirror Descent Policy Optimization

## Introduction
Mirror Descent Policy Optimization (MDPO), is a deep reinforcement learning (RL) algorithm that
uses the idea of Mirror Descent (MD) to enforce a trust-region constraint in a soft way when optimizing
for the policy. Researchers examine the performance of MDPO and notice it outperforms several popular
deep RL algorithms, such as TRPO, PPO and SAC. In the second paper of SAC, the authors extend the
vanilla SAC algorithm, they design and implement an approach to allow one critical hyperparameter -
entropy coefficient to be tuned automatically during the deep RL training. The entropy coefficient set the
relative importance of the entropy term when optimizing the policy and it is usually selected manually and
empirically by the researchers, which is extremely time-consuming.  

In this work, we will combine the work of the off-policy MDPO algorithm and the entropy coefficient
auto-tuning approach to obtain a new variant of the off-policy MDPO algorithm, which we name it
MDPO-Auto. We derive the update rule for the entropy coefficient and modify the off-policy MDPO
algorithm to be the MDPO-Auto algorithm. Besides, we conduct experiments on an OpenAI Gym
simulated environment and MDPO-Auto outperforms off-policy MDPO.

## Experiment Results
![alt text](https://github.com/liqianxi/auto_tune_MDPO/blob/a3613522e50e3de4886e3e68afc087ac8b04928f/compare.png)
![alt text](https://github.com/liqianxi/auto_tune_MDPO/blob/a3613522e50e3de4886e3e68afc087ac8b04928f/ent_coeff.png)


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
