# MDPO-Auto: Enabling Hyperparameter Auto-tuning in Off-policy Mirror Descent Policy Optimization

## Introduction
MDPO-Auto extends Mirror Descent Policy Optimization (MDPO) by introducing automatic tuning of the entropy coefficient (λ). MDPO is a deep reinforcement learning algorithm that uses Mirror Descent to enforce trust-region constraints when optimizing policies. While MDPO has shown superior performance compared to algorithms like TRPO, PPO, and SAC, it requires manual tuning of the entropy coefficient, which is time-consuming and potentially suboptimal.

This project combines MDPO with an entropy coefficient auto-tuning approach inspired by SAC (Soft Actor-Critic), creating a more efficient and automated variant of the algorithm.

## Key Features
- Automatic tuning of the entropy coefficient (λ)
- Integration with off-policy MDPO
- Built on stable-baselines framework
- Support for MuJoCo environments

## Performance Note
Current implementation achieves performance approximately equal to or slightly better than MDPO with manually tuned λ=0.2, with our auto-tuned λ converging to around 0.23. Both MDPO-Auto and MDPO outperform the baseline SAC method.

## Prerequisites
- Python 3.7
- MuJoCo 2.0
- stable-baselines==2.7.0
- tensorflow
- mujoco_py

## Installation

1. Install stable-baselines:
```bash
pip install stable-baselines[mpi]==2.7.0
```

2. Install MuJoCo:
   - [Download MuJoCo 2.0](https://www.roboti.us/index.html)
   - Copy MuJoCo files to `.mujoco/` directory
   - Set up license file

3. Set up MDPO:
   - Clone the MDPO repository
   - Copy `mdpo-on` and `mdpo-off` directories to your stable-baselines installation

4. Install dependencies:
```bash
source <virtual env path>/bin/activate
pip install -r requirements.txt
```

## Usage Examples

### Run MDPO-Auto
```bash
python3.7 mdpo_off/run_mujoco_auto.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1 --klcoeff=0.4 --tsallis_coeff=2.0
```

### Run Original Off-policy MDPO
```bash
python3 run_mujoco.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1000 --klcoeff=1.0 --lam=0.2 --tsallis_coeff=1.0
```

### Run SAC
```bash
python3 sac/run_mujoco.py --env=Walker2d-v2 --num_timesteps=1e6 --sgd_steps=1
```

## Hyperparameters
```python
{
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 1000000,
    'target_smoothing_coefficient': 0.005,
    'hidden_layers': 2,
    'hidden_units_per_layer': 256,
    'discount_factor': 0.99,
    'bregman_step_size': 0.4
}
```

## Algorithm Details

The key innovation in MDPO-Auto is the automated entropy coefficient adjustment. The algorithm:
1. Formulates the problem as a constrained optimization problem
2. Converts it to a dual optimization problem
3. Uses SGD to approximate the optimal λ value during training

The entropy coefficient (λ) converges to approximately 0.23 during training, slightly higher than the manually tuned value of 0.2 used in the original MDPO paper.

## Citations
If you use this code in your research, please cite our work:
```
[Citation information to be added]
```

## Acknowledgments
This project builds upon the work of:
- Mirror Descent Policy Optimization (MDPO) by Tomar et al.
- Soft Actor-Critic (SAC) by Haarnoja et al.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
