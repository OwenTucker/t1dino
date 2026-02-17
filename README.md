# EAGLE: Exercise Adapted glucose Learning Environment for Type 1 Diabetes Glucose Control using RL

T1Dino is a reinforcement learning system for automated glucose control in Type 1 Diabetes, incorporating exercise effects into insulin dosing decisions.

This project uses Deep Reinforcement Learning (specifically Soft Actor-Critic/SAC from RL4BG) to learn optimal insulin dosing policies in the Simglucose environment. The system models exercise effects using physiological modeling and supported through T1DEXI data.

##  Structure
```
t1dino/
  t1dino.py                                    # Main training script
  RL4BG/                                       # RL
    bgp/
      simglucose/
        envs/
          exercise_aware_env.py                # Exercise-aware glucose environment
      rl/             
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd t1dino
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- gymnasium
- stable-baselines3
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for complete list

## Usage

Run the training script:
```bash
python t1dino.py
```
Exercise can be toggled on and off through the use_exercise parameter when training a model. 
