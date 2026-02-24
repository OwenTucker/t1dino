# EAGLE: Exercise-Adapted Glucose Learning Environment

T1Dino is a reinforcement learning system for automated insulin dosing in Type 1 Diabetes that incorporates exercise effects. It uses Soft Actor-Critic (SAC) from [RL4BG](https://github.com/MLD3/RL4BG) within the SimGlucose environment, with exercise physiology modeled from T1DEXI data.

## Structure

The main training script, t1dino.py, trains a model for a given amount of timesteps. You can play around with the parameters in this file. You can also adjust the exercise-related parameters in exercise_aware_env.py. Once you create a model, you can use metrics.py to evaluate the model. 

t1dino/
- t1dino.py                          # Main training script
- metrics.py                         # Evaluation script
- RL4BG/
     - bgp/
        - simglucose/
           - envs/
              - exercise_aware_env.py   # Exercise-aware glucose environment

## Installation

EAGLE runs on RL4BG, which uses many old packages so you will likely need a virtual environment.

1. Clone the repository
```bash
   git clone <your-repo-url>
   cd t1dino
```

2. Create and activate a virtual environment

   **Windows (PowerShell):**
```bash
   py -m venv venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\venv\Scripts\activate
```
   **Mac/Linux:**
```bash
   python -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

## Usage
```bash
python t1dino.py
```

To toggle exercise modeling, set the `use_exercise` parameter in `train_sac()` at the bottom of `t1dino.py`.
