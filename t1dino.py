import gymnasium as gym
import numpy as np
import sys
import os
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
rl4bg_path = Path(__file__).resolve().parent / 'RL4BG'
if str(rl4bg_path) not in sys.path:
    sys.path.insert(0, str(rl4bg_path))
from bgp.simglucose.envs.exercise_aware_env import DeepSACT1DEnv
from bgp.rl.reward_functions import magni_reward
import pickle

class GymToGymnasiumWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        old_obs_space = env.observation_space
      
        if len(old_obs_space.shape) > 1:
            flat_size = np.prod(old_obs_space.shape)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(flat_size,),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=old_obs_space.low,
                high=old_obs_space.high,
                shape=old_obs_space.shape,
                dtype=np.float32
            )
        
        old_act_space = env.action_space
        self.action_space = spaces.Box(
            low=old_act_space.low,
            high=old_act_space.high,
            shape=old_act_space.shape,
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs = np.array(obs).flatten().astype(np.float32)
        return obs, {}
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = np.array(obs).flatten().astype(np.float32)
        return obs, reward, done, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
class RenderingEvalCallback(EvalCallback):
    def __init__(self, *args, render_at_end=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_at_end = render_at_end
    
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
    
        if self.render_at_end and self.n_calls % self.eval_freq == 0:
            try:
                # unwrap to get to the base DeepSACT1DEnv
                base_env = self.eval_env.envs[0]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                # render the completed episode
                #base_env.render()
                #print(f"\nRendered evaluation at step {self.num_timesteps}")
                
            except Exception as e:
                print(f"Could not render: {e}")
        
        return continue_training

class NextStateRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_bg = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_bg = obs[47] 
        return obs, info
    
    def step(self, action): 
        '''
        This properly defines our risk func, for our (s, a) pair, we take a step and the resulting risk
        is calculated here, which is -risk(b(t+1))
        '''
        obs, reward, done, truncated, info = self.env.step(action)
        next_bg = obs[47]
        bg = max(1, next_bg)
        fBG = 3.5506 * (np.log(bg)**0.8353 - 3.7932)
        risk = 10 * (fBG)**2
        reward = -1 * risk
        
        return obs, reward, done, truncated, info

def make_bgp_env(patient_name='adult#001', seed=0, use_exercise=True): 
    env = DeepSACT1DEnv(
        reward_fun=magni_reward,
        patient_name=patient_name,
        seeds={'numpy': seed, 'sensor': seed, 'scenario': seed},
        reset_lim={'lower_lim': 10, 'upper_lim': 1000},
        n_hours=4,
        termination_penalty=1e7,
        update_seed_on_reset=True,
        deterministic_meal_size=False,
        deterministic_meal_time=False,
        deterministic_meal_occurrence=False,
        use_exercise_env=use_exercise, 
        use_pid_load=False,
        hist_init=True,
        harrison_benedict=True,
        restricted_carb=False,
        meal_duration=5,  
        universal=False,
        reward_bias=0,
        source_dir=rl4bg_path
    )
    env = GymToGymnasiumWrapper(env) 
    env = NextStateRewardWrapper(env) 
    return Monitor(env)

def train_sac(use_exercise=True, model_name="bgp_sac_glucose"):  # ADD PARAMS
    train_env = DummyVecEnv([
        lambda i=i: make_bgp_env(
            patient_name=f'adult#{i:03d}', 
            seed=42+i,
            use_exercise=use_exercise
        ) 
        for i in range(1, 11)
    ])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True
    )
    
    eval_env = DummyVecEnv([lambda: make_bgp_env(
        seed=1000, 
        use_exercise=use_exercise
    )])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False
    )
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy='auto',
        policy_kwargs=dict(
            net_arch=[256, 256],  
        ),
        verbose=1,
        seed=42,
        device='auto'
    )
    
    eval_callback = RenderingEvalCallback(  
        eval_env,
        best_model_save_path="./bgp_sac_model/",
        log_path="./bgp_sac_logs/",
        eval_freq=50000,
        deterministic=True,
        n_eval_episodes=3,
        render_at_end=True
    )
    model.learn(
        total_timesteps=2_500_000,
        callback=eval_callback,
        progress_bar=True,
        log_interval=10
    )
    
    model.save(model_name)
    train_env.save(f"{model_name}_vec_normalize.pkl")
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    model_baseline = train_sac(
        use_exercise=True, 
        model_name="bgp_sac_glucose_exer_la"
    )
    model_baseline = train_sac(
        use_exercise=False, 
        model_name="bgp_sac_glucose_no_exer_la"
    )