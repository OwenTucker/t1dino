import gymnasium as gym
import numpy as np
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces

bgp_path = r'C:\Users\Owen Tucker\work\t1dino\RL4BG'
if bgp_path not in sys.path:
    sys.path.insert(0, bgp_path)

from bgp.simglucose.envs.simglucose_gym_env import DeepSACT1DEnv
from bgp.rl.reward_functions import magni_reward

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
        obs, reward, done, info = self.env.step(action)
        obs = np.array(obs).flatten().astype(np.float32)
        return obs, reward, done, False, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

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

def make_bgp_env(patient_name='adolescent#001', seed=0):
    env = DeepSACT1DEnv(
        reward_fun=magni_reward,
        patient_name=patient_name,
        seeds={'numpy': seed, 'sensor': seed, 'scenario': seed},
        reset_lim={'lower_lim': 10, 'upper_lim': 1000},
        time=False,
        meal=False,
        bw_meals=True,
        load=False,
        use_pid_load=False,
        hist_init=True,
        gt=False,
        n_hours=4,
        norm=False,
        time_std=None,
        use_old_patient_env=False,
        action_cap=None,
        action_bias=0,
        action_scale='basal',
        basal_scaling=43.2,
        meal_announce=None,
        residual_basal=False,
        residual_bolus=False,
        residual_PID=False,
        fake_gt=False,
        fake_real=False,
        suppress_carbs=False,
        limited_gt=False,
        termination_penalty=1e5,
        weekly=False,
        update_seed_on_reset=True,
        deterministic_meal_size=False,
        deterministic_meal_time=False,
        deterministic_meal_occurrence=False,
        harrison_benedict=True,
        restricted_carb=False,
        meal_duration=5,
        rolling_insulin_lim=None,
        universal=False,
        reward_bias=0,
        carb_error_std=0,
        carb_miss_prob=0,
        source_dir=bgp_path
    )
    env = GymToGymnasiumWrapper(env) 
    env = NextStateRewardWrapper(env) 
    return Monitor(env)

def train_sac():
    train_env = DummyVecEnv([lambda: make_bgp_env(seed=42)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=True
    )
    
    eval_env = DummyVecEnv([lambda: make_bgp_env(seed=1000)])
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
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./bgp_sac_model/",
        log_path="./bgp_sac_logs/",
        eval_freq=50000,
        deterministic=True,
        n_eval_episodes=1,
    )
    
    model.learn(
        total_timesteps=2_000_000,
        #callback=eval_callback,
        progress_bar=True,
        log_interval=10
    )
    
    model.save("bgp_sac_glucose")
    train_env.save("bgp_sac_vec_normalize.pkl")
    
    print("\n=== Training Complete ===")
    
    train_env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    model = train_sac()
