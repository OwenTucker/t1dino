import matplotlib.pyplot as plt
import numpy as np
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from t1dino import make_bgp_env

bgp_path = r'C:\Users\owenj\source\t1d_emory\d1no\RL4BG'
if bgp_path not in sys.path:
    sys.path.insert(0, bgp_path)

def diagnose_agent(model_path="bgp_sac_glucose.zip", n_steps=500):
    model = SAC.load(model_path)
    test_env = DummyVecEnv([lambda: make_bgp_env(seed=9999)])
    test_env = VecNormalize.load("bgp_sac_vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    obs = test_env.reset()
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        original_obs = test_env.get_original_obs()
        current_glucose = original_obs[0][47]
        
        if step % 50 == 0:
            print(f"Step {step}: Glucose={current_glucose:.1f} mg/dL, Action={action[0][0]:.4f}, Reward={reward[0]:.2f}")
        
        if reward[0] < -100:
            print(f"  LARGE PENALTY at step {step}!")
            
    test_env.close()

def test_and_visualize(model_path="bgp_sac_glucose.zip", n_episodes=1, max_steps=1000):
    model = SAC.load(model_path)
    
    test_env = DummyVecEnv([lambda: make_bgp_env(seed=9999)])
    test_env = VecNormalize.load("bgp_sac_vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        done = False
        
        glucose_levels = []
        insulin_doses = []
        rewards = []
        timesteps = []
        
        step = 0
        episode_reward = 0
        
        print(f"Starting episode {episode+1}...")
        
        while not done and step < max_steps:
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}...")
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            current_glucose = obs[0][47]
            
            glucose_levels.append(current_glucose)
            insulin_doses.append(action[0][0])
            rewards.append(reward[0])
            timesteps.append(step)
            episode_reward += reward[0]
            step += 1
            
            if done[0]:
                print(f"  Episode terminated naturally at step {step}")
                break
        
        if step >= max_steps:
            print(f"  Episode reached max steps ({max_steps})")
        
        print(f"Episode {episode+1} completed: {step} steps, reward: {episode_reward:.2f}")
    
    test_env.close()

if __name__ == "__main__":
    diagnose_agent()
    test_and_visualize("bgp_sac_glucose.zip", n_episodes=1, max_steps=1000)