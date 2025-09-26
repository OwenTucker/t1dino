import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from t1dino import make_bgp_env

def calculate_clinical_metrics(model_path="bgp_sac_glucose.zip", n_steps=2000, n_episodes=5):
    model = SAC.load(model_path)
    test_env = DummyVecEnv([lambda: make_bgp_env(seed=9999)])
    test_env = VecNormalize.load("bgp_sac_vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    all_glucose = []
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        done = False
        step = 0
        
        while not done and step < n_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            original_obs = test_env.get_original_obs()
            glucose = original_obs[0][47]
            all_glucose.append(glucose)
            step += 1
            
            if done[0]:
                break
    
    glucose_array = np.array(all_glucose)
    
    tir_70_180 = np.mean((glucose_array >= 70) & (glucose_array <= 180)) * 100
    tir_70_140 = np.mean((glucose_array >= 70) & (glucose_array <= 140)) * 100
    
    tbr_54 = np.mean(glucose_array < 54) * 100
    tbr_70 = np.mean(glucose_array < 70) * 100
    
    tar_180 = np.mean(glucose_array > 180) * 100
    tar_250 = np.mean(glucose_array > 250) * 100
    
    mean_glucose = np.mean(glucose_array)
    std_glucose = np.std(glucose_array)
    cv = (std_glucose / mean_glucose) * 100
    

    print(f"Time in Range (70-180 mg/dL):  {tir_70_180:.1f}%")
    print(f"Time in Range (70-140 mg/dL):  {tir_70_140:.1f}%")
    print(f"  Time Below 70 mg/dL:          {tbr_70:.1f}%")
    print(f"  Time Below 54 mg/dL (severe): {tbr_54:.1f}%")
    print(f"  Time Above 180 mg/dL:         {tar_180:.1f}%")
    print(f"  Time Above 250 mg/dL:         {tar_250:.1f}%")
    print(f"  Mean Glucose:                 {mean_glucose:.1f} mg/dL")
    print(f"  Standard Deviation:           {std_glucose:.1f} mg/dL")
    print(f"  Coefficient of Variation:     {cv:.1f}%")
    test_env.close()
    
    return {
        'TIR_70_180': tir_70_180,
        'TIR_70_140': tir_70_140,
        'TBR_70': tbr_70,
        'TBR_54': tbr_54,
        'TAR_180': tar_180,
        'TAR_250': tar_250,
        'mean_glucose': mean_glucose,
        'std_glucose': std_glucose,
        'cv': cv
    }

if __name__ == "__main__":
    calculate_clinical_metrics()