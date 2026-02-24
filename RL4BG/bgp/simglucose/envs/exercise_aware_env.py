from bgp.simglucose.simulation.env import T1DSimEnv
from bgp.simglucose.patient.t1dpatient import T1DPatientNew
from bgp.simglucose.sensor.cgm import CGMSensor
from bgp.simglucose.actuator.pump import InsulinPump
from bgp.simglucose.simulation.scenario_gen import RandomBalancedScenario
from bgp.simglucose.controller.base import Action
from bgp.simglucose.analysis.risk import magni_risk_index
from bgp.rl.helpers import Seed
from bgp.rl import pid
import random
import pandas as pd
import numpy as np
import joblib
import copy
import gym
import math
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

class DeepSACT1DEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_fun, patient_name=None, seeds=None, reset_lim=None, 
                 n_hours=4, termination_penalty=None, update_seed_on_reset=False,
                 deterministic_meal_size=False, deterministic_meal_time=False, 
                 deterministic_meal_occurrence=False, use_exercise_env=False, use_pid_load=False, hist_init=False,
                 harrison_benedict=False, restricted_carb=False, meal_duration=1, 
                 universal=False, reward_bias=0, source_dir=None, **kwargs):
      
        self.source_dir = source_dir
        self.patient_para_file = f'{source_dir}/bgp/simglucose/params/vpatient_params.csv'
        self.control_quest = f'{source_dir}/bgp/simglucose/params/Quest2.csv'
        self.pid_para_file = f'{source_dir}/bgp/simglucose/params/pid_params.csv'
        self.pid_env_path = f'{source_dir}/bgp/simglucose/params'
        self.sensor_para_file = f'{source_dir}/bgp/simglucose/params/sensor_params.csv'
        self.insulin_pump_para_file = f'{source_dir}/bgp/simglucose/params/pump_params.csv'
          
        with open('workout_library.pkl', 'rb') as f:
                   workout_library = pickle.load(f)
                   
        self.workout_lib = workout_library
        self.e1 = None  # glucose effectiveness gain
        self.e2 = None  # insulin sensitivity gain
        self._baseline_vmx = 0
        self._baseline_kp3 = None  #deapreciated 
        self.use_exercise_env = use_exercise_env
    
        self.exercise_schedule = {}  
        self.current_exercise = None  
        self.timestep = 0
        self._original_params = None
        self.exercise_dose = 0
        self.iexer = 0
        self.pexer = 0
        self.tau_post = 0
       
        self.exercise_hist = [] 
        self.si_mult_hist = []   
        self.ge_mult_hist = []   
        
        #self.universe = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
              #           ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
              #           ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
        
        self.universe = (['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
        
        
        self.universal = universal

        if seeds is None:
            seed_list = self._seed()
            seeds = Seed(numpy_seed=seed_list[0], sensor_seed=seed_list[1], scenario_seed=seed_list[2])
        
        if patient_name is None:
            patient_name = np.random.choice(self.universe) if self.universal else 'adolescent#001'
        
        np.random.seed(seeds['numpy'])
        self.seeds = seeds

        self.sample_time = 5  # minutes per timestep
        self.day = int(1440 / self.sample_time)
        self.state_hist = int((n_hours * 60) / self.sample_time)
        self.reward_fun = reward_fun
        self.reward_bias = reward_bias
        self.termination_penalty = termination_penalty
        self.deterministic_meal_size = deterministic_meal_size
        self.deterministic_meal_time = deterministic_meal_time
        self.deterministic_meal_occurrence = deterministic_meal_occurrence
        self.use_pid_load = use_pid_load
        self.hist_init = hist_init
        self.harrison_benedict = harrison_benedict
        self.restricted_carb = restricted_carb
        self.meal_duration = meal_duration
        self.update_seed_on_reset = update_seed_on_reset
        self.start_time = datetime(2018, 1, 1, 0, 0, 0)

        if reset_lim is None:
            self.reset_lim = {'lower_lim': 10, 'upper_lim': 1000}
        else:
            self.reset_lim = reset_lim
        
        self.env = None
        self.set_patient_dependent_values(patient_name)
        self.env.scenario.day = 0

    def _estimate_exercise_response(self, patient_params): 
        e1 = 1.6   
        e2 = 0.778 
        return e1, e2

    def _generate_exercise_schedule_realistic(self):
        with open('workout_library_full.pkl', 'rb') as f:
            lib = pickle.load(f)
        
        for day in range(10):
            if np.random.random() < 0.4:
                workout = np.random.choice(lib['workouts'])
                timestep = (day * 288) + (workout['start_hour'] * 12)
                
                self.exercise_schedule[timestep] = {
                    'hrr_curve': workout['hrr_curve'],
                    'duration_min': workout['duration_min'],
                    'start_step': timestep
                }
                
    def _apply_exercise_to_patient(self, si_mult, ge_mult):
        self.env.patient._params.Vmx = self._baseline_vmx * si_mult 
        self.env.patient._params.exercise_k1_mult = ge_mult 
    def _get_exercise_effect(self):
        beta_p = 0.30  
        if self.current_exercise is None:
            if self.timestep in self.exercise_schedule:
                self.current_exercise = self.exercise_schedule[self.timestep]
                self.exercise_dose = 0  
        if self.current_exercise is None:
            ge_mult = 1.0  
            if self.pexer > 0:
                delta_t = 5 / 60  
                self.pexer = self.pexer * math.exp(-delta_t / self.tau_post)
            
            si_mult = 1.0 + self.pexer
            return si_mult, ge_mult
  
        start_step = self.current_exercise['start_step']
        elapsed_steps = self.timestep - start_step
        elapsed_min = elapsed_steps * 5
        duration_min = self.current_exercise['duration_min']
        
       
        if elapsed_min >= duration_min:
            self.tau_post = (24 + min(self.exercise_dose, 2.0) * 24) * 12  
            self.pexer = beta_p * self.e2 * self.exercise_dose 
            self.current_exercise = None
            self.exercise_dose = 0
            si_mult = 1.0 + self.pexer
            ge_mult = 1.0
            return si_mult, ge_mult
        
      
        hrr_curve = self.current_exercise['hrr_curve']
        current_hrr = hrr_curve[elapsed_steps]
       
        ge_mult = 1.0 + self.e1 * current_hrr
        si_mult = 1.0  
        delta_t = 5 / 60 
        self.exercise_dose += current_hrr * delta_t 
        
        return si_mult, ge_mult
    def step(self, action):
        #if self.timestep == 0:
         #   print(f"[STEP] use_exercise_env={self.use_exercise_env}")
        
        if type(action) is np.ndarray:
            action = action.item()
        
        action = (action + 0) * ((self.ideal_basal * 43.2) / 1)
        action = max(0, action)
        
        if self.use_exercise_env: 
            si_mult, ge_mult = self._get_exercise_effect()
            self._apply_exercise_to_patient(si_mult, ge_mult)
        else:
            si_mult = 1.0
            ge_mult = 1.0
        
        act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun, cho=None)
        
        #  exercise history (will be all zeros if disabled)
        if self.use_exercise_env:
            current_hrr = 0
            if self.current_exercise is not None:
                elapsed_steps = self.timestep - self.current_exercise['start_step']
                if elapsed_steps < len(self.current_exercise['hrr_curve']):
                    current_hrr = self.current_exercise['hrr_curve'][elapsed_steps]
            self.exercise_hist.append(current_hrr)
        else:
            self.exercise_hist.append(0) 
        self.si_mult_hist.append(si_mult)
        self.ge_mult_hist.append(ge_mult)
        
        info.update({
            'exercise_active': self.current_exercise is not None if self.use_exercise_env else False,
            'si_multiplier': si_mult,
            'ge_multiplier': ge_mult,
            'e1': self.e1 if self.use_exercise_env else 0,
            'e2': self.e2 if self.use_exercise_env else 0,
        })
        
        state = self.get_state(normalize=False)
        done = self.is_done()
        truncated = (self.timestep >= 2880)
        
        if done and not truncated and self.termination_penalty is not None:
            reward = reward - self.termination_penalty
        
        self.timestep += 1
        return state, reward, done, truncated, info

    def get_state(self, normalize=False):
        bg = self.env.CGM_hist[-self.state_hist:]
        insulin = self.env.insulin_hist[-self.state_hist:]
        
        if normalize:
            bg = np.array(bg) / 400.
            insulin = np.array(insulin) * 10

        if len(bg) < self.state_hist:
            bg = np.concatenate((np.full(self.state_hist - len(bg), -1), bg))
        if len(insulin) < self.state_hist:
            insulin = np.concatenate((np.full(self.state_hist - len(insulin), -1), insulin))
        
        exercise_feature = np.zeros(self.state_hist)
        
        if self.use_exercise_env and len(self.exercise_hist) > 0:
            hist_len = min(len(self.exercise_hist), self.state_hist)
            for i in range(hist_len):
                idx = self.state_hist - hist_len + i
                exercise_feature[idx] = 1.0 if self.exercise_hist[-(hist_len - i)] > 0 else 0.0
        
        #return 3 channels regardless of use_exercise_env
        return np.stack([bg, insulin, exercise_feature]).flatten()

    def reset(self):
        self.timestep = 0
        self.current_exercise = None
        self.exercise_dose = 0
        self.pexer = 0  
        self.tau_post = 24 * 12  
        self.exercise_hist = []      
        self.si_mult_hist = []
        self.ge_mult_hist = []
        self.exercise_schedule = {}  
        
        #print(f"[RESET] use_exercise_env={self.use_exercise_env}")
        
        if self.update_seed_on_reset:
            self.increment_seed()
        
        if self.universal:
            patient_name = np.random.choice(self.universe)
            self.set_patient_dependent_values(patient_name)
        
        self.env.sensor.seed = self.seeds['sensor']
        self.env.scenario.seed = self.seeds['scenario']
        self.env.reset()
        self.pid.reset()

        
        if self.use_exercise_env:
            self._generate_exercise_schedule_realistic()
            #print(f"  Generated {len(self.exercise_schedule)} workouts")
        if self.use_pid_load:
            self.pid_load(1)
        if self.hist_init:
            self._hist_init()
        
        return self.get_state(normalize=False)

    def pid_load(self, n_days):
        for i in range(n_days * self.day):
            b_val = self.pid.step(self.env.CGM_hist[-1])
            act = Action(basal=0, bolus=b_val)
            _ = self.env.step(action=act, reward_fun=self.reward_fun, cho=None)

    def calculate_iob(self):
        ins = self.env.insulin_hist
        # In your env, what's the length of the IOB curve?
        print(f"IOB curve length: {len(self.iob)} timesteps = {len(self.iob) * 5 / 60} hours")
        return np.dot(np.flip(self.iob, axis=0)[-len(ins):], ins[-len(self.iob):])

    def avg_risk(self):
        return np.mean(self.env.risk_hist[max(self.state_hist, 2880):])

    def avg_magni_risk(self):
        return np.mean(self.env.magni_risk_hist[max(self.state_hist, 2880):])

    def glycemic_report(self):
        bg = np.array(self.env.BG_hist[max(self.state_hist, 2880):])
        ins = np.array(self.env.insulin_hist[max(self.state_hist, 2880):])
        hypo = (bg < 70).sum() / len(bg)
        hyper = (bg > 180).sum() / len(bg)
        euglycemic = 1 - (hypo + hyper)
        return bg, euglycemic, hypo, hyper, ins

    def is_done(self):
        return (self.env.BG_hist[-1] < self.reset_lim['lower_lim'] or 
                self.env.BG_hist[-1] > self.reset_lim['upper_lim'])

    def increment_seed(self, incr=1):
        self.seeds['numpy'] += incr
        self.seeds['scenario'] += incr
        self.seeds['sensor'] += incr

    def set_patient_dependent_values(self, patient_name):
        self.patient_name = patient_name
        vpatient_params = pd.read_csv(self.patient_para_file)
        quest = pd.read_csv(self.control_quest)
        self.kind = self.patient_name.split('#')[0]
        self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))['BW'].item()
        self.u2ss = vpatient_params.query('Name=="{}"'.format(self.patient_name))['u2ss'].item()
        self.ideal_basal = self.bw * self.u2ss / 6000.
        self.CR = quest.query('Name=="{}"'.format(patient_name)).CR.item()
        self.CF = quest.query('Name=="{}"'.format(patient_name)).CF.item()
        
        iob_all = joblib.load(f'{self.pid_env_path}/iob.pkl')
        self.iob = iob_all[self.patient_name]
        
        pid_df = pd.read_csv(self.pid_para_file)
        if patient_name not in pid_df.name.values:
            raise ValueError(f'{patient_name} not in PID csv')
        pid_params = pid_df.loc[pid_df.name == patient_name].squeeze()
        self.pid = pid.PID(setpoint=pid_params.setpoint,
                           kp=pid_params.kp, ki=pid_params.ki, kd=pid_params.kd)
        
        # estimate patient-specific exercise response coefficients

        patient_params = vpatient_params.query('Name=="{}"'.format(patient_name)).squeeze()
        
        self.e1, self.e2 = self._estimate_exercise_response(patient_params)
        
        patient = T1DPatientNew.withName(patient_name, self.patient_para_file)
        sensor = CGMSensor.withName('Dexcom', self.sensor_para_file, seed=self.seeds['sensor'])
        scenario = RandomBalancedScenario(
            bw=self.bw, start_time=self.start_time, seed=self.seeds['scenario'],
            kind=self.kind, restricted=self.restricted_carb,
            harrison_benedict=self.harrison_benedict, unrealistic=False,
            deterministic_meal_size=self.deterministic_meal_size,
            deterministic_meal_time=self.deterministic_meal_time,
            deterministic_meal_occurrence=self.deterministic_meal_occurrence,
            meal_duration=self.meal_duration
        )
        pump = InsulinPump.withName('Insulet', self.insulin_pump_para_file)
        
        self.env = T1DSimEnv(patient=patient, sensor=sensor, pump=pump,
                             scenario=scenario, sample_time=self.sample_time, 
                             source_dir=self.source_dir)
        self._baseline_vmx = self.env.patient._params.Vmx  
        if self.hist_init:
            self.env_init_dict = joblib.load(f"{self.pid_env_path}/{self.patient_name}_data.pkl")
            self.env_init_dict['magni_risk_hist'] = []
            for bg in self.env_init_dict['bg_hist']:
                self.env_init_dict['magni_risk_hist'].append(magni_risk_index([bg]))
            self._hist_init()

    def _hist_init(self):
        env_init_dict = copy.deepcopy(self.env_init_dict)
        self.env.patient._state = env_init_dict['state']
        self.env.patient._t = env_init_dict['time']
        self.env.time_hist = env_init_dict['time_hist']
        self.env.BG_hist = env_init_dict['bg_hist']
        self.env.CGM_hist = env_init_dict['cgm_hist']
        self.env.risk_hist = env_init_dict['risk_hist']
        self.env.LBGI_hist = env_init_dict['lbgi_hist']
        self.env.HBGI_hist = env_init_dict['hbgi_hist']
        self.env.CHO_hist = env_init_dict['cho_hist']
        self.env.insulin_hist = env_init_dict['insulin_hist']
        self.env.magni_risk_hist = env_init_dict['magni_risk_hist']

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        return [seed1, seed2, seed3]

    @property
    def action_space(self):
        return spaces.Box(low=0, high=0.1, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
        num_channels = int(len(st) / self.state_hist)
        return spaces.Box(low=0, high=np.inf, shape=(num_channels, self.state_hist))
    
    def render(self, mode='human'):
        if mode != 'human':
            return
        
        import matplotlib.pyplot as plt
        
      
        if len(self.env.BG_hist) < 2:
            return  
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        time_steps = np.arange(len(self.env.BG_hist)) * self.sample_time / 60  # convert to hours
       
        axes[0].plot(time_steps, self.env.BG_hist, 'b-', linewidth=2, label='Blood Glucose')
        axes[0].plot(time_steps, self.env.CGM_hist, 'c--', linewidth=1, alpha=0.7, label='CGM Reading')
        axes[0].axhline(y=70, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Hypoglycemia')
        axes[0].axhline(y=180, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Hyperglycemia')
        axes[0].fill_between(time_steps, 70, 180, alpha=0.1, color='green')
        axes[0].set_ylabel('BG (mg/dL)', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Patient: {self.patient_name} | Exercise-Aware Glucose Control', 
                          fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, max(400, max(self.env.BG_hist) * 1.1)])
        
      
        axes[1].bar(time_steps, self.env.CHO_hist, width=0.08, color='orange', alpha=0.7, label='Carbs')
        axes[1].set_ylabel('Carbs (g)', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
     
        axes[2].plot(time_steps, self.env.insulin_hist, 'purple', linewidth=2, label='Insulin')
        axes[2].set_ylabel('Insulin (U/hr)', fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        time_steps_ex = np.arange(len(self.exercise_hist)) * self.sample_time / 60
    
        axes[3].fill_between(time_steps_ex, 0, self.exercise_hist, 
                             color='green', alpha=0.3, label='Exercise Intensity')
        axes[3].plot(time_steps_ex, self.exercise_hist, 'g-', linewidth=2)
        ax3_twin = axes[3].twinx()
        ax3_twin.plot(time_steps_ex, self.si_mult_hist, 'r--', linewidth=1.5, 
                      alpha=0.6, label='SI Multiplier')
        ax3_twin.set_ylabel('SI Multiplier', fontsize=10, color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        axes[3].set_ylabel('Exercise\n(% HRR)', fontsize=11, fontweight='bold')
        axes[3].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        lines1, labels1 = axes[3].get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([0, 1])
        
        plt.tight_layout()
      
        return fig
    
    def close(self):
        pass