from bgp.simglucose.simulation.env import T1DSimEnv
from bgp.simglucose.patient.t1dpatient import T1DPatientNew
from bgp.simglucose.sensor.cgm import CGMSensor
from bgp.simglucose.actuator.pump import InsulinPump
from bgp.simglucose.simulation.scenario_gen import RandomBalancedScenario
from bgp.simglucose.controller.base import Action
from bgp.simglucose.analysis.risk import magni_risk_index
from bgp.rl.helpers import Seed
from bgp.rl import pid

import pandas as pd
import numpy as np
import joblib
import copy
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class DeepSACT1DEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_fun, patient_name=None, seeds=None, reset_lim=None, 
                 n_hours=4, termination_penalty=None, update_seed_on_reset=False,
                 deterministic_meal_size=False, deterministic_meal_time=False, 
                 deterministic_meal_occurrence=False, use_pid_load=False, hist_init=False,
                 harrison_benedict=False, restricted_carb=False, meal_duration=1, 
                 universal=False, reward_bias=0, source_dir=None, **kwargs):
      
        self.source_dir = source_dir
        self.patient_para_file = f'{source_dir}/bgp/simglucose/params/vpatient_params.csv'
        self.control_quest = f'{source_dir}/bgp/simglucose/params/Quest2.csv'
        self.pid_para_file = f'{source_dir}/bgp/simglucose/params/pid_params.csv'
        self.pid_env_path = f'{source_dir}/bgp/simglucose/params'
        self.sensor_para_file = f'{source_dir}/bgp/simglucose/params/sensor_params.csv'
        self.insulin_pump_para_file = f'{source_dir}/bgp/simglucose/params/pump_params.csv'
    
        self.e1 = None  # glucose effectiveness gain
        self.e2 = None  # insulin sensitivity gain
        self._baseline_vmx = 0
        self._baseline_kp3 = None  #storing the baselines so we multiple correctly.
    
        self.exercise_schedule = {}  
        self.current_exercise = None  
        self.timestep = 0
        self._original_params = None
        
        self.universe = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                         ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                         ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
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
        e1 = 1.6 #these we will change
        e2 = .778
        return e1, e2

    def _generate_exercise_schedule(self):
        self.exercise_schedule = {}
        
        for hour in range(24):
            timestep = hour * 12 
            prob = 0.20 if hour in [6, 7, 17, 18, 19] else (0.10 if hour in [12, 13] else 0.02)
            
            if np.random.random() < prob:
                start_step = timestep + np.random.randint(0, 12)
                PVO2max = np.clip(np.random.uniform(0.25, 0.75), 0.25, 0.75) 
                duration_min = np.clip(np.random.normal(45, 15), 30, 60)
                
                self.exercise_schedule[start_step] = (PVO2max, duration_min, start_step)
    
    def _get_exercise_effect(self):
        if self.current_exercise is None:
            if self.timestep in self.exercise_schedule:
                self.current_exercise = self.exercise_schedule[self.timestep]
                PVO2max, duration_min, _ = self.current_exercise

        if self.current_exercise is None:
            return 1.0, 1.0
        
        PVO2max, duration_min, start_step = self.current_exercise
        elapsed_min = (self.timestep - start_step) * 5
        te_hours = elapsed_min / 60.0  
        if elapsed_min < duration_min:
            # (insulin-independent glucose uptake)
            ge_mult = 1.0 + self.e1 * PVO2max
            
            # si_mult affects Vmx (insulin sensitivity)
            si_mult = 1.0 + self.e2 * (PVO2max + te_hours)
            
            return si_mult, ge_mult

    def _apply_exercise_to_patient(self, si_mult, ge_mult):
       
        self.env.patient._params.Vmx = self._baseline_vmx * si_mult
        self.env.patient._params.exercise_k1_mult = ge_mult
        self.env.patient._params.kp3 = self._baseline_kp3 * si_mult

    def step(self, action):
     
        si_mult, ge_mult = self._get_exercise_effect()
        self._apply_exercise_to_patient(si_mult, ge_mult)
        
        if type(action) is np.ndarray:
            action = action.item()
        
        action = (action + 0) * ((self.ideal_basal * 43.2) / 1)
        action = max(0, action)
        
        act = Action(basal=0, bolus=action)
        _, reward, _, info = self.env.step(act, reward_fun=self.reward_fun, cho=None)
        
        state = self.get_state(normalize=False)
        done = self.is_done()
        truncated = (self.timestep >= 2880)

        if done and not truncated and self.termination_penalty is not None:
            reward = reward - self.termination_penalty  

        truncated = (self.timestep >= 2880)  
        
        info.update({
            'exercise_active': self.current_exercise is not None,
            'si_multiplier': si_mult,
            'ge_multiplier': ge_mult,
            'e1': self.e1,
            'e2': self.e2,
        })
        
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
        
        return np.stack([bg, insulin]).flatten()

    def reset(self):
        self.timestep = 0
        self.current_exercise = None
        
        self._generate_exercise_schedule()
        
        print(f"Generated {len(self.exercise_schedule)} exercise events for {self.patient_name}")
        print(f"Patient exercise response: e1={self.e1:.3f}, e2={self.e2:.3f}")
        
        if self.update_seed_on_reset:
            self.increment_seed()
        
        if self.universal:
            patient_name = np.random.choice(self.universe)
            self.set_patient_dependent_values(patient_name)
        
        self.env.sensor.seed = self.seeds['sensor']
        self.env.scenario.seed = self.seeds['scenario']
        self.env.reset()
        self.pid.reset()
        
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
        pass
    
    def close(self):
        pass