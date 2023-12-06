
import time
import numpy as np
import mss as mss
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from pynput.mouse import Button, Controller
import cv2 
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium
import gymnasium.utils.performance

from stable_baselines3.common.callbacks import CheckpointCallback
import keyboard
import pyautogui
from PIL import ImageGrab
import dxcam
from FlappyBirdEnv import FlappyGame

env = FlappyGame()

env.reset()

# print(f"steps per second: {gymnasium.utils.performance.benchmark_step(env, target_duration=10)}")

# from stable_baselines3.common import env_checker

# #env_checker.check_env(env)

#____________________ Training and Saving Logs Callback Class _________________________
models_dir = "models/PPO"
log_dir = "logs"
model_path = f"{models_dir}/PPO_650000_steps.zip"


TIMESTEPS = 1800000
SAVE_FREQ = 50_000

callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_dir, name_prefix="PPO")


##___________________ Creating Model and Training ______________________________________________
model = PPO('CnnPolicy', env, learning_rate=0.001, gamma=0.99, verbose=1, device='cuda')

#kick off training
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=callback, progress_bar=True)

