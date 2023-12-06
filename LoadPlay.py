
import time
import mss as mss
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from FlappyBirdEnv import FlappyGame

env = FlappyGame()

env.reset()

models_dir = "models/PPO"
log_dir = "logs"
model_path = f"{models_dir}/PPO_650000_steps.zip"


model = PPO.load(model_path,env=env, tensorboard_log=log_dir, device="cuda", buffer_size=10_000)


# Random decisions
for episode in range(100): 
    obs = env.reset()  
    terminated = False  
    total_reward   = 0
    
    while not terminated: 
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward,  terminated, info =  env.step(action)
        time.sleep(0.01)
        total_reward  += reward 
         
    print('Total Reward for episode {} is {}'.format(episode, total_reward)) 
    time.sleep(1)

env.close()


