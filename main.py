
import time
import mss as mss
from pynput.mouse import Button, Controller

import keyboard
import pyautogui


mouse = Controller()
from FlappyBirdEnv import FlappyGame

IMAGE_REDUCTION = 100

width = 849
height = 719
top = 110
left = 518


game_location = 0

time.sleep(1)


################ CHECK GAME SCREEN TO START GAME #################
# Find game window
result = None
try:
    if pyautogui.locateOnScreen('GameScreen.png', confidence=0.6) is not None:
        print("Game button found")
        result = pyautogui.locateOnScreen('GameScreen.png', confidence=0.6)
except:
    print("Game Screen Not Found")


if result is not None:
    print("Game screen found, program starting...")
    left, top, width, height = result
    print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")
else:
    print("Game not found, program not started")


#############################################################################
env = FlappyGame()

env.reset()


while not keyboard.is_pressed('q'):
    # Random play
    for episode in range(100): 
        obs = env.reset()  
        terminated = False
        total_reward   = 0
        while not terminated:  
            obs, reward,  terminated, truncated , info =  env.step(env.action_space.sample())
            total_reward  += reward
            env.render()
            time.sleep(0.001)
                    
        print('Total Reward for episode {} is {}'.format(episode, total_reward)) 


