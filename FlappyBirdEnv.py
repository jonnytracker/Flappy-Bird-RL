
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
mouse = Controller()

IMAGE_REDUCTION = 200

game_location = 0

class FlappyGame(Env):
    def __init__(self, width, height, top, left ):
        self.start_time = time.time()
        self.last_check_time = 0
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(IMAGE_REDUCTION, IMAGE_REDUCTION, 3), dtype=np.uint8)
        self.action_space = Discrete(2)  
       
        self.game_location = {'top': top +100, 'left': left, 'width': width, 'height': height -200}
        
        #Dxcam implementation
        right = left + width
        bottom = top + height
        region = (left, top, right, bottom)
        self.camera = dxcam.create(device_idx=0, output_idx=0, region=region, output_color="BGR")
        self.camera.start(target_fps=60)  
   
       
    def step(self, action):
        action_map = {            
            0: 'no_op',
            1: 'space'            
        }

        
        if action != 0:                
            mouse.click(Button.left)
            
        terminated = self.get_done()
                
        # Modify the reward calculation
        if not terminated:
            reward = 1  # 1 for every frame the agent stays alive
        else:
            reward = -1  # -1.0 when the agent dies

        new_observation = self.get_observation()
        
        info = {}     
     
        return new_observation, reward, terminated, False, info
        
    
    def render(self):     
        cv2.imshow('Game', self.get_observation())
        # cv2.moveWindow('Game', 3333, 500)
        cv2.waitKey(1)      

    
    def close(self):
        cv2.destroyAllWindows()
        self.camera.stop()  # Release screen capture resources

    
    def reset(self, seed= None):       
        x = 760
        y = 630
        width = x + 465
        height = y + 170
        screen = ImageGrab.grab(bbox=(x, y, width, height)) 
        try:
            if pyautogui.locateOnScreen('Restart.png', region=(x, y, width, height) ,confidence=0.6) is not None:
                image = pyautogui.locateOnScreen('Restart.png',region=(x, y, width, height) , confidence=0.6)
                if image != None:
                    x, y = pyautogui.center(image)                   
                    
                    mouse.position = (x, y)
                    mouse.click(Button.left)
                   
        except:
            pass
        try:
            if pyautogui.locateOnScreen('Bird.png', region=(x, y, width, height) ,confidence=0.6) is not None:
                    image = pyautogui.locateOnScreen('Bird.png',region=(x, y, width, height) , confidence=0.6)
                    if image != None:
                        x, y = pyautogui.center(image)                                                                                                                                                     
                        
                        mouse.position = (x, y)
                        mouse.click(Button.left)
        except:
            pass


        initial_observation = self.get_observation()
        info = {}  # You can populate this info dictionary with additional information if needed.

        return initial_observation, info
    
    
    def get_observation(self):
        frame = self.camera.get_latest_frame()
        screen =  np.array(frame, dtype=np.uint8)
        screen = cv2.resize(screen, (IMAGE_REDUCTION, IMAGE_REDUCTION))
        return screen
    

    def get_done(self):
        done = False       
        
        try:
            #pixel position to check for color of game over screen
            x = 917 
            y = 580

            # Check the time interval before performing the check
            current_time = time.time()
            elapsed_time = current_time - self.last_check_time

            #check every 3 seconds for game over screen
            if elapsed_time >= 3:             
                screen = ImageGrab.grab(bbox=(x, y, x + 1, y + 1))
                mouse.position = (x,y)                                             
                pixel_value = screen.getpixel((0, 0))
               
                if pixel_value == (222, 217, 150):  # Adjust the pixel value as needed
                    done = True                  

                 # Update the last check time
                self.last_check_time = current_time
                
        except:
            pass
        
        return done
    
