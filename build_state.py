import cv2
import gymnasium as gym
import numpy as np

from collections import deque

#wrapper for env. will crop to 86x86 like in the paper and stack 4 frames, returning a (4, 86, 86)
class BuildState(gym.Wrapper):

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([],maxlen=4)
    
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, 84, 84),
            dtype=np.float32
        )
    
    # convert to greyscale and crop img to a 84x84
    def process_image(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        resized_img = cv2.resize(gray_img, (84, 110), interpolation=cv2.INTER_AREA)
        
        # remove from the top, the score board
        cropped_img  = resized_img[18:102, 0:84]
        # normalize for nn
        return  cropped_img.astype(np.float32) / 255.0


    #start a new game. get inital img, crop and copy it 4 times to fit nn tmplt 
    def reset(self,**kwargs):
        observation, info = self.env.reset(**kwargs)
        p_obs = self.process_image(observation)
        #stack k of the img in the Q
        for _ in range(self.k):
            self.frames.append(p_obs)
        #return the Q as a np array
        return np.stack(self.frames, axis=0), info


    #overwrites the step. will return 4 stack of greysacle 84x84 images    
    def step(self, action):
        tottal_reward = 0.0

        for _ in range(self.k):
            observation, reward, terminated, truncated, info = self.env.step(action)
            tottal_reward += reward
            #if game ends this is the last frame. dont want to mix new game with old
            if terminated or truncated:
                break

        p_obs = self.process_image(observation)

        #add to top of Q
        self.frames.append(p_obs)
        
        return np.stack(self.frames, axis=0), tottal_reward, terminated, truncated, info

