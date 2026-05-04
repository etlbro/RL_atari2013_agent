from DNQ_agent import DNQAgent
from replay_buffer import Replay_buffer
from build_state import BuildState

import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import random
import torch
import numpy as np
import time

gym.register_envs(ale_py)

#env = gym.make("VideoPinballNoFrameskip-v4", render_mode="human")
basic_env = gym.make("BreakoutNoFrameskip-v4", render_mode="human") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = BuildState(basic_env, k=4)


agent = DNQAgent(device=device, actions=4)

agent.DNQ.load_state_dict(torch.load('dqn_breakout.pth', map_location=device, weights_only=True))
agent.DNQ.eval()
print("weights loaded")  


frame,_ = env.reset()
frame,_, _ , _, _ = env.step(1)

terminated = False
truncated = False
# here agent playes the game:
while not (truncated or terminated):
    time.sleep(0.016)
    state_tensor = torch.from_numpy(np.array(frame)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        q_values = agent.DNQ(state_tensor)
        
        # A 5% chance to randomly twitch
        if random.random() < 0.05:
            best_action = env.action_space.sample()
        else:
            best_action = q_values.argmax().detach().cpu().item()
    
    frame, reward, terminated, truncated, info = env.step(best_action)

#episode +=1
#print("ep reward:", episode_reward)
#if episode > 1:
#    break
env.close()

#observation,_ = env.reset()
#action = env.action_space.sample()
#observation, reward, terminated, truncated, _= env.step(action)








