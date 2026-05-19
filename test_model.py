from py_version.DNQ_agent import DNQAgent
from py_version.replay_buffer import Replay_buffer
from py_version.build_state import BuildState

import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import random
import torch
import numpy as np
import time

gym.register_envs(ale_py)

basic_env = gym.make("AmidarNoFrameskip-v4",  render_mode="human") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX 2: Ensure 'k' matches exactly what you used during training!
env = BuildState(basic_env, k=4) 

# FIX 3: Turn off the training wrapper's fake termination
original_training_mode = getattr(env, 'training', True)
env.training = False

num_actions = env.action_space.n
agent = DNQAgent(device=device, actions=num_actions)

agent.DNQ.load_state_dict(torch.load('Amidar_v3.pth', map_location=device, weights_only=True))
agent.DNQ.eval()
print("Weights loaded successfully!")  

frame, _ = env.reset()
frame, _, _, _, _ = env.step(1)

terminated = False
truncated = False

# FIX 1: Set up the score tracker
total_score = 0 

print("Starting game...")

# Here agent plays the game:
while not (truncated or terminated):
    time.sleep(0.01) # ~60 FPS
    
    state_tensor = torch.from_numpy(np.array(frame)).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        q_values = agent.DNQ(state_tensor)
        
        # A 5% chance to randomly twitch
        if random.random() < 0.05:
            best_action = env.action_space.sample()
        else:
            best_action = q_values.argmax().detach().cpu().item()
    
    frame, reward, terminated, truncated, info = env.step(best_action)
    
    # Add to the total score
    total_score += reward 

print(f"\n--- GAME OVER ---")
print(f"Final Score: {total_score}")

# Clean up
env.training = original_training_mode
env.close()