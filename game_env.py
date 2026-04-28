import gymnasium as gym
import time
import ale_py


gym.register_envs(ale_py)

#env = gym.make("VideoPinballNoFrameskip-v4", render_mode="human")
env = gym.make("BoxingNoFrameskip-v4", render_mode="human")

episode = 0
while True:
    
    observation,_ = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    while not (truncated or terminated):
        #time.sleep(0.016)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    
    episode +=1
    print("ep reward:", episode_reward)
    if episode > 1:
        break

env.close()

observation,_ = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, _= env.step(action)








