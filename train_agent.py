from DNQ_agent import DNQAgent
from replay_buffer import Replay_buffer
from build_state import BuildState

import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import random
import torch
import numpy as np

BUFFER_SIZE = 100000
EPSILON = 0
NUM_EPISODES = 0
BATCH_SIZE=64
REVIEW_FREQUENCY = 1000
TOTTAL_EPISODES = 5000

def plot_training_results(q_history, score_history):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- Plot Average Q (Left Axis) ---
    color = 'tab:blue'
    ax1.set_xlabel('Episodes (x10)')
    ax1.set_ylabel('Average Q Value', color=color)
    ax1.plot(q_history, color=color, linewidth=2, label='Avg Q')
    ax1.tick_params(axis=y, labelcolor=color)

    # --- Plot Scores (Right Axis) ---
    ax2 = ax1.twinx() 
    color = 'tab:orange'
    ax2.set_ylabel('Score per Episode', color=color)
    ax2.plot(score_history, color=color, alpha=0.3, label='Raw Score')
    
    # Add a moving average for the score to see the trend through the noise
    if len(score_history) > 10:
        moving_avg = np.convolve(score_history, np.ones(10)/10, mode='valid')
        ax2.plot(moving_avg, color='red', linewidth=1.5, label='Score Trend (MA10)')

    fig.tight_layout()
    plt.title("2013 DQN Training Progress: Breakout")
    plt.show()



def main():
    #setup
    gym.register_envs(ale_py)
    basic_env = gym.make("BreakoutNoFrameskip-v4")
    # "VideoPinballNoFrameskip-v4"  , render_mode="human"

    env = BuildState(basic_env,k=4)
    agent = DNQAgent(actions=4)
    buffer = Replay_buffer(cap=BUFFER_SIZE)

    #evaluation vars-
    tottal_score = 0
    eval_frames = None
    avgQ_history = []
    avg_score_history = []

    epsilon = 1
    episode = 0

    while True:
        #episodes (games-rounds) loop. starts new game

        new_frame,_ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        ended = 0

        while not ended:
            #note- this env is a wrapper, the frames are already stacks of k frames
            old_frame = new_frame
            action = env.action_space.sample() if random.random() < epsilon else agent.select_action(old_frame)
            new_frame, reward, terminated, truncated, info = env.step(action)
            
            tottal_score += reward
            
            ended = terminated or truncated
            #save to buffer
            buffer.push(old_frame, action, reward, new_frame, ended)
            #builf Q evel when we have enought frames 
            if eval_frames == None and buffer.len() > 400:
                eval_batch = buffer.sample(100)
                states_only, _, _, _, _ = zip(*eval_batch)
                eval_frames = torch.tensor(np.array(states_only), dtype=torch.float32)
                print("set evaluation set of 500 states captured.")

            
            #now select from buffer and do the learning part
            if buffer.len() > BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)# = batch size, for now 16 
                agent.learn_samples(batch)

        if epsilon > 0.1:
                epsilon -= 0.0001

        episode +=1
        if episode % REVIEW_FREQUENCY == 0 and episode > 0:
            avg_reward = tottal_score/REVIEW_FREQUENCY
            tottal_score = 0
            avg_score_history.append(avg_reward)
            # Q avg eval:
            if eval_frames is not None:
                with torch.no_grad():
                    # 1. Get all Q-values for the 500 states (500, actions)
                    all_q_values = agent.DNQ(eval_frames)

                    # 2. Pick the max Q for each state (the best action the agent sees)
                    max_q_values = all_q_values.max(1)[0]

                    # 3. Average them
                    avg_q = max_q_values.mean().item()
                    avgQ_history.append(avg_q)
            else:
                avg_q = 0.0

            print(f" at episode {episode} avarage reard: {avg_reward} Avg Q: {avg_q:.4f}")


        if episode >= TOTTAL_EPISODES:
            break

    env.close()

    plot_training_results(avgQ_history, avg_score_history)

    # observation,_ = env.reset()
    # action = env.action_space.sample()
    # observation, reward, terminated, truncated, _= env.step(action)






if __name__ == '__main__':
    main()









