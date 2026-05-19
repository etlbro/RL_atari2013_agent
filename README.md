# Deep Q-Network (DQN) for Atari Amidar

## Overview
This repository contains a PyTorch implementation of a Deep Q-Network (DQN) trained to play the Atari game **Amidar** (`AmidarNoFrameskip-v4`). The project is built entirely from scratch without the use of high-level Reinforcement Learning libraries, adhering strictly to the preprocessing, evaluation, and architectural standards established in the foundational 2013 DeepMind paper (*Playing Atari with Deep Reinforcement Learning*).

## Project Architecture & Features
* **Custom Environment Wrapper (`BuildState`)**:
  * **Image Preprocessing:** Uses OpenCV to convert frames to grayscale, crop out static UI elements, resize to 84x84, and normalize pixel values (0.0 to 1.0).
  * **Frame Stacking & Skipping:** Implements a frame skip of $k=4$, allowing the agent to perceive motion and velocity without processing redundant intermediate frames.
* **DQN Model**: A custom Convolutional Neural Network (CNN) in PyTorch that maps pixel states directly to Q-values for the 10 discrete actions in Amidar.
* **Replay Memory & Optimization**: 
  * Features a fully implemented Replay Buffer (1,000,000 capacity) for random batch sampling to break correlation in observation sequences.
  * Utilizes the Bellman equation for Q-value updates and an $\epsilon$-greedy policy with linear decay for exploration.
* **Rigorous Evaluation Protocol**: 
  * Pauses training every 10,000 steps to run a strictly controlled evaluation sandbox (weights frozen via `.eval()`, randomness fixed at $\epsilon = 0.05$) to measure true learning progress without survivorship bias.
  * Automatically generates learning curves with moving averages to visualize performance trends.
* **Fail-Safe Checkpointing**: Implements a dual-save system that continuously overwrites a `Latest_Model.pth` to prevent data loss on cloud platforms, while isolating and saving `Best_Model.pth` only when the agent breaks its average score record.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Environment:** Gymnasium (ALE)
* **Data & Visualization:** NumPy, Matplotlib, OpenCV (`cv2`)

---
**Note:** This repository was developed as a university assignment to recreate early deep reinforcement learning methodologies from scratch.
