# Deep Q-Network (DQN) for Atari Video Pinball

## Overview
This repository contains a PyTorch implementation of a Deep Q-Network (DQN) trained to play the Atari game **Video Pinball**. The project is built entirely from scratch without the use of high-level Reinforcement Learning libraries, adhering closely to the preprocessing and architectural standards established in the foundational 2013 DeepMind paper (*Playing Atari with Deep Reinforcement Learning*).

## Project Architecture (Current State)
* **Custom Environment Wrapper (`BuildState`)**: 
  * Interfaces with the raw `NoFrameskip-v4` Arcade Learning Environment (ALE).
  * **Frame Skipping:** Manually implements an action-repeat loop (k=4) to accumulate rewards and optimize computation.
  * **Frame Stacking:** Maintains a rolling temporal window (a `deque` of 4 frames) allowing the AI to perceive ball velocity and trajectory.
  * **Image Preprocessing:** Uses OpenCV to convert frames to grayscale, crop out the static scoreboard, resize to 84x84, and normalize pixel values.
* **DQN Model**: A custom Convolutional Neural Network (CNN) in PyTorch that maps pixel states to Q-values.
* **Agent & Memory module**: *(In Progress)* Setting up the Replay Buffer for batch sampling, $\epsilon$-greedy action selection, and the PyTorch optimization loop using the Bellman equation.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Environment:** Gymnasium (ALE)
* **Image Processing:** OpenCV (`cv2`), NumPy

---
**Note:** This repository is currently a work in progress for a university assignment.
