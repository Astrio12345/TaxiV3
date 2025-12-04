# TaxiV3

The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.

## Description

There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the 5x5 grid world. The taxi starts off at a random square and the passenger at one of the designated locations.

The goal is move the taxi to the passenger’s location, pick up the passenger, move to the passenger’s desired destination, and drop off the passenger. Once the passenger is dropped off, the episode ends.

The player receives positive rewards for successfully dropping-off the passenger at the correct location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and for each step where another reward is not received.

It is an interactive GUI application that uses Reinforcement Learning to find optimal paths in the classic Taxi problem from OpenAI Gymnasium. Watch as an AI agent learns to navigate a grid world, pick up passengers, and deliver them to their destinations.

## Features

- Interactive Configuration: Set custom taxi positions, passenger locations, and destinations through an intuitive GUI
- Intelligent Path Finding: Uses Q-Learning algorithm to find optimal navigation paths
- Real-time Training: Watch the agent learn with live training progress updates
- Visual Execution: Step-by-step visualization of the agent's decisions and movements
- Adaptive Learning: Fine-tunes the model for specific scenarios to ensure optimal performance
- Performance Metrics: Displays total steps, rewards, and success rates

## Technologies & Concepts

### Reinforcement Learning (RL)
A machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards over time.

### Q-Learning
A model-free reinforcement learning algorithm that learns the value of actions in particular states. It builds a Q-table that maps state-action pairs to expected rewards, enabling the agent to choose optimal actions without requiring a model of the environment.

### Gymnasium (OpenAI Gym)
A toolkit for developing and comparing reinforcement learning algorithms. Provides standardized environments (like the Taxi problem) with consistent APIs for training and testing RL agents.

### Taxi Problem
A classic RL benchmark where a taxi must navigate a 5×5 grid to pick up a passenger from one of four locations and drop them off at another. The agent must learn efficient navigation while avoiding walls and handling pickup/dropoff logic.

## echnology Stack

Python 3.8+: Core programming language
Gymnasium: RL environment framework
NumPy: Numerical computations and Q-table management
Tkinter: Cross-platform GUI framework for the control interface
Pygame: Visual rendering of the taxi environment (optional)
Threading: Asynchronous training and execution

##  Performance

Training Episodes: Configurable (default: 1000)
Average Training Time: 10-30 seconds (depending on episodes)
Success Rate: ~95%+ after training
Optimal Steps: Typically 10-20 steps for most configurations

## How It Works

Environment Setup: Configure the initial state with custom positions
Agent Training: The Q-Learning agent trains on thousands of episodes, building a Q-table of optimal actions
Fine-tuning: Agent performs focused training on the specific scenario you configured
Path Execution: Agent uses learned policy to navigate optimally from start to goal
Visualization: Watch each step with detailed action logs and reward tracking

🔬 Q-Learning Algorithm
The agent uses the Bellman equation to update its Q-values:
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

Where:
α (alpha): Learning rate - how much new information overrides old
γ (gamma): Discount factor - importance of future rewards
ε (epsilon): Exploration rate - balance between exploration and exploitation
