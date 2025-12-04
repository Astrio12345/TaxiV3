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

