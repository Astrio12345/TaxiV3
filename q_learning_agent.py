"""
q_learning_agent.py - Q-Learning agent to find optimal paths in Taxi environment
"""

import numpy as np


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """
        Initialize Q-Learning Agent

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with zeros
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def get_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Explore: random action
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        # Exploit: best action from Q-table
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, terminated):
        """
        Update Q-table using Q-learning update rule

        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state, action]

        if terminated:
            # No future reward if episode terminated
            max_next_q = 0
        else:
            # Maximum Q-value for next state
            max_next_q = np.max(self.q_table[next_state])

        # Q-learning update
        new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state, action] = new_q

    def train(self, episodes=1000, max_steps=200, callback=None):
        """
        Train the agent

        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            callback: Optional callback function(episode, total_episodes, avg_reward)
        """
        rewards_history = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                # Select and perform action
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Update Q-table
                self.update_q_table(state, action, reward, next_state, terminated)

                total_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Track rewards
            rewards_history.append(total_reward)

            # Callback for progress updates
            if callback and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                callback(episode + 1, episodes, avg_reward)

        return rewards_history

    def test(self, num_episodes=10, render=False):
        """
        Test the trained agent

        Args:
            num_episodes: Number of test episodes
            render: Whether to render the environment

        Returns:
            Average reward over test episodes
        """
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Use greedy policy (no exploration)
                action = self.get_action(state, epsilon=0)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if render:
                    self.env.render()

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)