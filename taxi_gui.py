"""
taxi_gui.py - Main GUI application for Taxi Environment

PyCharm Project Structure:
taxi_project/
├── taxi_env.py          # (The code you provided - save it here)
├── taxi_gui.py          # (This file - main GUI application)
├── q_learning_agent.py  # (Agent that finds the path)
└── requirements.txt     # (Dependencies)

requirements.txt contents:
gymnasium
numpy
pygame
tkinter (usually comes with Python)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import gymnasium as gym
from taxi_env import TaxiEnv
from q_learning_agent import QLearningAgent
import threading


class TaxiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Taxi Environment - Path Finder")
        self.root.geometry("600x700")

        self.env = None
        self.agent = None
        self.is_running = False

        # Location names and coordinates
        self.locations = {
            "Red (0,0)": 0,
            "Green (0,4)": 1,
            "Yellow (4,0)": 2,
            "Blue (4,3)": 3
        }

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Taxi Environment Controller",
                         font=("Arial", 18, "bold"))
        title.pack(pady=10)

        # Configuration Frame
        config_frame = tk.LabelFrame(self.root, text="Environment Configuration",
                                     font=("Arial", 12, "bold"), padx=20, pady=20)
        config_frame.pack(padx=20, pady=10, fill="x")

        # Taxi Position
        tk.Label(config_frame, text="Taxi Start Position:",
                 font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=5)

        taxi_pos_frame = tk.Frame(config_frame)
        taxi_pos_frame.grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(taxi_pos_frame, text="Row:").pack(side="left")
        self.taxi_row = tk.Spinbox(taxi_pos_frame, from_=0, to=4, width=5)
        self.taxi_row.pack(side="left", padx=5)

        tk.Label(taxi_pos_frame, text="Col:").pack(side="left", padx=(10, 0))
        self.taxi_col = tk.Spinbox(taxi_pos_frame, from_=0, to=4, width=5)
        self.taxi_col.pack(side="left", padx=5)

        # Passenger Location
        tk.Label(config_frame, text="Passenger Location:",
                 font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=5)

        self.passenger_loc = ttk.Combobox(config_frame,
                                          values=list(self.locations.keys()),
                                          state="readonly", width=20)
        self.passenger_loc.grid(row=1, column=1, sticky="w", padx=10)
        self.passenger_loc.current(0)

        # Destination Location
        tk.Label(config_frame, text="Destination:",
                 font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=5)

        self.destination_loc = ttk.Combobox(config_frame,
                                            values=list(self.locations.keys()),
                                            state="readonly", width=20)
        self.destination_loc.grid(row=2, column=1, sticky="w", padx=10)
        self.destination_loc.current(1)

        # Training Parameters Frame
        train_frame = tk.LabelFrame(self.root, text="Training Parameters",
                                    font=("Arial", 12, "bold"), padx=20, pady=20)
        train_frame.pack(padx=20, pady=10, fill="x")

        tk.Label(train_frame, text="Training Episodes:",
                 font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=5)

        self.episodes = tk.Spinbox(train_frame, from_=100, to=10000,
                                   increment=100, width=10)
        self.episodes.delete(0, tk.END)
        self.episodes.insert(0, "1000")
        self.episodes.grid(row=0, column=1, sticky="w", padx=10)

        # Buttons Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.train_button = tk.Button(button_frame, text="Train Agent",
                                      command=self.train_agent,
                                      bg="#4CAF50", fg="white",
                                      font=("Arial", 12, "bold"),
                                      padx=20, pady=10)
        self.train_button.grid(row=0, column=0, padx=10)

        self.run_button = tk.Button(button_frame, text="Find & Show Path",
                                    command=self.run_environment,
                                    bg="#2196F3", fg="white",
                                    font=("Arial", 12, "bold"),
                                    padx=20, pady=10,
                                    state="disabled")
        self.run_button.grid(row=0, column=1, padx=10)

        # Progress Frame
        progress_frame = tk.LabelFrame(self.root, text="Progress",
                                       font=("Arial", 12, "bold"), padx=20, pady=20)
        progress_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.progress_text = tk.Text(progress_frame, height=15, width=60,
                                     font=("Courier", 9))
        self.progress_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(progress_frame, command=self.progress_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.progress_text.config(yscrollcommand=scrollbar.set)

    def log(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update()

    def train_agent(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return

        self.is_running = True
        self.train_button.config(state="disabled")
        self.run_button.config(state="disabled")
        self.progress_text.delete(1.0, tk.END)

        # Run training in a separate thread
        thread = threading.Thread(target=self._train_agent_thread)
        thread.start()

    def _train_agent_thread(self):
        try:
            self.log("Creating environment...")
            self.env = gym.make('Taxi-v3')

            episodes = int(self.episodes.get())
            self.log(f"Training agent for {episodes} episodes...")
            self.log("This may take a moment...\n")

            self.agent = QLearningAgent(self.env)
            self.agent.train(episodes, callback=self.training_callback)

            self.log("\n✓ Training completed successfully!")
            self.log(f"Agent is ready to find paths.")

            self.run_button.config(state="normal")

        except Exception as e:
            self.log(f"\n✗ Error during training: {str(e)}")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
        finally:
            self.is_running = False
            self.train_button.config(state="normal")

    def training_callback(self, episode, total_episodes, avg_reward):
        if episode % 100 == 0:
            self.log(f"Episode {episode}/{total_episodes} - Avg Reward: {avg_reward:.2f}")

    def run_environment(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Environment is already running!")
            return

        if self.agent is None:
            messagebox.showwarning("Warning", "Please train the agent first!")
            return

        try:
            taxi_row = int(self.taxi_row.get())
            taxi_col = int(self.taxi_col.get())
            passenger_idx = self.locations[self.passenger_loc.get()]
            destination_idx = self.locations[self.destination_loc.get()]

            if passenger_idx == destination_idx:
                messagebox.showwarning("Warning",
                                       "Passenger and destination cannot be the same!")
                return

            self.is_running = True
            self.run_button.config(state="disabled")
            self.train_button.config(state="disabled")

            # Run visualization in a separate thread
            thread = threading.Thread(target=self._run_environment_thread,
                                      args=(taxi_row, taxi_col, passenger_idx, destination_idx))
            thread.start()

        except ValueError as e:
            messagebox.showerror("Error", "Invalid input values!")

    def _run_environment_thread(self, taxi_row, taxi_col, passenger_idx, destination_idx):
        try:
            self.log("\n" + "=" * 50)
            self.log("FINDING PATH...")
            self.log("=" * 50)

            # Create environment with rendering
            env = TaxiEnv(render_mode="human")

            # Set initial state
            initial_state = env.encode(taxi_row, taxi_col, passenger_idx, destination_idx)
            env.s = initial_state
            env.lastaction = None

            self.log(f"Start: Taxi at ({taxi_row},{taxi_col})")
            self.log(f"Passenger at: {self.passenger_loc.get()}")
            self.log(f"Destination: {self.destination_loc.get()}\n")

            # Run agent
            state = initial_state
            total_reward = 0
            steps = 0

            terminated = False
            truncated = False

            while not (terminated or truncated) and steps < 200:
                action = self.agent.get_action(state, epsilon=0)  # Greedy policy
                state, reward, terminated, truncated, info = env.step(action)

                action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
                self.log(f"Step {steps + 1}: {action_names[action]} (Reward: {reward})")

                total_reward += reward
                steps += 1

                import time
                time.sleep(0.5)  # Slow down for visualization

            self.log(f"\n{'=' * 50}")
            self.log(f"COMPLETED!")
            self.log(f"Total Steps: {steps}")
            self.log(f"Total Reward: {total_reward}")
            self.log(f"{'=' * 50}\n")

            env.close()

        except Exception as e:
            self.log(f"\n✗ Error: {str(e)}")
            messagebox.showerror("Error", f"Execution failed: {str(e)}")
        finally:
            self.is_running = False
            self.run_button.config(state="normal")
            self.train_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = TaxiGUI(root)
    root.mainloop()