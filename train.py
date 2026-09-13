import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import sys

from game import DinoGame
from settings import FPS

# Device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Training on: {device}")

# 1. the state extraction function
def extract_state(state_dict):

    """
    Converts the game state into 7 normalized values:

    0 -> Dino Y position
    1 -> Dino Y velocity
    2 -> Distance to closest upcoming obstacle
    3 -> Obstacle Y position
    4 -> Obstacle width
    5 -> Obstacle height
    6 -> Game speed
    """
    dino_x = state_dict['dino_x']
    dino_y = state_dict['dino_y']
    dino_vel_y = state_dict['dino_vel_y']
    speed = state_dict['speed']

    obstacles = state_dict['obstacles']

    #only consider obsetecles that have not fully passed dino
    upcoming_obstecles = [
        obs
        for obs in obstacles
        if obs[0] + obs[2] > dino_x
    ]

    if upcoming_obstecles:

        # find the closest one
        obs_x, obs_y, obs_w, obs_h = min(
            upcoming_obstecles,
            key=lambda obs: obs[0] 
        ) 

    else:

         # Default values when no obstacle exists
        obs_x = 800
        obs_y = 300
        obs_w = 0
        obs_h = 0

    # Distance between Dino and obstacles

    distance_x = obs_x - dino_x

    # Prevent starting the game with a negative distance (if Dino is already past the obstacle)
    distance_x = max(0, distance_x)

    return np.array(
        [
            dino_y / 400.0,
            (dino_vel_y + 15.0) / 30.0,

            # More useful than absolute obstacle X
            distance_x / 800.0,

            obs_y / 400.0,
            obs_w / 50.0,
            obs_h / 100.0,
            speed / 20.0
        ],
        dtype=np.float32
    )


#2. the neural network model
class DinoBrain(nn.Module):
    def __init__(self, input_dim = 7, output_dim = 3):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            # Output:
            # 0 = Run
            # 1 = Jump
            # 2 = Duck

            nn.Linear(64, output_dim)

        )

    def forward(self, x):
        # Outputs 3 values: Score for Nothing, Score for Jump, Score for Duck
        return self.network(x)

# 3. replay buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    def __len__(self):
        return len(self.memory)
    
# -----------------------------------------
# 4. THE TRAINING LOOP
# -----------------------------------------
def train():
    env = DinoGame()
    model = DinoBrain()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    memory = ReplayBuffer()

    # Hyperparameters
    batch_size = 64
    gamma = 0.99          # Discount factor (cares about future rewards)
    epsilon = 1.0         # 100% random actions initially
    epsilon_min = 0.05
    epsilon_decay = 0.995 # Gradually rely on the neural network

    episodes = 500
    best_score = 0

    for episode in range(episodes):
        state_dict = env.reset()
        state = extract_state(state_dict)
        done = False
        total_reward = 0
        
        while not done:
            # Keep Pygame window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # --- EXPLORE VS EXPLOIT ---
            if random.random() < epsilon:
                action = random.randint(0, 2) # Random action
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item() # AI decides

            # --- TAKE ACTION ---
            next_state_dict, reward, done = env.step(action)

            # --- NEW: PENALIZE SPAMMING ---
            # Give a tiny punishment (-0.05) for jumping or ducking 
            # to force the AI to prefer running normally.
            if action == 1 or action == 2:
                reward -= 0.05
                
            next_state = extract_state(next_state_dict)
            
            # Save to memory
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # --- TRAIN THE NETWORK ---
            if len(memory) > batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(batch_size)
                
                # Current Q-Values
                current_q = model(b_states).gather(1, b_actions)
                
                # Target Q-Values (Bellman Equation)
                with torch.no_grad():
                    max_next_q = model(b_next_states).max(1)[0].unsqueeze(1)
                    target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))
                    
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Render the game so you can watch
            env.draw()
            # You can increase FPS here to make training faster visually
            env.clock.tick(120) 

        # --- END OF EPISODE UPDATES ---
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Save the best brain
        if env.score > best_score:
            best_score = env.score
            torch.save(model.state_dict(), "best_dino.pth")
            print(f"New Best Score: {int(best_score)}! Model Saved.")

        print(f"Episode {episode+1} | Score: {int(env.score)} | Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    train()