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

    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device = device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device = device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device = device
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device = device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device = device
        ).unsqueeze(1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.memory)
    
# -----------------------------------------
# 4. THE TRAINING LOOP
# -----------------------------------------

def train():

    env = DinoGame()

    # Main network
    model = DinoBrain().to(device)

    # Target network
    target_model = DinoBrain().to(device)

    target_model = load_state_dict(
        model.state_dict()
    )

    target_model.eval() # Target network is not trained directly

    # Optimizer

    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.0005
    )

    # Huber Loss 
    loss_fn = nn.SmoothL1Loss()

    # Replay memory

    memory = ReplayBuffer(
        capacity=100000
    )

    # Hyperparameters

    batch_size = 128
    gamma = 0.99          # Discount factor (cares about future rewards)
    epsilon = 1.0         # 100% random actions initially
    epsilon_min = 0.02
    epsilon_decay = 0.997 # Gradually rely on the neural network

    #Do not start learning immediately, let the AI play a few games first to fill the replay buffer
    learning_starts = 2000

    #update the target network every 10 episodes
    target_update_freq = 10

    # total number of episodes to train
    episodes = 3000

    # Score taacking
    scores = deque(maxlen=100)

    best_score = 0
    best_average_score = 0

    # Episode loop

    for episode in range(episodes):

        state_dict = env.reset()

        state = extract_state(
            state_dict
        )

        done = False
        total_reward = 0.0

        episode_loss = 0.0
        training_steps = 0

        # Frame Loop

        while not done:

            # Keep Pygame window responsive
            
            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    pygame.quit()
                    sys.exit()

            # Episode greedy action

            if random.random() < epsilon:

                # Exploration
                action = random.randint(0, 2) # Random action

            else:

                #Use trained network
                state_tensor = torch.FloatTensor(
                    state,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                with torch.no_grad():

                    q_values = model(state_tensor)

                action = torch.argmax(
                    q_values,
                    dim=1
                ).item()


            # Execute Action
            next_state_dict, reward, done = env.step(
                action
            )
            
            # ------------------------------------------------
            # ACTION PENALTIES
            # ------------------------------------------------
            #
            # Prevent unnecessary action spam.
            #
            # Duck penalty is smaller because the Dino may
            # need to stay ducked for multiple frames.
            # ------------------------------------------------

            if action == 1:
                reward -= 0.01

            elif action == 2:
                reward -= 0.002

            # Next state
                             
            next_state = extract_state(
                next_state_dict
            )
            
            # Save to memory
            memory.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state

            total_reward += reward

            #  TRAIN THE NETWORK 

            if len(memory) > learning_starts:

                (
                    b_states,
                    b_actions,
                    b_rewards,
                    b_next_states,
                    b_dones
                ) = memory.sample(
                    batch_size
                )
                
                # Current Q-Values

                current_q = model(
                    b_states
                    ).gather(1, b_actions)
                
                # Target Q-Values (Bellman Equation)
                 # ---------------------------------------------
                # DOUBLE DQN
                # ---------------------------------------------
                #
                # Main model selects the next action.
                #
                # Target model evaluates that action.
                # ---------------------------------------------

                with torch.no_grad():

                    next_actions = model(
                        b_next_states
                    ).argmax(
                        dim=1,
                        keepdim=True
                    )

                    next_q_values = target_model(
                        b_next_states
                    ).gather(
                        1,
                        next_actions
                    )

                    target_q = (

                        b_rewards + gamma * next_q_values * (1 - b_dones)
                    )
                    
                # Loss

                loss = loss_fn(current_q, target_q)

                # Backpropagation and optimization
                optimizer.zero_grad()

                loss.backward()

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10.0
                )

                optimizer.step()

                episode_loss += loss.item()
                training_steps += 1

            # Render the game so you can watch
            env.draw()
            # You can increase FPS here to make training faster visually
            env.clock.tick(120) 

        # ====================================================
        # END OF EPISODE
        # ====================================================


        # ----------------------------------------------------
        # REDUCE EXPLORATION
        # ----------------------------------------------------

        epsilon = max(
            epsilon_min, 
            epsilon * epsilon_decay
        )

        # Update target network
        if (episode + 1) % target_update_freq == 0:
            target_model.load_state_dict(
                model.state_dict()
            )

        # score stats
        scores.append(env.score)

        average_score = sum(scores) / len(scores)

        # Calculate average loss for this episode
        if training_steps > 0:
            average_loss = episode_loss / training_steps
        else:
            average_loss = 0

        # save best single score model
        if env.score > best_score:

            best_score = env.score

            torch.save(
                model.state_dict(),
                "best_dino.pth"
            )

            print(
                f"\n*** NEW BEST SCORE: "
                f"{int(best_score)} ***"
            )

         # ----------------------------------------------------
        # SAVE BEST AVERAGE MODEL
        # ----------------------------------------------------

        if (
            len(scores) >= 20
            and
            average_score > best_average_score
        ):

            best_average_score = average_score

            torch.save(
                model.state_dict(),
                "best_average_dino.pth"
            )

            print(
                f"*** NEW BEST AVERAGE SCORE: "
                f"{best_average_score:.2f} ***"
            )

        
# ----------------------------------------------------
        # SAVE LATEST MODEL
        # ----------------------------------------------------

        torch.save(
            model.state_dict(),
            "latest_dino.pth"
        )


        # ----------------------------------------------------
        # PRINT TRAINING STATUS
        # ----------------------------------------------------

        print(
            f"Episode: {episode + 1}/{episodes} | "
            f"Score: {int(env.score)} | "
            f"Avg100: {average_score:.2f} | "
            f"Reward: {total_reward:.2f} | "
            f"Loss: {average_loss:.5f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Memory: {len(memory)}"
        )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")

    print(
        f"Best Score: "
        f"{int(best_score)}"
    )

    print(
        f"Best Average Score: "
        f"{best_average_score:.2f}"
    )

    pygame.quit()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    train()