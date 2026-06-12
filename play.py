 # Same OpenMP fix as before just in case
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import pygame
import sys

# Import from your existing files
from game import DinoGame
from train import DinoBrain, extract_state 
from settings import FPS

def play():
    # 1. Setup the Game and the Brain
    env = DinoGame()
    model = DinoBrain()
    
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load("best_dino.pth"))
        print("Successfully loaded best_dino.pth! Getting ready to play...")
    except FileNotFoundError:
        print("Error: Could not find 'best_dino.pth'. Make sure the file exists in this folder.")
        return

    # Tell PyTorch we are evaluating (playing), not training
    model.eval() 
    
    # Device setup (CPU is perfectly fine for testing this tiny model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. The Play Loop
    state_dict = env.reset()
    state = extract_state(state_dict)

    while True:
        # Handle closing the window gracefully
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- AI MAKES A DECISION ---
        # Notice: No epsilon (randomness). The AI uses 100% brain power.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad(): # We use no_grad() because we don't need to learn/update weights
            q_values = model(state_tensor)
            
        action = torch.argmax(q_values).item()

        # --- AI TAKES ACTION ---
        next_state_dict, reward, done = env.step(action)
        state = extract_state(next_state_dict)

        # Draw the game to the screen
        env.draw()
        
        # Keep it at a normal playable speed (60 FPS) so you can watch
        env.clock.tick(FPS)

        # If the AI crashes, print the score and restart instantly
        if done:
            print(f"AI died. Final Score: {int(env.score)}. Restarting...")
            state_dict = env.reset()
            state = extract_state(state_dict)

if __name__ == "__main__":
   
    
    play()