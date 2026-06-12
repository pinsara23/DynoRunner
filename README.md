# Dino Runner With NN

An endless-runner style Dino game built with `pygame`, plus a small neural network agent built with `PyTorch` that can be trained to play it.

## Features

- Play the game manually with keyboard controls.
- Train a simple DQN-style agent against the game loop.
- Save the best-performing model to `best_dino.pth`.
- Reuse the saved model to run the AI in play mode.

## Requirements

- Python 3.10 or newer
- `pygame`
- `torch`
- `numpy`

## Setup

Install the dependencies with pip:

```bash
pip install pygame torch numpy
```

If you want a virtual environment, create and activate one first, then install the packages there.

## How To Run

Start the manual game:

```bash
python main.py
```

Run the trained AI player:

```bash
python play.py
```

Train the neural network agent:

```bash
python train.py
```

Training will periodically overwrite `best_dino.pth` whenever the current run beats the previous best score.

## Controls

- `Space` or `Up Arrow`: jump
- `Down Arrow`: duck
- `R`: restart after game over

## Project Structure

- `main.py` - manual game loop and keyboard input
- `play.py` - loads `best_dino.pth` and lets the AI play
- `train.py` - trains the neural network agent and saves the best weights
- `game.py` - game environment, scoring, collision checks, and rewards
- `sprites.py` - Dino, obstacle, and ground sprites
- `settings.py` - screen size, physics, colors, and tuning constants
- `best_dino.pth` - saved model weights

## Notes

- The game runs with a fixed 800x400 window.
- Obstacles include both cactus-style ground obstacles and occasional birds.
- The AI state uses the Dino position, velocity, nearest obstacle, and current speed.