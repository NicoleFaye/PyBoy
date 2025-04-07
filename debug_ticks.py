#!/usr/bin/env python3
"""
Debug script to analyze what's happening with PyBoy ticks and screen updates.
"""
import time
import numpy as np
from pyboy import PyBoy

# Set up PyBoy with visible window
rom_path = "/home/nicole-demera/PyBoy/test_roms/secrets/pokemon_pinball.gbc"
pyboy = PyBoy(rom_path, window="SDL2")
print(f"PyBoy initialized with ROM: {pyboy.cartridge_title}")

# Get game wrapper
game_wrapper = pyboy.game_wrapper
print(f"Game wrapper type: {type(game_wrapper).__name__}")

# Start the game
print("Starting game...")
game_wrapper.start_game()
print("Game started.")

# Function to print game state
def print_game_state():
    print(f"Score: {game_wrapper.score}")
    print(f"Balls left: {game_wrapper.balls_left}")
    print(f"Game over: {game_wrapper.game_over}")
    print(f"Current stage: {game_wrapper.current_stage}")
    # Print top-left corner of game area to see if it's changing
    game_area = pyboy.game_area()
    print(f"Game area shape: {game_area.shape}")
    print(f"Game area top-left (5x5):")
    print(game_area[0:5, 0:5])
    print("-" * 40)

# Run ticks in batches, checking the game state after each batch
total_ticks = 0
ticks_per_batch = 30

# Initial state
print("\nInitial state (before any ticks):")
print_game_state()

# Run ticks in batches
for batch in range(10):  # 10 batches
    # Run a batch of ticks
    print(f"\nRunning batch {batch+1} ({ticks_per_batch} ticks)...")
    for _ in range(ticks_per_batch):
        pyboy.tick()
    total_ticks += ticks_per_batch
    
    # Print state after batch
    print(f"State after {total_ticks} total ticks:")
    print_game_state()
    
    # Pause to allow viewing the current state
    time.sleep(1)

# Test button presses
print("\nTesting button presses...")
print("Pressing start to begin game...")
pyboy.button_press("start")
pyboy.tick(10)
pyboy.button_release("start")
pyboy.tick(30)
print("State after pressing start:")
print_game_state()

print("\nPressing A button (right flipper)...")
pyboy.button_press("a")
pyboy.tick(10)
pyboy.button_release("a")
pyboy.tick(30)
print("State after pressing A:")
print_game_state()

# Check environment observations
print("\nChecking environment observations...")
print("Creating normalized observation (float32 / 255.0):")
normalized_obs = pyboy.game_area().astype(np.float32) / 255.0
print(f"Normalized observation shape: {normalized_obs.shape}")
print(f"Normalized observation range: {normalized_obs.min()} to {normalized_obs.max()}")
print(f"Average pixel value: {normalized_obs.mean()}")

# Clean up
print("\nStopping PyBoy...")
pyboy.stop()
print("Debug complete.")