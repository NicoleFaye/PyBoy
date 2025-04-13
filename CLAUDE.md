# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run tests: `pytest tests/`
- Run a single test: `pytest tests/test_file.py::test_function`
- Run AI training: `python train.py --rom [ROM_PATH] --algorithm [dqn|a2c|ppo] --reward-shaping [basic|catch_focused|comprehensive]`
- Resume training: `python train.py --checkpoint [CHECKPOINT_DIR]/[CHECKPOINT_FILE].zip --model-name [MODEL_NAME]`
  - Example: `python train.py --checkpoint checkpoints/ppo_comprehensive/final.zip --model-name ppo_comprehensive`
  - Will automatically find the most recent checkpoint if the specified file doesn't exist
- PyBoy scripts: `python -m pyboy [ROM_PATH] [OPTIONS]`
- Check logs of recent runs base command: cat ./checkpoints/*/*.log

## Code Style
- Follow Google Python style guide
- Use docstrings for all functions/classes (Google style)
- Use type hints for function parameters and return values
- Imports: stdlib first, then third-party, then local modules
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use specific exceptions with informative messages
- Keep line length under 100 characters

## Notes
- For Gymnasium 1.1.1, use `from gymnasium.wrappers import FrameStackObservation as FrameStack`
- PyBoy initialization: `pyboy = PyBoy(rom_path, window="SDL2")` (don't use game_wrapper parameter)
- Access the game wrapper via: `pyboy.game_wrapper`
- PyBoy uses Cython for performance-critical components
- Reinforcement learning uses Stable-Baselines3 with support for DQN, A2C, and PPO algorithms