# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run tests: `pytest tests/`
- Run a single test: `pytest tests/test_file.py::test_function`
- Run AI training: `python train.py --rom [ROM_PATH] --agent [dqn|sb3] --reward-shaping [basic|catch_focused|comprehensive]`
- PyBoy scripts: `python -m pyboy [ROM_PATH] [OPTIONS]`

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
- Reinforcement learning agents (DQN and SB3) have different implementations but similar interfaces