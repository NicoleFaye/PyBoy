"""
Performance-optimized environment wrapper.
This module provides a low-level optimized implementation of key environment operations.
"""
import numpy as np
from enum import Enum

# Define action mappings
ACTION_MAPPINGS = {
    1: ("left", None),     # LEFT_FLIPPER_PRESS
    2: ("a", None),        # RIGHT_FLIPPER_PRESS
    3: (None, "left"),     # LEFT_FLIPPER_RELEASE
    4: (None, "a"),        # RIGHT_FLIPPER_RELEASE
    5: ("down", None),     # LEFT_TILT
    6: ("b", None),        # RIGHT_TILT
    7: ("select", None),   # UP_TILT
    8: ("select+down", None),  # LEFT_UP_TILT
    9: ("select+b", None), # RIGHT_UP_TILT
}


def optimize_step(pyboy, action, headless=True):
    """
    Optimized environment step function.
    This bypasses some of the object-oriented checks for better performance.
    
    Args:
        pyboy: PyBoy instance
        action: The action ID (0-9)
        headless: Whether to run in headless mode
    
    Returns:
        None
    """
    # Skip processing for idle action (0)
    if action > 0:
        # Get the action mapping
        buttons = ACTION_MAPPINGS.get(action)
        if not buttons:
            return  # Invalid action
            
        press_button, release_button = buttons
        
        # Handle combination buttons
        if press_button and "+" in press_button:
            button1, button2 = press_button.split("+")
            pyboy.button(button1)
            pyboy.button(button2)
        # Handle normal buttons
        elif press_button:
            pyboy.button_press(press_button)
        
        # Handle releases
        if release_button:
            pyboy.button_release(release_button)
    
    # Tick the emulator (optimized)
    pyboy.tick(1, not headless)


def fast_observation(pyboy):
    """
    Get an observation from the environment as efficiently as possible.
    
    Args:
        pyboy: PyBoy instance
        
    Returns:
        The observation
    """
    return pyboy.game_area()


def fast_info(game_wrapper, info_level=0):
    """
    Get info dictionary with minimal overhead.
    
    Args:
        game_wrapper: PyBoy game wrapper
        info_level: Level of detail (0-3)
        
    Returns:
        Dictionary of info
    """
    # Level 0 - Absolute minimal info
    if info_level == 0:
        return {"score": game_wrapper.score}
    
    # Level 1 - Basic info (efficient)
    if info_level == 1:
        return {
            "score": game_wrapper.score,
            "current_stage": game_wrapper.current_stage
        }
    
    # Higher levels as needed
    info = {
        "score": game_wrapper.score,
        "current_stage": game_wrapper.current_stage
    }
    
    # Add more info for higher levels
    if info_level >= 2:
        info.update({
            "ball_x": game_wrapper.ball_x,
            "ball_y": game_wrapper.ball_y,
            "ball_type": game_wrapper.ball_type,
        })
        
        if game_wrapper.ball_saver_seconds_left > 0:
            info["saver_active"] = True
    
    return info