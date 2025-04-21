"""
Environment implementations for Pokemon Pinball AI.
"""
# Set default for PufferLib availability
PUFFERLIB_AVAILABLE = False

try:
    from environment.pokemon_pinball_env import PokemonPinballEnv, Actions, RewardShaping
    from environment.wrappers import SkipFrame, FrameStack
    
        
except ImportError as e:
    print(f"Error importing environment modules: {e}")
    
# Define __all__ to expose the public API
__all__ = [
    'PokemonPinballEnv',
    'Actions',
    'RewardShaping',
    'SkipFrame',
    'FrameStack'
]
