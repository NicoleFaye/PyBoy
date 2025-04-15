"""
Environment implementations for Pokemon Pinball AI.
"""
# Set default for PufferLib availability
PUFFERLIB_AVAILABLE = False

try:
    from environment.pokemon_pinball_env import PokemonPinballEnv, Actions, RewardShaping
    from environment.wrappers import SkipFrame, FrameStack
    
    # Try to import PufferLib environment
    try:
        import pufferlib
        from environment.puffer_wrapper import PufferPokemonPinballEnv, make_puffer_env
        PUFFERLIB_AVAILABLE = True
    except ImportError:
        # PufferLib environment is not available
        PUFFERLIB_AVAILABLE = False
        
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

# Add PufferLib components if available
if PUFFERLIB_AVAILABLE:
    __all__.extend([
        'PufferPokemonPinballEnv',
        'make_puffer_env'
    ])