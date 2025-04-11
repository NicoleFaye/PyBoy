#!/usr/bin/env python3
"""
Profiling script for Pokemon Pinball AI training.
This identifies bottlenecks in the training process.
"""
import cProfile
import io
import pstats
import time
from pathlib import Path
import os
import psutil
import argparse
import tempfile

# Make sure matplotlib doesn't use too many resources
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from train import parse_args, setup_environment, setup_agent, train


def profile_environment_step(env, num_steps=1000):
    """Profile the environment step method."""
    print(f"Profiling environment.step() for {num_steps} steps...")
    
    action = 0  # Use a simple action
    pr = cProfile.Profile()
    pr.enable()
    
    # Run the environment for several steps
    start_time = time.time()
    for _ in range(num_steps):
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()
    
    pr.disable()
    
    elapsed_time = time.time() - start_time
    steps_per_second = num_steps / elapsed_time
    
    print(f"Environment performance: {steps_per_second:.2f} steps/second")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Sort stats by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    return ps


def profile_training_loop(agent, env, num_steps=1000):
    """Profile the full training loop."""
    print(f"Profiling full training loop for {num_steps} steps...")
    
    # Mock the model methods to isolate training infrastructure
    if hasattr(agent, 'model'):
        # Store the real predict method
        original_predict = agent.model.predict
        # Mock method that doesn't do actual inference
        agent.model.predict = lambda *args, **kwargs: (0, None)
    
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    
    # Run the agent's step method
    obs, info = env.reset()
    for _ in range(num_steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.cache(obs, next_obs, action, reward, done)
        obs = next_obs
        if done:
            obs, info = env.reset()
    
    pr.disable()
    
    # Restore the original predict method
    if hasattr(agent, 'model'):
        agent.model.predict = original_predict
    
    elapsed_time = time.time() - start_time
    steps_per_second = num_steps / elapsed_time
    
    print(f"Training loop performance: {steps_per_second:.2f} steps/second")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Sort stats by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    return ps


def profile_logging(agent, env, save_dir, num_iterations=20):
    """Profile the metrics logging components."""
    from utils.logger import MetricLogger
    
    print(f"Profiling metrics logging for {num_iterations} iterations...")
    
    # Setup logger
    logger = MetricLogger(save_dir, resume=False)
    
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    
    # Simulate logging activity
    for i in range(num_iterations):
        # Log step data
        logger.log_step(reward=10.0, loss=0.5, q=0.8, info={'score': 1000 + i})
        
        # Log episode completion
        logger.log_episode()
        
        # Record metrics periodically
        if i % 5 == 0:
            logger.record(episode=i, epsilon=0.1, step=i*100)
    
    pr.disable()
    
    elapsed_time = time.time() - start_time
    iterations_per_second = num_iterations / elapsed_time
    
    print(f"Logging performance: {iterations_per_second:.2f} iterations/second")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Sort stats by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    return ps


def print_memory_usage():
    """Print current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")


def main():
    """Main function for profiling various components."""
    # Use a temporary directory for profiling outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Profile Pokemon Pinball AI training")
        parser.add_argument("--rom", type=str, required=True, help="Path to the ROM file")
        parser.add_argument("--profile", choices=["env", "train", "logging", "all"], 
                            default="all", help="Component to profile")
        parser.add_argument("--steps", type=int, default=1000, 
                            help="Number of steps to profile")
        parser.add_argument("--headless", action="store_true", 
                            help="Run without visualization (fastest training)")
        
        # Parse args and convert to train.py format
        profile_args = parser.parse_args()
        
        # Convert to the format expected by train.py
        train_args = parse_args([
            "--rom", profile_args.rom,
            "--algorithm", "ppo",
            "--episodes", "1",
            "--frame-skip", "4",
            "--frame-stack", "4",
            "--reward-shaping", "basic",
        ])
        
        if profile_args.headless:
            train_args.headless = True
        
        print("Setting up environment...")
        env = setup_environment(train_args)
        
        print("Setting up agent...")
        agent = setup_agent(train_args, env, save_dir)
        
        # Initialize the agent
        from utils.logger import MetricLogger
        logger = MetricLogger(save_dir, resume=False)
        agent.initialize(env, logger)
        
        print("Starting profiling...")
        print_memory_usage()
        
        # Profile the selected component
        if profile_args.profile in ["env", "all"]:
            profile_environment_step(env, profile_args.steps)
            print_memory_usage()
            
        if profile_args.profile in ["train", "all"]:
            profile_training_loop(agent, env, profile_args.steps)
            print_memory_usage()
            
        if profile_args.profile in ["logging", "all"]:
            profile_logging(agent, env, save_dir, profile_args.steps // 50)
            print_memory_usage()
        
        print("Profiling complete")
        env.close()


if __name__ == "__main__":
    main()