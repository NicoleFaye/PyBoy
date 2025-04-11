#!/usr/bin/env python3
"""
Benchmark script for Pokemon Pinball AI training.
This measures performance of different components of the training pipeline.
"""
import argparse
import time
import numpy as np
import psutil
import os
import cProfile
import pstats
import io
import gc
from pathlib import Path
from functools import wraps
import tempfile

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Import from PyBoy project
from pyboy import PyBoy
from environment.pokemon_pinball_env import PokemonPinballEnv, RewardShaping
from environment.wrappers import SkipFrame, NormalizedObservation, EpisodicLifeEnv, FrameStack

# Optional - if it exists, import optimized version
try:
    from environment.low_level_env import optimize_step, fast_observation, fast_info
    OPTIMIZED_ENV = True
    print("Using optimized environment functions")
except ImportError:
    OPTIMIZED_ENV = False
    print("Optimized environment functions not available")


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper


class PerformanceStats:
    """Class to track performance statistics."""
    def __init__(self):
        self.times = {}
        self.counts = {}
        
    def record(self, name, elapsed_time):
        """Record a timing measurement."""
        if name not in self.times:
            self.times[name] = []
            self.counts[name] = 0
        self.times[name].append(elapsed_time)
        self.counts[name] += 1
        
    def get_stats(self, name):
        """Get statistics for a specific measurement."""
        if name not in self.times or not self.times[name]:
            return {"avg": 0, "min": 0, "max": 0, "count": 0, "total": 0}
        
        times = self.times[name]
        return {
            "avg": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "count": self.counts[name],
            "total": np.sum(times)
        }
    
    def get_all_stats(self):
        """Get statistics for all measurements."""
        return {name: self.get_stats(name) for name in self.times}
    
    def report(self):
        """Print a report of all statistics."""
        print("\n=== Performance Statistics ===")
        stats = self.get_all_stats()
        for name, stat in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
            print(f"{name}:")
            print(f"  Count: {stat['count']}")
            print(f"  Avg: {stat['avg']*1000:.2f} ms")
            print(f"  Min: {stat['min']*1000:.2f} ms")
            print(f"  Max: {stat['max']*1000:.2f} ms")
            print(f"  Total: {stat['total']:.2f} sec")


class InstrumentedEnvironment:
    """Environment wrapper that measures performance."""
    def __init__(self, env, stats):
        self.env = env
        self.stats = stats
        
    def step(self, action):
        start = time.perf_counter()
        result = self.env.step(action)
        self.stats.record("environment.step", time.perf_counter() - start)
        return result
    
    def reset(self, **kwargs):
        start = time.perf_counter()
        result = self.env.reset(**kwargs)
        self.stats.record("environment.reset", time.perf_counter() - start)
        return result
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def benchmark_environment(rom_path, num_steps=1000, headless=True):
    """Benchmark the environment step function."""
    print("\n=== Environment Benchmark ===")
    stats = PerformanceStats()
    
    # Create PyBoy instance
    print("Initializing PyBoy...")
    window_type = "null" if headless else "SDL2"
    pyboy = PyBoy(rom_path, window=window_type)
    
    # Create environment
    print("Setting up environment...")
    env = PokemonPinballEnv(pyboy, debug=False, headless=headless)
    env = SkipFrame(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = NormalizedObservation(env)
    env = FrameStack(env, num_stack=4)
    
    # Reset environment
    start = time.perf_counter()
    obs, info = env.reset()
    reset_time = time.perf_counter() - start
    stats.record("environment.reset", reset_time)
    print(f"Environment reset time: {reset_time:.6f} seconds")
    
    # Benchmark stepping
    print(f"Running {num_steps} environment steps...")
    action_space = env.action_space.n
    actions = np.random.randint(0, action_space, size=num_steps)
    
    # Warm-up
    for _ in range(10):
        env.step(0)
    
    # Run benchmark
    steps_per_second_values = []
    start_total = time.perf_counter()
    
    for i, action in enumerate(actions):
        if i % 100 == 0:
            gc.collect()  # Reduce impact of garbage collection
            print(f"Step {i}/{num_steps}...")
            
        # Time the step function
        start = time.perf_counter()
        _, reward, done, _, _ = env.step(action)
        step_time = time.perf_counter() - start
        stats.record("environment.step", step_time)
        
        # Calculate steps per second
        steps_per_second = 1.0 / step_time
        steps_per_second_values.append(steps_per_second)
        
        if done:
            obs, info = env.reset()
    
    total_time = time.perf_counter() - start_total
    steps_per_second_avg = num_steps / total_time
    
    # Report results
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average steps per second: {steps_per_second_avg:.2f}")
    print(f"Average step time: {(total_time / num_steps) * 1000:.2f} ms")
    
    # Clean up
    env.close()
    
    return stats


def benchmark_optimized_vs_standard(rom_path, num_steps=1000, headless=True):
    """Compare performance of optimized vs standard environment functions."""
    if not OPTIMIZED_ENV:
        print("Optimized environment not available, skipping comparison")
        return None
    
    print("\n=== Optimized vs Standard Environment Benchmark ===")
    stats = PerformanceStats()
    
    # Create PyBoy instance
    print("Initializing PyBoy...")
    window_type = "null" if headless else "SDL2"
    pyboy = PyBoy(rom_path, window=window_type)
    game_wrapper = pyboy.game_wrapper
    game_wrapper.start_game()
    
    # Test standard step function
    print("Benchmarking standard environment step...")
    env = PokemonPinballEnv(pyboy, debug=False, headless=headless)
    standard_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        env.step(0)  # IDLE action
        step_time = time.perf_counter() - start
        standard_times.append(step_time)
        stats.record("standard.step", step_time)
    
    # Test optimized step function
    print("Benchmarking optimized environment step...")
    optimized_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        optimize_step(pyboy, 0, headless)
        step_time = time.perf_counter() - start
        optimized_times.append(step_time)
        stats.record("optimized.step", step_time)
    
    # Test standard observation function
    print("Benchmarking standard environment observation...")
    standard_obs_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        env._get_obs()
        obs_time = time.perf_counter() - start
        standard_obs_times.append(obs_time)
        stats.record("standard.observation", obs_time)
    
    # Test optimized observation function
    print("Benchmarking optimized environment observation...")
    optimized_obs_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        fast_observation(pyboy)
        obs_time = time.perf_counter() - start
        optimized_obs_times.append(obs_time)
        stats.record("optimized.observation", obs_time)
    
    # Test standard info function
    print("Benchmarking standard environment info...")
    standard_info_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        env._get_info()
        info_time = time.perf_counter() - start
        standard_info_times.append(info_time)
        stats.record("standard.info", info_time)
    
    # Test optimized info function
    print("Benchmarking optimized environment info...")
    optimized_info_times = []
    
    for _ in range(num_steps):
        start = time.perf_counter()
        fast_info(game_wrapper, info_level=0)
        info_time = time.perf_counter() - start
        optimized_info_times.append(info_time)
        stats.record("optimized.info", info_time)
    
    # Report comparison results
    standard_avg = np.mean(standard_times) * 1000
    optimized_avg = np.mean(optimized_times) * 1000
    standard_obs_avg = np.mean(standard_obs_times) * 1000
    optimized_obs_avg = np.mean(optimized_obs_times) * 1000
    standard_info_avg = np.mean(standard_info_times) * 1000
    optimized_info_avg = np.mean(optimized_info_times) * 1000
    
    improvement_step = (standard_avg - optimized_avg) / standard_avg * 100
    improvement_obs = (standard_obs_avg - optimized_obs_avg) / standard_obs_avg * 100
    improvement_info = (standard_info_avg - optimized_info_avg) / standard_info_avg * 100
    
    print("\n=== Optimization Results ===")
    print(f"Standard step: {standard_avg:.3f} ms")
    print(f"Optimized step: {optimized_avg:.3f} ms")
    print(f"Improvement: {improvement_step:.2f}%")
    
    print(f"\nStandard observation: {standard_obs_avg:.3f} ms")
    print(f"Optimized observation: {optimized_obs_avg:.3f} ms")
    print(f"Improvement: {improvement_obs:.2f}%")
    
    print(f"\nStandard info: {standard_info_avg:.3f} ms")
    print(f"Optimized info: {optimized_info_avg:.3f} ms")
    print(f"Improvement: {improvement_info:.2f}%")
    
    # Clean up
    pyboy.stop()
    
    return stats


def benchmark_memory_usage(rom_path, headless=True):
    """Benchmark memory usage during training."""
    print("\n=== Memory Usage Benchmark ===")
    
    # Create PyBoy instance
    print("Initializing PyBoy...")
    window_type = "null" if headless else "SDL2"
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Create environment
    pyboy = PyBoy(rom_path, window=window_type)
    
    env = PokemonPinballEnv(pyboy, debug=False, headless=headless)
    after_env_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"After environment creation: {after_env_memory:.2f} MB")
    print(f"Environment memory footprint: {after_env_memory - initial_memory:.2f} MB")
    
    # Apply wrappers
    env = SkipFrame(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = NormalizedObservation(env)
    env = FrameStack(env, num_stack=4)
    after_wrappers_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"After wrappers: {after_wrappers_memory:.2f} MB")
    print(f"Wrappers memory footprint: {after_wrappers_memory - after_env_memory:.2f} MB")
    
    # Initialize SB3 agent
    try:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=0)
        after_model_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"After model creation: {after_model_memory:.2f} MB")
        print(f"Model memory footprint: {after_model_memory - after_wrappers_memory:.2f} MB")
    except ImportError:
        print("Stable-Baselines3 not installed, skipping model memory usage")
    
    # Run some steps to measure peak usage
    print("Running steps to measure peak memory usage...")
    for _ in range(100):
        env.step(0)
    
    peak_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Total memory growth: {peak_memory - initial_memory:.2f} MB")
    
    # Clean up
    env.close()
    
    return {
        "initial_memory": initial_memory,
        "environment_memory": after_env_memory,
        "wrappers_memory": after_wrappers_memory,
        "peak_memory": peak_memory
    }


def profile_environment(rom_path, num_steps=100, headless=True):
    """Profile the environment using cProfile."""
    print("\n=== Environment Profiling ===")
    
    # Create PyBoy instance
    print("Initializing PyBoy...")
    window_type = "null" if headless else "SDL2"
    pyboy = PyBoy(rom_path, window=window_type)
    
    # Create environment
    env = PokemonPinballEnv(pyboy, debug=False, headless=headless)
    env = SkipFrame(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = NormalizedObservation(env)
    env = FrameStack(env, num_stack=4)
    
    # Reset environment
    obs, info = env.reset()
    
    # Profile stepping
    print(f"Profiling {num_steps} environment steps...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done:
            obs, info = env.reset()
    
    profiler.disable()
    
    # Get and print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # top 20 functions
    print(s.getvalue())
    
    # Save to file
    with open("environment_profile.txt", "w") as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
        ps.print_stats(50)  # top 50 functions
    
    print("Detailed profile saved to environment_profile.txt")
    
    # Clean up
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pokemon Pinball AI training")
    parser.add_argument("--rom", type=str, required=True, help="Path to the ROM file")
    parser.add_argument("--test", type=str, choices=["environment", "optimized", "memory", "profile", "all"], 
                        default="all", help="Which benchmark to run")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to benchmark")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    
    args = parser.parse_args()
    
    # Check that the ROM file exists
    if not os.path.exists(args.rom):
        print(f"Error: ROM file '{args.rom}' not found")
        return 1
    
    results = {}
    
    try:
        if args.test in ["environment", "all"]:
            env_stats = benchmark_environment(args.rom, args.steps, args.headless)
            results["environment"] = env_stats.get_all_stats()
        
        if args.test in ["optimized", "all"] and OPTIMIZED_ENV:
            opt_stats = benchmark_optimized_vs_standard(args.rom, args.steps // 10, args.headless)
            if opt_stats:
                results["optimized"] = opt_stats.get_all_stats()
        
        if args.test in ["memory", "all"]:
            memory_info = benchmark_memory_usage(args.rom, args.headless)
            results["memory"] = memory_info
        
        if args.test in ["profile", "all"]:
            profile_environment(args.rom, args.steps // 10, args.headless)
    
        # Overall report
        print("\n=== Benchmark Summary ===")
        
        if "environment" in results:
            env_step = results["environment"]["environment.step"]
            print(f"Environment step: {env_step['avg']*1000:.3f} ms average")
            print(f"Steps per second: {1.0/env_step['avg']:.2f}")
        
        if "optimized" in results:
            std_step = results["optimized"]["standard.step"]["avg"]
            opt_step = results["optimized"]["optimized.step"]["avg"]
            improvement = (std_step - opt_step) / std_step * 100
            print(f"Optimization improvement: {improvement:.2f}%")
        
        if "memory" in results:
            mem = results["memory"]
            print(f"Memory usage: {mem['peak_memory']:.2f} MB peak")
            
        return 0
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1


if __name__ == "__main__":
    main()