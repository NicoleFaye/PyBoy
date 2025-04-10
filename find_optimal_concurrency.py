#!/usr/bin/env python3
"""
Script to find the optimal number of concurrent training processes.
This benchmarks different configurations to determine the best concurrency level.
"""
import argparse
import subprocess
import time
import os
import signal
import psutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import json


def clean_temp_dirs(base_path, pattern="benchmark_temp_"):
    """Clean up temporary directories from previous runs"""
    print("Cleaning up temporary directories...")
    for path in Path(base_path).glob(f"{pattern}*"):
        if path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"Removed {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")


def create_temp_dirs(base_path, count, pattern="benchmark_temp_"):
    """Create temporary directories for the benchmark processes"""
    print(f"Creating {count} temporary directories...")
    paths = []
    for i in range(count):
        path = Path(base_path) / f"{pattern}{i}"
        path.mkdir(parents=True, exist_ok=True)
        paths.append(path)
    return paths


def start_processes(rom_path, count, headless=True, temp_dirs=None):
    """Start a specified number of training processes"""
    processes = []
    
    algorithms = ["ppo", "a2c"]
    reward_shapes = ["basic", "comprehensive"]
    
    print(f"Starting {count} processes...")
    
    for i in range(count):
        # Cycle through different configurations
        algo = algorithms[i % len(algorithms)]
        reward = reward_shapes[i % len(reward_shapes)]
        
        # Use a different temp dir for each process
        temp_dir = temp_dirs[i] if temp_dirs else None
        save_dir = str(temp_dir) if temp_dir else f"benchmark_temp_{i}"
        
        # Build the command
        cmd = [
            "python", "train.py",
            "--rom", rom_path,
            "--algorithm", algo,
            "--reward-shaping", reward,
            "--model-name", f"benchmark_{algo}_{reward}_{i}",
            "--episodes", "10",  # Short training run
            "--frame-skip", "4",
            "--frame-stack", "4",
        ]
        
        if headless:
            cmd.append("--headless")
            
        # Launch the process
        print(f"Launching: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        
        processes.append(proc)
        print(f"Started process {i+1}/{count}: PID {proc.pid}")
        
        # Wait briefly to ensure process has started
        time.sleep(2)
        
        # Verify process is running
        try:
            p = psutil.Process(proc.pid)
            print(f"Process {proc.pid} status: {p.status()}")
        except psutil.NoSuchProcess:
            print(f"WARNING: Process {proc.pid} not found. It may have failed to start.")
        
    return processes


def monitor_processes(processes, duration=300, interval=5):
    """Monitor the processes for a specified duration"""
    start_time = time.time()
    end_time = start_time + duration
    
    # Lists to store metrics
    timestamps = []
    cpu_usage = []
    memory_usage = []
    proc_metrics = {proc.pid: {"cpu": [], "memory": []} for proc in processes}
    
    # Create process objects for monitoring
    process_objects = {}
    for proc in processes:
        try:
            p = psutil.Process(proc.pid)
            # Do an initial CPU percent call to initialize the counter
            p.cpu_percent(interval=None)
            process_objects[proc.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Could not monitor process {proc.pid}: {e}")
    
    # Check for child processes (PyBoy might spawn children)
    for pid, p in process_objects.items():
        try:
            children = p.children(recursive=True)
            for child in children:
                try:
                    # Initialize CPU monitoring for child
                    child.cpu_percent(interval=None)
                    process_objects[child.pid] = child
                    proc_metrics[child.pid] = {"cpu": [], "memory": []}
                    print(f"Found child process: {child.pid}, parent: {pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print(f"Monitoring {len(process_objects)} processes for {duration} seconds...")
    
    while time.time() < end_time:
        current_time = time.time() - start_time
        timestamps.append(current_time)
        
        # Overall system metrics
        total_cpu = 0
        total_memory = 0
        
        # First, check for any child processes to track
        for proc in list(process_objects.values()):
            try:
                children = proc.children(recursive=True)
                for child in children:
                    if child.pid not in process_objects:
                        try:
                            # Initialize CPU monitoring for new child
                            child.cpu_percent(interval=None)
                            process_objects[child.pid] = child
                            proc_metrics[child.pid] = {"cpu": [], "memory": []}
                            print(f"Found new child process: {child.pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Check each process
        for pid, p in list(process_objects.items()):
            try:
                # Get CPU and memory usage with a short interval
                cpu_percent = p.cpu_percent(interval=0.1)
                memory_info = p.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Add to process-specific metrics if valid
                if pid in proc_metrics:
                    proc_metrics[pid]["cpu"].append(cpu_percent)
                    proc_metrics[pid]["memory"].append(memory_mb)
                
                # Add to totals
                total_cpu += cpu_percent
                total_memory += memory_mb
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have terminated
                if pid in process_objects:
                    print(f"Process {pid} no longer available")
                    del process_objects[pid]
        
        # Add to overall metrics
        cpu_usage.append(total_cpu)
        memory_usage.append(total_memory)
        
        # Also check overall system usage
        system_cpu = psutil.cpu_percent(interval=0.1)
        
        # Status update
        elapsed = int(current_time)
        remaining = int(duration - current_time)
        print(f"Elapsed: {elapsed}s, Remaining: {remaining}s - " +
              f"Process CPU: {total_cpu:.1f}%, System CPU: {system_cpu:.1f}%, " +
              f"Memory: {total_memory:.1f} MB")
        
        # Wait for next interval
        time.sleep(interval)
    
    return {
        "timestamps": timestamps,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "process_metrics": proc_metrics
    }


def terminate_processes(processes):
    """Safely terminate all processes"""
    print(f"Terminating {len(processes)} processes...")
    
    for proc in processes:
        try:
            # Try graceful termination first
            proc.terminate()
        except Exception:
            pass
    
    # Give processes time to terminate gracefully
    time.sleep(5)
    
    # Force kill any remaining processes
    for proc in processes:
        try:
            if proc.poll() is None:  # Process still running
                proc.kill()
        except Exception:
            pass


def calculate_metrics(monitoring_data, process_count):
    """Calculate performance metrics from the monitoring data"""
    cpu_data = np.array(monitoring_data["cpu_usage"])
    memory_data = np.array(monitoring_data["memory_usage"])
    
    # Check for valid data
    if len(cpu_data) < 2:
        print("WARNING: Not enough monitoring data points collected")
        # Return default values
        return {
            "process_count": process_count,
            "avg_cpu": 0,
            "avg_memory": 0,
            "cpu_efficiency": 0,
            "raw_cpu": [],
            "raw_memory": []
        }
    
    # Skip the first few data points (startup) and ensure we have valid data
    skip_points = min(5, len(cpu_data) // 3)
    if len(cpu_data) > skip_points:
        cpu_data = cpu_data[skip_points:]
        memory_data = memory_data[skip_points:]
    
    # Calculate averages
    avg_cpu = float(np.mean(cpu_data))
    avg_memory = float(np.mean(memory_data))
    
    # Calculate efficiency (CPU usage per process)
    # For process count 0, set efficiency to 0
    if process_count <= 0:
        cpu_efficiency = 0
    # For very low CPU usage, consider efficiency poor
    elif avg_cpu < 5.0:
        cpu_efficiency = 0.1
    else:
        # Normal efficiency calculation
        cpu_efficiency = avg_cpu / process_count
    
    print(f"Raw metrics: Process count: {process_count}, Avg CPU: {avg_cpu:.2f}%, " +
          f"Avg Memory: {avg_memory:.2f} MB, CPU Efficiency: {cpu_efficiency:.2f}")
    
    return {
        "process_count": process_count,
        "avg_cpu": float(avg_cpu),
        "avg_memory": float(avg_memory),
        "cpu_efficiency": float(cpu_efficiency),
        "raw_cpu": cpu_data.tolist(),
        "raw_memory": memory_data.tolist()
    }


def plot_results(results):
    """Plot the benchmark results"""
    process_counts = [r["process_count"] for r in results]
    avg_cpus = [r["avg_cpu"] for r in results]
    cpu_efficiencies = [r["cpu_efficiency"] for r in results]
    avg_memories = [r["avg_memory"] for r in results]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot CPU usage
    ax1.plot(process_counts, avg_cpus, 'b-', linewidth=2, label='Total CPU Usage (%)')
    ax1.plot(process_counts, cpu_efficiencies, 'g--', linewidth=2, 
             label='CPU Efficiency (% per process)')
    
    # Find the optimal process count based on CPU efficiency
    optimal_idx = np.argmax(cpu_efficiencies)
    optimal_count = process_counts[optimal_idx]
    
    ax1.axvline(x=optimal_count, color='r', linestyle='--', label=f'Optimal: {optimal_count} processes')
    
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('CPU Usage vs. Process Count')
    ax1.set_xticks(process_counts)
    ax1.grid(True)
    ax1.legend()
    
    # Plot memory usage
    ax2.plot(process_counts, avg_memories, 'm-', linewidth=2, label='Total Memory Usage (MB)')
    ax2.plot(process_counts, [m/p for m, p in zip(avg_memories, process_counts)], 'c--', 
             linewidth=2, label='Memory per Process (MB)')
    
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs. Process Count')
    ax2.set_xticks(process_counts)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('concurrency_benchmark_results.png')
    print("Benchmark results saved to concurrency_benchmark_results.png")
    
    # Also save as JSON for future reference
    with open('concurrency_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return optimal_count


def run_benchmark(rom_path, max_processes, duration=120, headless=True):
    """Run the benchmark with different numbers of processes"""
    results = []
    
    # Determine system specs
    cpu_count = psutil.cpu_count(logical=True)
    physical_cpu_count = psutil.cpu_count(logical=False)
    total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
    
    print(f"System has {physical_cpu_count} physical cores, {cpu_count} logical cores, and {total_memory:.0f} MB RAM")
    
    # Determine process counts to test
    if max_processes <= 0:
        # Use more granular testing for lower core counts
        if physical_cpu_count <= 4:
            # For 1-4 cores, test 1 through 2x physical cores
            max_processes = physical_cpu_count * 2
            process_counts = []
            for i in range(1, max_processes + 1):
                process_counts.append(i)
        else:
            # For higher core counts, test fewer combinations
            max_processes = min(physical_cpu_count * 2, 16)
            # Create a more sparse list focusing on key values
            process_counts = [1, 2]
            # Add values around physical core count
            if physical_cpu_count > 3:
                process_counts.append(physical_cpu_count // 2)
                process_counts.append(physical_cpu_count - 1)
                process_counts.append(physical_cpu_count)
                process_counts.append(physical_cpu_count + 1)
                process_counts.append(physical_cpu_count * 3 // 2)
                process_counts.append(physical_cpu_count * 2)
            # Deduplicate and sort
            process_counts = sorted(list(set(process_counts)))
            # Ensure we don't exceed max_processes
            process_counts = [c for c in process_counts if c <= max_processes]
    else:
        # Use custom process counts 1 through max_processes
        process_counts = list(range(1, max_processes + 1))
    
    print(f"Will test the following process counts: {process_counts}")
    
    # Clean up any existing temp dirs
    clean_temp_dirs(".")
    
    # Run benchmark for each process count
    for count in process_counts:
        print(f"\n=== Testing with {count} concurrent processes ===\n")
        
        # Create temp dirs
        temp_dirs = create_temp_dirs(".", count)
        
        # Start processes
        processes = start_processes(rom_path, count, headless, temp_dirs)
        
        # Short delay to let processes initialize
        print("Waiting for processes to initialize...")
        time.sleep(5)
        
        # Check if processes are still running
        running_count = 0
        for proc in processes:
            if proc.poll() is None:  # None means still running
                running_count += 1
            else:
                print(f"Process {proc.pid} exited with code {proc.returncode}")
                # Capture any error output
                stderr = proc.stderr.read()
                if stderr:
                    print(f"Error output: {stderr[:500]}...")
        
        if running_count < count:
            print(f"WARNING: Only {running_count}/{count} processes are still running")
        
        if running_count == 0:
            print("ERROR: No processes are running. Skipping this test.")
            metrics = {
                "process_count": count,
                "avg_cpu": 0,
                "avg_memory": 0,
                "cpu_efficiency": 0,
                "raw_cpu": [],
                "raw_memory": []
            }
        else:
            # Monitor for specified duration
            print(f"Monitoring {running_count} processes for {duration} seconds...")
            monitoring_data = monitor_processes(processes, duration)
            
            # Calculate metrics
            metrics = calculate_metrics(monitoring_data, running_count)
        
        results.append(metrics)
        
        # Terminate processes
        terminate_processes(processes)
        
        # Brief pause between tests
        time.sleep(10)
    
    # Check if we have valid results
    if not results or all(r["avg_cpu"] == 0 for r in results):
        print("ERROR: No valid benchmark data collected. Check if processes are starting correctly.")
        return 0
    
    # Plot and analyze results
    optimal_count = plot_results(results)
    
    print(f"\n=== Benchmark Results ===")
    print(f"Optimal number of concurrent processes: {optimal_count}")
    
    # Show efficient ranges
    best_idx = np.argmax([r["cpu_efficiency"] for r in results])
    best_process_count = results[best_idx]["process_count"]
    best_efficiency = results[best_idx]["cpu_efficiency"]
    
    # Identify acceptable range (within 90% of max efficiency)
    acceptable_counts = [
        r["process_count"] for r in results
        if r["cpu_efficiency"] >= best_efficiency * 0.9
    ]
    
    if acceptable_counts:
        min_acceptable = min(acceptable_counts)
        max_acceptable = max(acceptable_counts)
        print(f"Acceptable range: {min_acceptable} to {max_acceptable} processes")
        print(f"For maximum throughput: Use {max_acceptable} processes")
        print(f"For efficiency: Use {min_acceptable} processes")
    
    # Clean up temp dirs
    clean_temp_dirs(".")
    
    return optimal_count


def main():
    parser = argparse.ArgumentParser(description="Find optimal concurrency for Pokemon Pinball training")
    parser.add_argument("--rom", type=str, required=True,
                        help="Path to the Pokemon Pinball ROM file")
    parser.add_argument("--max-processes", type=int, default=0,
                        help="Maximum number of processes to test (default: 2x CPU count)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration in seconds to run each test (default: 120)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run in headless mode (default: True)")
    parser.add_argument("--no-headless", action="store_false", dest="headless",
                        help="Run with visualization")
    
    args = parser.parse_args()
    
    # Check that the ROM file exists
    if not os.path.exists(args.rom):
        print(f"Error: ROM file '{args.rom}' not found")
        return 1
    
    try:
        run_benchmark(args.rom, args.max_processes, args.duration, args.headless)
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1


if __name__ == "__main__":
    main()