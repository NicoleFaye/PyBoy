#!/usr/bin/env python3
"""
Multi-Session Training Manager for Pokemon Pinball AI.

This script manages multiple training sessions with different configurations.
Features:
- Loads configurations from a JSON file
- Spawns multiple training processes
- Manages CPU affinity
- Provides status monitoring
- Handles graceful termination
"""

import argparse
import json
import os
import psutil
import signal
import subprocess
import sys
import time
import itertools
from pathlib import Path
import datetime


class MultiTrainer:
    """Manages multiple training sessions for Pokemon Pinball."""
    
    def __init__(self, config_file, rom_path, max_processes=None, dry_run=False):
        """
        Initialize the multi-trainer.
        
        Args:
            config_file: Path to the configuration JSON file
            rom_path: Path to the Pokemon Pinball ROM
            max_processes: Maximum number of concurrent processes (None for no limit)
            dry_run: If True, just print commands without executing
        """
        self.rom_path = rom_path
        self.max_processes = max_processes
        self.dry_run = dry_run
        self.processes = {}  # pid -> process object
        self.configs = []  # List of configurations to run
        self.running = True
        self.load_config(config_file)
        
    def load_config(self, config_file):
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            if 'experiments' not in config:
                raise ValueError("Config file must contain an 'experiments' array")
                
            # Extract all parameter combinations
            param_keys = set()
            param_values = {}
            
            # Collect all unique parameter keys and their values
            for exp in config['experiments']:
                for key, value in exp.items():
                    param_keys.add(key)
                    if key not in param_values:
                        param_values[key] = set()
                    
                    # Handle both single values and lists
                    if isinstance(value, list):
                        param_values[key].update(value)
                    else:
                        param_values[key].add(value)
            
            # Convert sets to lists for easier iteration
            for key in param_values:
                param_values[key] = list(param_values[key])
                
            # Generate all combinations
            param_combinations = []
            for exp in config['experiments']:
                local_combinations = [{}]
                
                for key, value in exp.items():
                    new_combinations = []
                    
                    # If value is a list, generate combinations for each value
                    values_to_use = value if isinstance(value, list) else [value]
                    
                    for combo in local_combinations:
                        for val in values_to_use:
                            new_combo = combo.copy()
                            new_combo[key] = val
                            new_combinations.append(new_combo)
                            
                    local_combinations = new_combinations
                
                param_combinations.extend(local_combinations)
                
            # Remove duplicate combinations
            unique_combinations = []
            seen = set()
            
            for combo in param_combinations:
                # Convert to a hashable representation
                combo_tuple = tuple(sorted((k, str(v)) for k, v in combo.items()))
                
                if combo_tuple not in seen:
                    seen.add(combo_tuple)
                    unique_combinations.append(combo)
                    
            self.configs = unique_combinations
            print(f"Loaded {len(self.configs)} unique configurations")
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    def generate_checkpoint_path(self, config):
        """
        Generate a unique checkpoint directory path for a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Path for checkpoints
        """
        # Extract key components for the directory name
        algorithm = config.get('algorithm', 'unknown')
        policy_type = config.get('policy_type', 'mlp')
        reward_shaping = config.get('reward_shaping', 'comprehensive')
        episode_mode = config.get('episode_mode', 'ball')
        gamma = str(config.get('gamma', '0.99')).replace('.', '')
        lr = str(config.get('lr', '0.00025')).replace('.', '')
        
        # Create a unique timestamp
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        
        # Construct the directory name
        dir_name = f"{algorithm}_{policy_type}_{reward_shaping}_{episode_mode}_g{gamma}_lr{lr}_{timestamp}"
        
        return Path("checkpoints") / dir_name
        
    def build_command(self, config, checkpoint_dir):
        """
        Build the training command for a configuration.
        
        Args:
            config: Configuration dictionary
            checkpoint_dir: Path to the checkpoint directory
            
        Returns:
            Command list for subprocess
        """
        cmd = ["python", "train.py"]
        
        # Add ROM path
        cmd.extend(["--rom", self.rom_path])

        # Add checkpoint directory as model name to avoid creating another directory
        cmd.extend(["--model-name", checkpoint_dir.name])

        cmd.extend(["--headless"])
        
        # Add all other parameters
        for key, value in config.items():
            if key in ['rom']:
                continue  # Skip these as they're already added
                
            param_key = f"--{key.replace('_', '-')}"
            
            if isinstance(value, bool):
                if value:
                    cmd.append(param_key)
            else:
                cmd.extend([param_key, str(value)])
                
        return cmd
        
    def start_process(self, config):
        """
        Start a training process with the given configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Process object or None if dry run
        """
        # Create checkpoint directory first
        checkpoint_dir = self.generate_checkpoint_path(config)
        
        # Check for required dependencies based on config
        if config.get('policy_type') == 'lstm' and config.get('algorithm') == 'ppo':
            try:
                import importlib
                importlib.import_module('sb3_contrib')
            except ImportError:
                print(f"Warning: PPO with LSTM policy requires sb3_contrib package.")
                print(f"Consider installing it with: pip install sb3_contrib")
                print(f"The training will fall back to MLP policy.")
        
        # Build command with the checkpoint directory
        cmd = self.build_command(config, checkpoint_dir)
        cmd_str = " ".join(cmd)
        
        print(f"Starting: {cmd_str}")
        
        if self.dry_run:
            return None
            
        # Create output file
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        output_file = open(f"{checkpoint_dir}/training.log", "w")
        
        # Check if stable-baselines3 is installed
        try:
            import stable_baselines3
        except ImportError as e:
            # Write error to the log file instead of starting the process
            output_file.write("Error: Stable-Baselines3 is not installed. Please install it with:\n")
            output_file.write("pip install stable-baselines3[extra]\n")
            output_file.write(str(e))
            output_file.close()
            print(f"Error: Stable-Baselines3 not installed. Skipping {checkpoint_dir.name}")
            return None
            
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        return process
        
    def set_cpu_affinity(self):
        """
        Set CPU affinity for running processes to balance the load.
        """
        if not self.processes:
            return
            
        # Get all logical CPUs
        cpu_count = psutil.cpu_count(logical=True)
        all_cpus = list(range(cpu_count))
        
        # Determine CPUs per process
        cpus_per_process = max(1, cpu_count // len(self.processes))
        
        print(f"Setting CPU affinity ({cpus_per_process} cores per process)...")
        
        # Assign CPUs to processes
        for i, (pid, _) in enumerate(self.processes.items()):
            try:
                start_idx = (i * cpus_per_process) % cpu_count
                cpu_list = all_cpus[start_idx:start_idx + cpus_per_process]
                if not cpu_list:  # Handle wrap-around
                    cpu_list = all_cpus[start_idx:] + all_cpus[:cpus_per_process - (cpu_count - start_idx)]
                    
                process = psutil.Process(pid)
                process.cpu_affinity(cpu_list)
                print(f"Process {pid} assigned to CPUs: {cpu_list}")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Cannot set affinity for process {pid}: {e}")
                
    def start_training(self):
        """
        Start training processes based on configurations.
        """
        for i, config in enumerate(self.configs):
            # Check if we're at the process limit
            if self.max_processes and len(self.processes) >= self.max_processes:
                print(f"Reached maximum of {self.max_processes} concurrent processes. Waiting...")
                self.wait_for_completion(skip_initial_message=True)
                
            # Start the process
            print(f"\nStarting training job {i+1}/{len(self.configs)}")
            for key, value in config.items():
                print(f"  {key}: {value}")
                
            process = self.start_process(config)
            
            if not self.dry_run and process:
                self.processes[process.pid] = process
                print(f"Started process with PID {process.pid}")
                
            # Set CPU affinity after adding the new process
            if len(self.processes) > 1:
                self.set_cpu_affinity()
                
            # Short delay to avoid race conditions
            time.sleep(1)
            
        if not self.dry_run:
            self.wait_for_completion()
            
    def wait_for_completion(self, skip_initial_message=False):
        """
        Wait for all processes to complete.
        
        Args:
            skip_initial_message: Whether to skip the initial status message
        """
        # Setup for non-blocking input
        import select
        import termios
        import fcntl
        import tty
        import readline
        import threading
        import queue
        import re
        
        # Save terminal state
        old_settings = termios.tcgetattr(sys.stdin)
        
        # Configure terminal for immediate character-by-character input
        tty.setcbreak(sys.stdin.fileno())
        
        # Make stdin non-blocking
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        # Initial setup
        if not skip_initial_message:
            print("\nStarting training process monitor...")
        
        current_view_pid = None
        input_buffer = ""
        last_update_time = 0
        update_interval = 2  # seconds
        
        # Create a queue for threaded metric updates
        metric_queue = queue.Queue()
        
        # Thread function to update process metrics
        def update_process_metrics():
            while True:
                try:
                    # Check if we should exit
                    if not hasattr(threading.current_thread(), 'keep_running') or not threading.current_thread().keep_running:
                        break
                        
                    # Get all running PIDs
                    pids = list(self.processes.keys())
                    if not pids:  # No processes to check
                        time.sleep(0.5)
                        continue
                    
                    # Process batch to reduce system load
                    batch_results = {'completed': [], 'metrics': {}, 'configs': {}}
                    
                    # Check for completed processes
                    for pid in pids:
                        if pid not in self.processes:
                            continue  # Process may have been removed
                            
                        process = self.processes[pid]
                        if process.poll() is not None:  # Process has terminated
                            exit_code = process.returncode
                            status = "completed successfully" if exit_code == 0 else f"failed with code {exit_code}"
                            batch_results['completed'].append((pid, status))
                            continue
                            
                        # Only get config if we don't have it yet
                        if pid not in batch_results['configs']:
                            try:
                                ps_output = subprocess.check_output(
                                    ["ps", "-p", str(pid), "-o", "args="], 
                                    universal_newlines=True
                                ).strip()
                                
                                # Extract model name from command line
                                cmd_parts = ps_output.split()
                                for i, part in enumerate(cmd_parts):
                                    if part == "--model-name" and i+1 < len(cmd_parts):
                                        batch_results['configs'][pid] = cmd_parts[i+1]
                                        break
                            except:
                                batch_results['configs'][pid] = "unknown"
                        
                        # Get metrics
                        try:
                            proc = psutil.Process(pid)
                            cpu_percent = proc.cpu_percent(interval=0.05)
                            memory_mb = proc.memory_info().rss / (1024 * 1024)
                            batch_results['metrics'][pid] = {
                                'cpu': cpu_percent,
                                'memory': memory_mb
                            }
                            
                            # Get progress info if possible
                            if current_view_pid == pid:
                                try:
                                    # Get config name
                                    config_name = batch_results['configs'].get(pid)
                                    if config_name:
                                        log_path = os.path.join("checkpoints", config_name, "training.log")
                                        if os.path.exists(log_path):
                                            # Get last few lines to extract progress info
                                            tail = subprocess.check_output(
                                                ["tail", "-n", "5", log_path], 
                                                universal_newlines=True
                                            )
                                            
                                            # Extract timesteps from log line (e.g., "timesteps: 123/10000")
                                            progress_match = re.search(r'timesteps?:?\s*(\d+)/(\d+)', tail)
                                            if progress_match:
                                                current = int(progress_match.group(1))
                                                total = int(progress_match.group(2))
                                                batch_results['metrics'][pid]['progress'] = {
                                                    'current': current,
                                                    'total': total,
                                                    'percent': (current / total) * 100 if total > 0 else 0
                                                }
                                except:
                                    pass  # Skip progress info if there's an error
                        except:
                            # Use default metrics
                            batch_results['metrics'][pid] = {'cpu': 0, 'memory': 0}
                    
                    # Put results in queue
                    metric_queue.put(batch_results)
                    
                    # Sleep to prevent CPU hogging (shorter than main update interval)
                    time.sleep(0.5)
                    
                except Exception as e:
                    # Put error in queue for debugging
                    metric_queue.put({"error": str(e)})
                    time.sleep(1)  # Sleep on error
        
        # Start metrics thread
        metrics_thread = threading.Thread(target=update_process_metrics)
        metrics_thread.daemon = True  # Dies when main thread exits
        metrics_thread.keep_running = True
        metrics_thread.start()
        
        # Prepare for ncurses-like UI
        try:
            # Store process metrics for smoother display
            pid_to_config = {}
            pid_to_metrics = {}
            completed_processes = []
            command_result = None
            
            while self.processes and self.running:
                current_time = time.time()
                
                # Process any metrics updates from the thread
                try:
                    while not metric_queue.empty():
                        results = metric_queue.get_nowait()
                        
                        # Handle errors if any
                        if "error" in results:
                            command_result = f"Metrics error: {results['error']}"
                            continue
                            
                        # Process completed processes
                        for pid, status in results.get('completed', []):
                            if pid in self.processes:
                                completed_processes.append(f"Process {pid} {status}")
                                del self.processes[pid]
                                if pid in pid_to_metrics:
                                    del pid_to_metrics[pid]
                                if current_view_pid == pid:
                                    current_view_pid = None
                        
                        # Update metrics
                        for pid, metrics in results.get('metrics', {}).items():
                            if pid in self.processes:  # Only update for active processes
                                pid_to_metrics[pid] = metrics
                        
                        # Update configs
                        for pid, config in results.get('configs', {}).items():
                            if pid in self.processes:  # Only update for active processes
                                pid_to_config[pid] = config
                                
                        # Limit completed process messages
                        if len(completed_processes) > 5:
                            completed_processes = completed_processes[-5:]
                            
                except queue.Empty:
                    pass  # No updates from thread
                
                # Prepare display buffer without clearing screen
                display_buffer = []
                
                # ANSI escape codes for cursor control
                display_buffer.append("\033[H\033[J")  # Clear screen and home cursor
                
                # Show header
                if self.processes:
                    running_pids = list(self.processes.keys())
                    display_buffer.append(f"Running: {len(running_pids)} processes")
                    display_buffer.append("-" * 100)
                    
                    # Sort PIDs consistently for stable numbering
                    running_pids.sort()
                    
                    # Create a mapping of proc_num -> pid for user commands
                    proc_to_pid = {}
                    
                    # Display process info
                    for proc_num, pid in enumerate(running_pids, 1):
                        # Store mapping for command processing
                        proc_to_pid[proc_num] = pid
                        
                        config_name = pid_to_config.get(pid, "unknown")
                        metrics = pid_to_metrics.get(pid, {'cpu': 0, 'memory': 0})
                        display_buffer.append(f"#{proc_num} (PID {pid}): {config_name}")
                        
                        # Check if we have progress info
                        if 'progress' in metrics:
                            progress = metrics['progress']
                            percent = progress['percent']
                            # Create progress bar (40 chars wide)
                            bar_width = 40
                            filled_width = int(bar_width * percent / 100)
                            bar = '█' * filled_width + '░' * (bar_width - filled_width)
                            display_buffer.append(f"  Progress: {bar} {percent:.1f}% ({progress['current']}/{progress['total']})")
                            
                        display_buffer.append(f"  CPU {metrics['cpu']:.1f}%, Memory {metrics['memory']:.1f} MB")
                    
                    # Show completed processes
                    if completed_processes:
                        display_buffer.append("\nRecently completed:")
                        for msg in completed_processes:
                            display_buffer.append(f"  {msg}")
                    
                    # If viewing a process log, show it
                    if current_view_pid is not None and current_view_pid in self.processes:
                        # Find process number for current view PID
                        current_proc_num = next((num for num, pid in proc_to_pid.items() if pid == current_view_pid), None)
                        display_buffer.append(f"\nOutput for #{current_proc_num} (PID {current_view_pid}):")
                        display_buffer.append("-" * 100)
                        
                        # Get the log file path
                        config_name = pid_to_config.get(current_view_pid)
                        if config_name:
                            log_path = os.path.join("checkpoints", config_name, "training.log")
                            if os.path.exists(log_path):
                                try:
                                    # Get log output - use grep to look for progress-related lines first
                                    try:
                                        # Get progress lines first
                                        progress_output = subprocess.check_output(
                                            ["grep", "-E", "(steps|timesteps|episode|reward|FPS)", log_path, "|", "tail", "-n", "10"],
                                            shell=True,  # Need shell for pipe
                                            universal_newlines=True,
                                            stderr=subprocess.STDOUT
                                        )
                                    except subprocess.CalledProcessError:
                                        progress_output = ""
                                    
                                    # Then get general log tail
                                    tail_output = subprocess.check_output(
                                        ["tail", "-n", "30", log_path], 
                                        universal_newlines=True
                                    )
                                    
                                    # Combine outputs with header if we have progress info
                                    if progress_output.strip():
                                        display_buffer.append("Progress information:")
                                        display_buffer.append(progress_output.rstrip())
                                        display_buffer.append("\nRecent log output:")
                                    
                                    display_buffer.append(tail_output.rstrip())
                                except Exception as e:
                                    display_buffer.append(f"Error reading log: {e}")
                            else:
                                display_buffer.append(f"Log file not found at {log_path}")
                        else:
                            display_buffer.append("Could not determine log file for this process")
                
                # Show command result if any
                if command_result:
                    display_buffer.append(f"\n{command_result}")
                    command_result = None  # Clear after showing once
                
                # Show input area
                display_buffer.append("\n" + "-" * 100)
                display_buffer.append("Commands: l <#> (view log) | k <#> (kill process) | q (quit)")
                display_buffer.append(f"> {input_buffer}")
                
                # Print display buffer
                print("\n".join(display_buffer), end="")
                sys.stdout.flush()
                
                # Check for user input (non-blocking)
                try:
                    # Poll for input with small timeout
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    
                    if ready:
                        char = sys.stdin.read(1)
                        
                        # Handle Enter key
                        if char == '\n':
                            command = input_buffer.strip()
                            input_buffer = ""
                            
                            if command == "q":
                                raise KeyboardInterrupt
                                
                            elif command.startswith("l "):
                                try:
                                    proc_num = int(command.split()[1])
                                    if proc_num in proc_to_pid:
                                        pid = proc_to_pid[proc_num]
                                        current_view_pid = pid
                                        command_result = f"Now showing output for #{proc_num} (PID {pid})"
                                    else:
                                        command_result = f"Process #{proc_num} not found."
                                        current_view_pid = None
                                except (ValueError, IndexError):
                                    command_result = "Invalid command format. Use 'l <#>'"
                                    
                            elif command.startswith("k "):
                                try:
                                    proc_num = int(command.split()[1])
                                    if proc_num in proc_to_pid:
                                        pid = proc_to_pid[proc_num]
                                        command_result = f"Killing process #{proc_num} (PID {pid})..."
                                        self.kill_process(pid)
                                        if current_view_pid == pid:
                                            current_view_pid = None
                                    else:
                                        command_result = f"Process #{proc_num} not found."
                                except (ValueError, IndexError):
                                    command_result = "Invalid command format. Use 'k <#>'"
                            else:
                                command_result = f"Unknown command: {command}"
                                
                        # Handle backspace/delete
                        elif char in ('\x7f', '\x08'):
                            if input_buffer:
                                input_buffer = input_buffer[:-1]
                                
                        # Handle normal character input
                        elif char.isprintable():
                            input_buffer += char
                    
                except (IOError, OSError):
                    pass  # No input available
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nInterrupted! Stopping all training processes...")
            self.stop_all()
            
        finally:
            # Stop metrics thread
            if 'metrics_thread' in locals():
                metrics_thread.keep_running = False
                metrics_thread.join(timeout=1.0)
                
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Final clear screen and reset cursor
            print("\033[H\033[J", end="")
    def kill_process(self, pid):
        """
        Kill a specific process.
        
        Args:
            pid: The process ID to kill
        """
        if pid not in self.processes:
            print(f"Process {pid} not found.")
            return
            
        process = self.processes[pid]
        
        try:
            # Try graceful termination first
            print(f"Sending SIGTERM to process {pid}...")
            process.terminate()
            
            # Give it some time to terminate
            for _ in range(10):  # 5 seconds max wait
                if process.poll() is not None:
                    break
                time.sleep(0.5)
                
            # If still running, force kill
            if process.poll() is None:
                print(f"Process {pid} not responding, sending SIGKILL...")
                process.kill()
                
            del self.processes[pid]
            print(f"Process {pid} stopped.")
            
        except (psutil.NoSuchProcess, ProcessLookupError):
            # Process might have already terminated
            del self.processes[pid]
            print(f"Process {pid} was already stopped.")
            
    def stop_all(self):
        """Stop all running training processes."""
        self.running = False
        
        if not self.processes:
            print("No processes running.")
            return
            
        print(f"Stopping {len(self.processes)} training processes...")
        
        for pid, process in list(self.processes.items()):
            try:
                # Try graceful termination first
                print(f"Sending SIGTERM to process {pid}...")
                process.terminate()
                
                # Give it some time to terminate
                for _ in range(10):  # 5 seconds max wait
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)
                    
                # If still running, force kill
                if process.poll() is None:
                    print(f"Process {pid} not responding, sending SIGKILL...")
                    process.kill()
                    
                del self.processes[pid]
                print(f"Process {pid} stopped.")
                
            except (psutil.NoSuchProcess, ProcessLookupError):
                # Process might have already terminated
                del self.processes[pid]
                print(f"Process {pid} was already stopped.")
                
        print("All processes stopped.")
        

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Multi-session training manager")
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    parser.add_argument("--rom", required=True, help="Path to Pokemon Pinball ROM")
    parser.add_argument("--max-processes", type=int, default=None, 
                       help="Maximum number of concurrent training processes")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print commands without executing them")
    args = parser.parse_args()
    
    # Setup the trainer
    trainer = MultiTrainer(
        config_file=args.config,
        rom_path=args.rom,
        max_processes=args.max_processes,
        dry_run=args.dry_run
    )
    
    # Handle graceful shutdown on SIGTERM
    def signal_handler(sig, frame):
        print("\nReceived termination signal.")
        trainer.stop_all()
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start training
    trainer.start_training()
    
    
if __name__ == "__main__":
    main()