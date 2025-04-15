#!/usr/bin/env python3
"""
Reattachment Monitor for Pokemon Pinball AI Training.

This script allows reconnecting to training processes started by multi_trainer.py
that are still running in the background after a disconnect.
"""

import os
import sys
import time
import psutil
import select
import signal
import termios
import tty
import fcntl
import subprocess
import threading
import queue
import re
from pathlib import Path

class ReattachMonitor:
    """Monitor and control running training processes."""
    
    def __init__(self):
        """Initialize the monitor."""
        self.processes = {}  # pid -> process object
        self.running = True
        
    def find_training_processes(self):
        """Find all running training processes."""
        found_count = 0
        
        # Look for Python processes running train.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline:
                    continue
                
                # Check if this is a train.py process
                if 'python' in cmdline[0].lower() and any('train.py' in arg for arg in cmdline):
                    pid = proc.info['pid']
                    process = psutil.Process(pid)
                    
                    # Create a subprocess.Popen-like object for compatibility
                    class ProcessWrapper:
                        def __init__(self, pid):
                            self.pid = pid
                        
                        def poll(self):
                            try:
                                proc = psutil.Process(self.pid)
                                if proc.is_running():
                                    return None  # Still running
                                return 0  # Assume success if not running anymore
                            except psutil.NoSuchProcess:
                                return 1  # Process doesn't exist
                        
                        def terminate(self):
                            try:
                                proc = psutil.Process(self.pid)
                                proc.terminate()
                            except psutil.NoSuchProcess:
                                pass
                        
                        def kill(self):
                            try:
                                proc = psutil.Process(self.pid)
                                proc.kill()
                            except psutil.NoSuchProcess:
                                pass
                    
                    self.processes[pid] = ProcessWrapper(pid)
                    found_count += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        return found_count
    
    def monitor_processes(self):
        """Monitor all discovered training processes."""
        # Save terminal state
        old_settings = termios.tcgetattr(sys.stdin)
        
        # Configure terminal for immediate character-by-character input
        tty.setcbreak(sys.stdin.fileno())
        
        # Make stdin non-blocking
        flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
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
                            exit_code = process.poll()
                            status = "completed successfully" if exit_code == 0 else f"failed with code {exit_code}"
                            batch_results['completed'].append((pid, status))
                            continue
                            
                        # Get model name from command line
                        try:
                            ps_output = subprocess.check_output(
                                ["ps", "-p", str(pid), "-o", "args="], 
                                universal_newlines=True
                            ).strip()
                            
                            # Extract model name from command line
                            cmd_parts = ps_output.split()
                            model_name = None
                            for i, part in enumerate(cmd_parts):
                                if part == "--model-name" and i+1 < len(cmd_parts):
                                    model_name = cmd_parts[i+1]
                                    break
                            
                            batch_results['configs'][pid] = model_name or "unknown"
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
                            
                            # Try to get progress from log file
                            model_name = batch_results['configs'].get(pid)
                            if model_name and model_name != "unknown":
                                log_path = os.path.join("checkpoints", model_name, "training.log")
                                if os.path.exists(log_path):
                                    try:
                                        # Get last few lines
                                        tail = subprocess.check_output(
                                            ["tail", "-n", "10", log_path], 
                                            universal_newlines=True
                                        )
                                        
                                        # Try to extract progress information
                                        timestep_match = re.search(r'timesteps?:?\s*(\d+)/(\d+)', tail)
                                        if timestep_match:
                                            current = int(timestep_match.group(1))
                                            total = int(timestep_match.group(2))
                                            batch_results['metrics'][pid]['progress'] = {
                                                'current': current,
                                                'total': total,
                                                'percent': (current / total) * 100 if total > 0 else 0,
                                                'type': 'timesteps'
                                            }
                                    except:
                                        pass  # Skip progress info if error
                            
                        except:
                            # Use default metrics
                            batch_results['metrics'][pid] = {'cpu': 0, 'memory': 0}
                    
                    # Put results in queue
                    metric_queue.put(batch_results)
                    
                    # Sleep to prevent CPU hogging
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
        
        try:
            # Variables for interface
            pid_to_config = {}
            pid_to_metrics = {}
            completed_processes = []
            current_view_pid = None
            input_buffer = ""
            command_result = None
            
            while self.processes and self.running:
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
                    display_buffer.append(f"Reattached to {len(running_pids)} running training process(es)")
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
                            progress_type = progress.get('type', 'steps')
                            display_buffer.append(f"  Progress: {bar} {percent:.1f}% ({progress['current']}/{progress['total']} {progress_type})")
                            
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
                        if config_name and config_name != "unknown":
                            log_path = os.path.join("checkpoints", config_name, "training.log")
                            if os.path.exists(log_path):
                                try:
                                    # Get log output - use grep to look for progress-related lines first
                                    try:
                                        # Get progress lines first
                                        progress_output = subprocess.check_output(
                                            ["grep", "-E", "(steps|timesteps|episode|reward|FPS)", log_path],
                                            shell=True,  # Need shell for pipe
                                            universal_newlines=True,
                                            stderr=subprocess.PIPE
                                        )
                                        # Take just the last few lines
                                        progress_lines = progress_output.strip().split('\n')
                                        if len(progress_lines) > 10:
                                            progress_output = '\n'.join(progress_lines[-10:])
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
            print("\nExiting monitor. Training processes will continue running.")
            
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
            process.terminate()
            
            # Give it some time to terminate
            for _ in range(10):  # 5 seconds max wait
                if process.poll() is not None:
                    break
                time.sleep(0.5)
                
            # If still running, force kill
            if process.poll() is None:
                process.kill()
                
            del self.processes[pid]
            print(f"Process {pid} stopped.")
            
        except Exception as e:
            print(f"Error killing process {pid}: {e}")
                
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
                
            except Exception as e:
                print(f"Error stopping process {pid}: {e}")
                
        print("All processes stopped.")

def main():
    """Main entry point for script."""
    print("Pokemon Pinball Training - Reattach Monitor")
    print("="*50)
    
    # Create the monitor
    monitor = ReattachMonitor()
    
    # Find running processes
    found = monitor.find_training_processes()
    
    if found == 0:
        print("No running training processes found.")
        print("If you believe processes are running, they might be:")
        print("1. Running under a different user")
        print("2. Not using 'train.py' in their command line")
        print("3. Already completed")
        return
    
    print(f"Found {found} running training process(es)!")
    print("Starting monitor interface...")
    print("Press 'q' to exit monitor (processes will continue running)")
    time.sleep(1)
    
    # Handle graceful shutdown on SIGTERM
    def signal_handler(sig, frame):
        print("\nReceived termination signal.")
        monitor.stop_all()
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    monitor.monitor_processes()
    
if __name__ == "__main__":
    main()