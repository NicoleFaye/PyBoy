#!/usr/bin/env python3
"""
Utility to set CPU affinity for training processes.
This helps distribute load across cores when running multiple training instances.
"""
import argparse
import os
import psutil
import time
import subprocess
import sys


def list_training_processes():
    """List all Python processes running train.py"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process running train.py
            if proc.info['name'] == 'python' and any('train.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes


def set_process_affinity(pid, cpu_list):
    """Set CPU affinity for a process"""
    try:
        proc = psutil.Process(pid)
        proc.cpu_affinity(cpu_list)
        return True
    except Exception as e:
        print(f"Error setting affinity for process {pid}: {e}")
        return False


def assign_cpus_to_processes(processes, cpu_distribution=None):
    """
    Assign CPUs to processes based on a distribution strategy.
    
    Args:
        processes: List of process objects
        cpu_distribution: Optional dictionary mapping process PIDs to CPU lists
    """
    # Get available CPUs
    all_cpus = list(range(psutil.cpu_count(logical=True)))
    
    # If no distribution provided, automatically assign CPUs
    if not cpu_distribution:
        cpu_distribution = {}
        # Determine how to distribute CPUs
        if len(processes) <= len(all_cpus):
            # We have enough CPUs to dedicate some to each process
            cpus_per_process = len(all_cpus) // len(processes)
            
            for i, proc in enumerate(processes):
                # Assign a chunk of CPUs to each process
                start_idx = i * cpus_per_process
                end_idx = start_idx + cpus_per_process if i < len(processes) - 1 else len(all_cpus)
                cpu_distribution[proc.pid] = all_cpus[start_idx:end_idx]
        else:
            # More processes than CPUs, try to evenly distribute
            for i, proc in enumerate(processes):
                # Assign one CPU to each process, cycling through available CPUs
                cpu_idx = i % len(all_cpus)
                cpu_distribution[proc.pid] = [all_cpus[cpu_idx]]
    
    # Apply the CPU distribution
    for proc in processes:
        if proc.pid in cpu_distribution:
            cpu_list = cpu_distribution[proc.pid]
            success = set_process_affinity(proc.pid, cpu_list)
            if success:
                print(f"Process {proc.pid} assigned to CPUs: {cpu_list}")
                cmd_line = " ".join(proc.cmdline())
                print(f"  Command: {cmd_line[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Set CPU affinity for training processes")
    parser.add_argument("--list", action="store_true", help="List training processes")
    parser.add_argument("--set", action="store_true", help="Set CPU affinity for training processes")
    parser.add_argument("--pid", type=int, nargs="+", help="Specific process ID(s) to set affinity for")
    parser.add_argument("--cpu", type=int, nargs="+", help="CPU IDs to assign (for specific pid)")
    parser.add_argument("--monitor", action="store_true", help="Monitor and set affinity for new processes")
    args = parser.parse_args()
    
    if args.list:
        processes = list_training_processes()
        if not processes:
            print("No training processes found")
            return
            
        print(f"Found {len(processes)} training processes:")
        for proc in processes:
            try:
                cpu_affinity = proc.cpu_affinity()
                memory_info = proc.memory_info()
                cpu_percent = proc.cpu_percent(interval=0.1)
                cmd_line = " ".join(proc.cmdline())
                
                print(f"PID: {proc.pid}")
                print(f"  CPU Affinity: {cpu_affinity}")
                print(f"  Memory: {memory_info.rss / (1024 * 1024):.1f} MB")
                print(f"  CPU Usage: {cpu_percent:.1f}%")
                print(f"  Command: {cmd_line[:80]}...")
                print()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"PID: {proc.pid} - Process gone or no access")
    
    elif args.set:
        if args.pid and args.cpu:
            # Set affinity for specific processes
            for pid in args.pid:
                success = set_process_affinity(pid, args.cpu)
                if success:
                    print(f"Set CPU affinity for process {pid} to {args.cpu}")
        else:
            # Auto-distribute all training processes
            processes = list_training_processes()
            if not processes:
                print("No training processes found")
                return
                
            assign_cpus_to_processes(processes)
    
    elif args.monitor:
        print("Monitoring for new training processes. Press Ctrl+C to stop.")
        known_pids = set()
        
        try:
            while True:
                processes = list_training_processes()
                current_pids = {proc.pid for proc in processes}
                
                # Find new processes
                new_pids = current_pids - known_pids
                if new_pids:
                    new_processes = [proc for proc in processes if proc.pid in new_pids]
                    print(f"Found {len(new_processes)} new training processes")
                    assign_cpus_to_processes(new_processes)
                    known_pids.update(new_pids)
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("Monitoring stopped")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()