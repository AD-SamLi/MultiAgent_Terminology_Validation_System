#!/usr/bin/env python3
"""
🔧 FIXED DUAL-MODEL MONITOR
===========================

Real-time monitoring for the fixed dual-model runner.
"""

import psutil
import subprocess
import json
import os
import time
from datetime import datetime

def get_gpu_status():
    """Get GPU status using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            temp, mem_used, mem_total, util, power = result.stdout.strip().split(', ')
            return {
                'temperature': int(temp),
                'memory_used_gb': int(mem_used) / 1024,
                'memory_total_gb': int(mem_total) / 1024,
                'utilization': int(util),
                'power_draw': float(power)
            }
    except Exception as e:
        print(f"⚠️ Could not get GPU status: {e}")
    return None

def get_fixed_dual_process():
    """Get fixed dual-model process status"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'fixed_dual_model_runner.py' in cmdline:
                return {
                    'pid': proc.info['pid'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_gb': proc.info['memory_info'].rss / (1024 * 1024 * 1024),
                    'runtime_minutes': (time.time() - proc.info['create_time']) / 60,
                }
    except Exception as e:
        print(f"⚠️ Could not get process: {e}")
    return None

def get_latest_checkpoint():
    """Get latest fixed dual-model checkpoint"""
    try:
        checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
            
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) 
                          if f.startswith("fixed_dual_") and f.endswith("_checkpoint.json")]
        
        if not checkpoint_files:
            return None
            
        latest_file = max(checkpoint_files, 
                        key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
        filepath = os.path.join(checkpoints_dir, latest_file)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data['file_mtime'] = os.path.getmtime(filepath)
        return data
        
    except Exception as e:
        print(f"⚠️ Could not get checkpoint: {e}")
    return None

def main():
    print(f"🔧 FIXED DUAL-MODEL MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # GPU Status
    gpu_status = get_gpu_status()
    if gpu_status:
        print(f"🎮 GPU Status:")
        print(f"   • Temperature: {gpu_status['temperature']}°C")
        print(f"   • Memory: {gpu_status['memory_used_gb']:.1f}GB / {gpu_status['memory_total_gb']:.1f}GB")
        print(f"   • Utilization: {gpu_status['utilization']}%")
        print(f"   • Power: {gpu_status['power_draw']:.1f}W")
        
        # Estimate models loaded based on memory
        if gpu_status['memory_used_gb'] > 12:
            print(f"   🎮 Status: Both models likely loaded (~{gpu_status['memory_used_gb']:.1f}GB)")
        elif gpu_status['memory_used_gb'] > 6:
            print(f"   🎮 Status: First model loaded, loading second (~{gpu_status['memory_used_gb']:.1f}GB)")
        else:
            print(f"   🎮 Status: Loading first model (~{gpu_status['memory_used_gb']:.1f}GB)")
    else:
        print("❌ GPU status unavailable")
    
    # Process Status
    process = get_fixed_dual_process()
    if process:
        print(f"\n⚡ Fixed Dual-Model Process:")
        print(f"   • PID: {process['pid']}")
        print(f"   • Runtime: {process['runtime_minutes']:.1f} minutes")
        print(f"   • CPU: {process['cpu_percent']:.1f}%")
        print(f"   • RAM: {process['memory_gb']:.1f}GB")
    else:
        print("\n❌ Fixed dual-model process not running")
    
    # Checkpoint Status
    checkpoint = get_latest_checkpoint()
    if checkpoint:
        processed = checkpoint.get('processed_terms', 0)
        failed = checkpoint.get('failed_terms', 0)
        total = checkpoint.get('total_terms', 0)
        rate = checkpoint.get('processing_rate', 0)
        
        progress = (processed / total * 100) if total > 0 else 0
        eta_hours = (total - processed) / (rate * 3600) if rate > 0 else 0
        
        checkpoint_age = time.time() - checkpoint.get('file_mtime', time.time())
        
        print(f"\n📊 Progress:")
        print(f"   • Completed: {processed:,} / {total:,} terms ({progress:.1f}%)")
        print(f"   • Failed: {failed:,} terms")
        print(f"   • Rate: {rate:.3f} terms/sec")
        print(f"   • ETA: {eta_hours:.1f} hours")
        print(f"   • Last update: {checkpoint_age:.0f} seconds ago")
        
        # Configuration
        config = checkpoint.get('config', {})
        print(f"\n🔧 Configuration:")
        print(f"   • Model: {config.get('model_size', 'Unknown')}")
        print(f"   • GPU Workers: {config.get('gpu_workers', 0)}")
        print(f"   • CPU Workers: {config.get('cpu_workers', 0)}")
        print(f"   • Batch Size: {config.get('gpu_batch_size', 0)} per model")
        print(f"   • Load Delay: {config.get('model_load_delay', 0)}s")
    else:
        print("\n❌ No fixed dual-model checkpoint found yet")
    
    # System Resources
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"\n💻 System Resources:")
    print(f"   • CPU Usage: {cpu_percent:.1f}%")
    print(f"   • RAM Usage: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
