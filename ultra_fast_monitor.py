#!/usr/bin/env python3
"""
🚀 ULTRA-FAST MONITOR - Real-time Performance Tracking
====================================================

Monitors the ultra-fast runner with advanced metrics:
- GPU utilization and memory usage
- CPU core-level utilization
- Processing throughput and ETA
- Memory usage and optimization
- Real-time performance alerts
"""

import psutil
import subprocess
import json
import time
import os
import argparse
from datetime import datetime

class UltraFastMonitor:
    """Real-time monitor for ultra-fast processing"""
    
    def __init__(self):
        """Initialize monitor"""
        self.start_time = time.time()
        self.last_check = 0
        
    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split('\n')[0]
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) >= 7:
                    name = parts[0]
                    memory_used = float(parts[1]) if parts[1] != 'N/A' else 0
                    memory_total = float(parts[2]) if parts[2] != 'N/A' else 0
                    gpu_util = float(parts[3]) if parts[3] != 'N/A' else 0
                    temp = float(parts[4]) if parts[4] != 'N/A' else 0
                    power_draw = float(parts[5]) if parts[5] != 'N/A' else 0
                    power_limit = float(parts[6]) if parts[6] != 'N/A' else 0
                    
                    memory_free = memory_total - memory_used
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    power_percent = (power_draw / power_limit * 100) if power_limit > 0 else 0
                    
                    return {
                        'name': name,
                        'memory_used_gb': memory_used / 1024,
                        'memory_total_gb': memory_total / 1024,
                        'memory_free_gb': memory_free / 1024,
                        'memory_percent': memory_percent,
                        'gpu_utilization': gpu_util,
                        'temperature': temp,
                        'power_draw': power_draw,
                        'power_percent': power_percent
                    }
        except Exception as e:
            pass
            
        return {
            'name': 'Unknown',
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'memory_free_gb': 0,
            'memory_percent': 0,
            'gpu_utilization': 0,
            'temperature': 0,
            'power_draw': 0,
            'power_percent': 0
        }
    
    def find_ultra_fast_process(self):
        """Find running ultra-fast runner process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'ultra_fast_runner.py' in cmdline:
                    runtime = time.time() - proc.info['create_time']
                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                    
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': memory_mb,
                        'runtime': runtime,
                        'status': 'ACTIVE'
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return None
    
    def get_checkpoint_data(self):
        """Get latest checkpoint data"""
        try:
            checkpoint_dir = '/home/samli/Documents/Python/Term_Verify/checkpoints'
            if not os.path.exists(checkpoint_dir):
                return None
            
            # Find latest ultra-fast checkpoint
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ultra_fast_') and f.endswith('_checkpoint.json')]
            if not checkpoint_files:
                return None
            
            latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
            
            with open(os.path.join(checkpoint_dir, latest_file), 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def get_cpu_core_utilization(self):
        """Get per-core CPU utilization"""
        try:
            return psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception:
            return []
    
    def format_cpu_cores(self, core_utils):
        """Format CPU core utilization display"""
        if not core_utils:
            return "No CPU data"
        
        lines = []
        cores_per_line = 4
        
        for i in range(0, len(core_utils), cores_per_line):
            line_cores = core_utils[i:i + cores_per_line]
            line_parts = []
            
            for j, util in enumerate(line_cores):
                core_num = i + j
                if util >= 80:
                    emoji = "🔥"
                elif util >= 60:
                    emoji = "⚡"
                elif util >= 40:
                    emoji = "🔧"
                else:
                    emoji = "💤"
                
                line_parts.append(f"C{core_num:2d}:{emoji}{util:5.1f}%")
            
            lines.append("   ".join(line_parts))
        
        return "\n".join(lines)
    
    def display_status(self, once=False):
        """Display real-time status"""
        try:
            while True:
                # Clear screen for better readability
                if not once:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🚀 ULTRA-FAST REAL-TIME MONITOR")
                print("=" * 80)
                print(f"⏰ Monitor Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # Process status
                process_info = self.find_ultra_fast_process()
                if process_info:
                    runtime_str = f"{int(process_info['runtime']//3600):02d}:{int((process_info['runtime']%3600)//60):02d}:{int(process_info['runtime']%60):02d}"
                    memory_gb = process_info['memory_mb'] / 1024
                    
                    print("✅ ULTRA-FAST RUNNER: ACTIVE")
                    print(f"   • PID: {process_info['pid']}")
                    print(f"   • Runtime: {runtime_str}")
                    print(f"   • Process CPU: {process_info['cpu_percent']:.1f}%")
                    print(f"   • Process Memory: {process_info['memory_mb']:.1f}MB ({memory_gb:.1f}GB)")
                else:
                    print("❌ ULTRA-FAST RUNNER: NOT RUNNING")
                
                print()
                
                # System utilization
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                gpu_info = self.get_gpu_info()
                
                print("🖥️  SYSTEM UTILIZATION (REAL-TIME):")
                print(f"   • CPU Average: {cpu_percent:.1f}% (Target: >80%)")
                print(f"   • CPU Maximum: {max(self.get_cpu_core_utilization()):.1f}%")
                print(f"   • Active Cores: {psutil.cpu_count()}/{psutil.cpu_count()} cores")
                print(f"   • RAM Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
                print(f"   • GPU Usage: {gpu_info['gpu_utilization']:.1f}% ({gpu_info['memory_used_gb']:.1f}GB/{gpu_info['memory_total_gb']:.1f}GB)")
                print(f"   • GPU Free: {gpu_info['memory_free_gb']:.1f}GB")
                
                print()
                
                # CPU cores detailed view
                core_utils = self.get_cpu_core_utilization()
                print("💻 CPU CORES UTILIZATION:")
                print(self.format_cpu_cores(core_utils))
                print()
                
                # GPU detailed info
                print("🎮 GPU DETAILED STATUS:")
                print(f"   • Model: {gpu_info['name']}")
                print(f"   • Utilization: {gpu_info['gpu_utilization']:.1f}%")
                print(f"   • Memory: {gpu_info['memory_used_gb']:.1f}GB / {gpu_info['memory_total_gb']:.1f}GB ({gpu_info['memory_percent']:.1f}%)")
                print(f"   • Temperature: {gpu_info['temperature']:.0f}°C")
                print(f"   • Power: {gpu_info['power_draw']:.0f}W / {gpu_info['power_percent']:.1f}%")
                print()
                
                # Processing progress
                checkpoint_data = self.get_checkpoint_data()
                if checkpoint_data:
                    processed = checkpoint_data.get('processed_terms', 0)
                    failed = checkpoint_data.get('failed_terms', 0)
                    total = checkpoint_data.get('total_terms', 0)
                    rate = checkpoint_data.get('processing_rate', 0)
                    
                    if total > 0:
                        progress_pct = ((processed + failed) / total) * 100
                        remaining = total - processed - failed
                        eta_hours = remaining / max(rate, 0.001) / 3600 if rate > 0 else 0
                    else:
                        progress_pct = 0
                        eta_hours = 0
                    
                    print("📊 PROCESSING PROGRESS:")
                    print(f"   • Completed: {processed + failed:,}/{total:,} ({progress_pct:.1f}%)")
                    print(f"   • Success: {processed:,} | Failed: {failed:,}")
                    print(f"   • Rate: {rate:.2f} terms/sec")
                    print(f"   • ETA: {eta_hours:.1f} hours")
                else:
                    print("📊 PROCESSING PROGRESS: No checkpoint data available")
                
                print()
                
                # Performance analysis
                performance_status = []
                
                if cpu_percent < 70:
                    performance_status.append("🔧 CPU can be pushed higher")
                elif cpu_percent > 90:
                    performance_status.append("🔥 CPU at maximum utilization")
                else:
                    performance_status.append("⚡ CPU well utilized")
                
                if gpu_info['gpu_utilization'] < 70:
                    performance_status.append("🎮 GPU can be utilized more")
                elif gpu_info['gpu_utilization'] > 90:
                    performance_status.append("🔥 GPU at maximum utilization")
                else:
                    performance_status.append("⚡ GPU well utilized")
                
                if memory.percent > 80:
                    performance_status.append("⚠️  High memory usage")
                
                if gpu_info['memory_percent'] > 90:
                    performance_status.append("⚠️  GPU memory almost full")
                
                print("💡 ULTRA-FAST STATUS:")
                for status in performance_status:
                    print(f"   {status}")
                
                print()
                print("=" * 80)
                print("🚀 SYSTEM RUNNING AT ULTRA-FAST MODE")
                
                if once:
                    break
                
                print("Press Ctrl+C to stop monitoring")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n👋 Ultra-fast monitoring stopped")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast Runner Monitor')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    monitor = UltraFastMonitor()
    monitor.display_status(once=args.once)

if __name__ == "__main__":
    main()
