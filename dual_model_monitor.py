#!/usr/bin/env python3
"""
🎮 DUAL-MODEL MONITOR
====================

Real-time monitoring for dual-model ultra-fast runner.
Tracks both GPU workers, queue status, and performance metrics.
"""

import psutil
import time
import subprocess
import json
import os
import sys
from datetime import datetime

class DualModelMonitor:
    def __init__(self):
        self.checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        
    def get_gpu_status(self):
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
                    'memory_used_mb': int(mem_used),
                    'memory_total_mb': int(mem_total),
                    'memory_used_gb': int(mem_used) / 1024,
                    'memory_total_gb': int(mem_total) / 1024,
                    'utilization': int(util),
                    'power_draw': float(power)
                }
        except Exception as e:
            print(f"⚠️ Could not get GPU status: {e}")
        return None
    
    def get_dual_model_process(self):
        """Get dual-model runner process status"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'dual_model_ultra_fast_runner.py' in cmdline:
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / (1024 * 1024),
                        'memory_gb': proc.info['memory_info'].rss / (1024 * 1024 * 1024),
                        'runtime_minutes': (time.time() - proc.info['create_time']) / 60,
                        'cmdline': cmdline
                    }
        except Exception as e:
            print(f"⚠️ Could not get dual-model process: {e}")
        return None
    
    def get_latest_dual_model_checkpoint(self):
        """Get latest dual-model checkpoint data"""
        try:
            if not os.path.exists(self.checkpoints_dir):
                return None
                
            checkpoint_files = [f for f in os.listdir(self.checkpoints_dir) 
                              if f.startswith("dual_model_") and f.endswith("_checkpoint.json")]
            
            if not checkpoint_files:
                return None
                
            # Find most recent checkpoint
            latest_file = max(checkpoint_files, 
                            key=lambda f: os.path.getmtime(os.path.join(self.checkpoints_dir, f)))
            filepath = os.path.join(self.checkpoints_dir, latest_file)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Add file modification time
            data['file_mtime'] = os.path.getmtime(filepath)
            data['checkpoint_file'] = latest_file
            
            return data
            
        except Exception as e:
            print(f"⚠️ Could not get dual-model checkpoint: {e}")
        return None
    
    def display_status(self, once=False):
        """Display comprehensive dual-model status"""
        if once:
            self._display_single_status()
        else:
            self._display_continuous_status()
    
    def _display_single_status(self):
        """Display single status check"""
        print(f"🎮 DUAL-MODEL MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # GPU Status
        gpu_status = self.get_gpu_status()
        if gpu_status:
            print(f"🎮 GPU Status:")
            print(f"   • Temperature: {gpu_status['temperature']}°C")
            print(f"   • Memory: {gpu_status['memory_used_gb']:.1f}GB / {gpu_status['memory_total_gb']:.1f}GB ({gpu_status['memory_used_gb']/gpu_status['memory_total_gb']*100:.1f}%)")
            print(f"   • Utilization: {gpu_status['utilization']}%")
            print(f"   • Power: {gpu_status['power_draw']:.1f}W")
        else:
            print("❌ GPU status unavailable")
        
        # Process Status
        process = self.get_dual_model_process()
        if process:
            print(f"\n⚡ Dual-Model Process:")
            print(f"   • PID: {process['pid']}")
            print(f"   • Runtime: {process['runtime_minutes']:.1f} minutes")
            print(f"   • CPU: {process['cpu_percent']:.1f}%")
            print(f"   • RAM: {process['memory_gb']:.1f}GB")
        else:
            print("\n❌ Dual-model process not running")
        
        # Checkpoint Status
        checkpoint = self.get_latest_dual_model_checkpoint()
        if checkpoint:
            processed = checkpoint.get('processed_terms', 0)
            failed = checkpoint.get('failed_terms', 0)
            total = checkpoint.get('total_terms', 0)
            rate = checkpoint.get('processing_rate', 0)
            
            progress = (processed / total * 100) if total > 0 else 0
            eta_hours = (total - processed) / (rate * 3600) if rate > 0 else 0
            
            # Time since last checkpoint
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
            print(f"   • GPU Workers: {config.get('gpu_workers', 0)} (dual models)")
            print(f"   • CPU Workers: {config.get('cpu_workers', 0)}")
            print(f"   • Batch Size: {config.get('gpu_batch_size', 0)} per model")
        else:
            print("\n❌ No dual-model checkpoint found")
        
        # System Resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"\n💻 System Resources:")
        print(f"   • CPU Usage: {cpu_percent:.1f}%")
        print(f"   • RAM Usage: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        
        print("=" * 80)
    
    def _display_continuous_status(self):
        """Display continuous monitoring"""
        print("🎮 DUAL-MODEL CONTINUOUS MONITOR")
        print("Press Ctrl+C to stop...")
        print("=" * 80)
        
        try:
            while True:
                # Clear screen (optional)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self._display_single_status()
                
                # Wait before next update
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def check_dual_model_health(self):
        """Check dual-model system health and return alerts"""
        alerts = []
        critical_alerts = []
        
        # Check GPU
        gpu_status = self.get_gpu_status()
        if gpu_status:
            if gpu_status['temperature'] >= 70:
                critical_alerts.append(f"🚨 GPU temperature critical: {gpu_status['temperature']}°C")
            elif gpu_status['temperature'] >= 65:
                alerts.append(f"⚠️ GPU temperature high: {gpu_status['temperature']}°C")
            
            memory_percent = gpu_status['memory_used_gb'] / gpu_status['memory_total_gb'] * 100
            if memory_percent >= 90:
                critical_alerts.append(f"🚨 GPU memory critical: {memory_percent:.1f}%")
            elif memory_percent >= 80:
                alerts.append(f"⚠️ GPU memory high: {memory_percent:.1f}%")
        else:
            critical_alerts.append("🚨 Cannot access GPU status")
        
        # Check process
        process = self.get_dual_model_process()
        if not process:
            critical_alerts.append("🚨 Dual-model process not running")
        elif process['memory_gb'] > 10:  # More than 10GB RAM
            alerts.append(f"⚠️ High RAM usage: {process['memory_gb']:.1f}GB")
        
        # Check checkpoint freshness
        checkpoint = self.get_latest_dual_model_checkpoint()
        if checkpoint:
            checkpoint_age = time.time() - checkpoint.get('file_mtime', time.time())
            if checkpoint_age > 300:  # 5 minutes
                critical_alerts.append(f"🚨 No progress for {checkpoint_age/60:.1f} minutes")
            elif checkpoint_age > 120:  # 2 minutes
                alerts.append(f"⚠️ No progress for {checkpoint_age/60:.1f} minutes")
        else:
            alerts.append("⚠️ No dual-model checkpoint found")
        
        return alerts, critical_alerts

def main():
    monitor = DualModelMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--once":
            monitor.display_status(once=True)
        elif command == "--health":
            alerts, critical = monitor.check_dual_model_health()
            
            print("🛡️ DUAL-MODEL HEALTH CHECK")
            print("=" * 40)
            
            if critical:
                print("🚨 CRITICAL ALERTS:")
                for alert in critical:
                    print(f"   {alert}")
                print()
            
            if alerts:
                print("⚠️ WARNINGS:")
                for alert in alerts:
                    print(f"   {alert}")
                print()
            
            if not alerts and not critical:
                print("✅ All systems healthy!")
            
            # Exit with error code if critical issues
            if critical:
                sys.exit(1)
        
        elif command == "--continuous":
            monitor.display_status(once=False)
        
        else:
            print("Usage:")
            print("  python dual_model_monitor.py --once      # Single status check")
            print("  python dual_model_monitor.py --health    # Health check only")
            print("  python dual_model_monitor.py --continuous # Continuous monitoring")
            sys.exit(1)
    else:
        # Default: single status check
        monitor.display_status(once=True)

if __name__ == "__main__":
    main()
