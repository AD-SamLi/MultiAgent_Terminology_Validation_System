#!/usr/bin/env python3
"""
System Hardware Detection and Optimization
Detects available hardware resources and provides optimal allocation strategies
"""

import psutil
import platform
import json
from typing import Dict, Any, Tuple
import subprocess
import os

class SystemHardwareDetector:
    """Detects and analyzes system hardware for optimal resource allocation"""
    
    def __init__(self):
        self.system_info = self._detect_system_info()
        self.cpu_info = self._detect_cpu_info()
        self.memory_info = self._detect_memory_info()
        self.gpu_info = self._detect_gpu_info()
        self.optimal_config = self._calculate_optimal_allocation()
    
    def _detect_system_info(self) -> Dict[str, Any]:
        """Detect basic system information"""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def _detect_cpu_info(self) -> Dict[str, Any]:
        """Detect CPU information and capabilities"""
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': 0,
            'min_frequency': 0,
            'current_frequency': 0,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'load_average': None
        }
        
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['max_frequency'] = cpu_freq.max
                cpu_info['min_frequency'] = cpu_freq.min  
                cpu_info['current_frequency'] = cpu_freq.current
        except:
            pass
        
        try:
            if hasattr(os, 'getloadavg'):
                cpu_info['load_average'] = os.getloadavg()
        except:
            pass
            
        return cpu_info
    
    def _detect_memory_info(self) -> Dict[str, Any]:
        """Detect memory information"""
        svmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_ram_gb': round(svmem.total / (1024**3), 2),
            'available_ram_gb': round(svmem.available / (1024**3), 2),
            'used_ram_gb': round(svmem.used / (1024**3), 2),
            'ram_percentage': svmem.percent,
            'total_swap_gb': round(swap.total / (1024**3), 2),
            'used_swap_gb': round(swap.used / (1024**3), 2),
            'swap_percentage': swap.percent
        }
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information using multiple methods"""
        gpu_info = {
            'nvidia_gpus': [],
            'total_gpus': 0,
            'gpu_memory_total': 0,
            'gpu_memory_available': 0,
            'cuda_available': False,
            'torch_available': False
        }
        
        # Check for NVIDIA GPUs using nvidia-ml-py
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info['total_gpus'] = gpu_count
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_data = {
                    'id': i,
                    'name': name,
                    'memory_total_gb': round(mem_info.total / (1024**3), 2),
                    'memory_used_gb': round(mem_info.used / (1024**3), 2),
                    'memory_free_gb': round(mem_info.free / (1024**3), 2),
                    'utilization': None
                }
                
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_data['utilization'] = util.gpu
                except:
                    pass
                
                gpu_info['nvidia_gpus'].append(gpu_data)
                gpu_info['gpu_memory_total'] += gpu_data['memory_total_gb']
                gpu_info['gpu_memory_available'] += gpu_data['memory_free_gb']
                
        except ImportError:
            # Try using nvidia-smi command
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_data = {
                                'id': i,
                                'name': parts[0],
                                'memory_total_gb': round(int(parts[1]) / 1024, 2),
                                'memory_used_gb': round(int(parts[2]) / 1024, 2),
                                'memory_free_gb': round(int(parts[3]) / 1024, 2),
                                'utilization': None
                            }
                            gpu_info['nvidia_gpus'].append(gpu_data)
                            gpu_info['total_gpus'] += 1
                            gpu_info['gpu_memory_total'] += gpu_data['memory_total_gb']
                            gpu_info['gpu_memory_available'] += gpu_data['memory_free_gb']
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        except Exception:
            pass
        
        # Check CUDA availability
        try:
            import torch
            gpu_info['torch_available'] = True
            gpu_info['cuda_available'] = torch.cuda.is_available()
            if gpu_info['cuda_available'] and gpu_info['total_gpus'] == 0:
                # Fallback: Use torch to detect GPUs
                gpu_info['total_gpus'] = torch.cuda.device_count()
        except ImportError:
            pass
        
        return gpu_info
    
    def _calculate_optimal_allocation(self) -> Dict[str, Any]:
        """Calculate optimal resource allocation based on detected hardware"""
        config = {
            'translation_strategy': 'cpu_only',
            'gpu_workers': 0,
            'cpu_workers': 4,
            'queue_workers': 2,
            'batch_size': 32,
            'memory_limit_gb': 4,
            'recommended_setup': 'Basic CPU processing'
        }
        
        # Determine optimal strategy based on available hardware
        total_cores = self.cpu_info['logical_cores']
        available_ram = self.memory_info['available_ram_gb']
        gpu_count = self.gpu_info['total_gpus']
        gpu_memory = self.gpu_info['gpu_memory_available']
        
        if gpu_count > 0 and gpu_memory > 2.0 and self.gpu_info['cuda_available']:
            # GPU-accelerated strategy
            if gpu_count >= 2 and gpu_memory >= 8.0:
                # Dual-GPU setup
                config.update({
                    'translation_strategy': 'dual_gpu_hybrid',
                    'gpu_workers': 2,
                    'cpu_workers': max(4, total_cores // 2),
                    'queue_workers': max(2, total_cores // 4),
                    'batch_size': 64,
                    'memory_limit_gb': min(8, available_ram * 0.6),
                    'recommended_setup': 'Dual-GPU hybrid processing with CPU queue management'
                })
            else:
                # Single GPU setup
                config.update({
                    'translation_strategy': 'single_gpu_hybrid',
                    'gpu_workers': 1,
                    'cpu_workers': max(4, total_cores // 2),
                    'queue_workers': max(2, total_cores // 4),
                    'batch_size': 48,
                    'memory_limit_gb': min(6, available_ram * 0.5),
                    'recommended_setup': '1 GPU translation + CPU workers for queueing and preprocessing'
                })
        else:
            # CPU-only strategy
            if total_cores >= 16 and available_ram >= 16:
                # High-end CPU setup
                config.update({
                    'translation_strategy': 'high_performance_cpu',
                    'gpu_workers': 0,
                    'cpu_workers': max(8, total_cores // 2),
                    'queue_workers': max(4, total_cores // 4),
                    'batch_size': 32,
                    'memory_limit_gb': min(8, available_ram * 0.5),
                    'recommended_setup': 'High-performance CPU parallel processing'
                })
            elif total_cores >= 8:
                # Mid-range CPU setup
                config.update({
                    'translation_strategy': 'standard_cpu',
                    'cpu_workers': max(4, total_cores // 2),
                    'queue_workers': max(2, total_cores // 4),
                    'batch_size': 24,
                    'memory_limit_gb': min(4, available_ram * 0.4),
                    'recommended_setup': 'Standard CPU processing with parallel workers'
                })
        
        # Adjust based on available memory
        if available_ram < 8:
            config['batch_size'] = max(16, config['batch_size'] // 2)
            config['memory_limit_gb'] = min(config['memory_limit_gb'], available_ram * 0.3)
        
        return config
    
    def get_system_report(self) -> str:
        """Generate a comprehensive system report"""
        report = f"""
🖥️  SYSTEM HARDWARE ANALYSIS REPORT
{'=' * 50}

💻 System Information:
   Platform: {self.system_info['platform']} {self.system_info['platform_release']}
   Architecture: {self.system_info['architecture']}
   Processor: {self.system_info['processor']}

⚙️  CPU Information:
   Physical Cores: {self.cpu_info['physical_cores']}
   Logical Cores: {self.cpu_info['logical_cores']}
   Current Usage: {self.cpu_info['cpu_percent']}%
   Max Frequency: {self.cpu_info['max_frequency']:.0f} MHz

💾 Memory Information:
   Total RAM: {self.memory_info['total_ram_gb']} GB
   Available RAM: {self.memory_info['available_ram_gb']} GB ({100-self.memory_info['ram_percentage']:.1f}% free)
   Used RAM: {self.memory_info['used_ram_gb']} GB ({self.memory_info['ram_percentage']:.1f}%)

🎮 GPU Information:
   Total GPUs: {self.gpu_info['total_gpus']}
   CUDA Available: {self.gpu_info['cuda_available']}
   Total GPU Memory: {self.gpu_info['gpu_memory_total']} GB
   Available GPU Memory: {self.gpu_info['gpu_memory_available']} GB
"""
        
        if self.gpu_info['nvidia_gpus']:
            report += "\n   NVIDIA GPUs:\n"
            for gpu in self.gpu_info['nvidia_gpus']:
                util_str = f" ({gpu['utilization']}% util)" if gpu['utilization'] is not None else ""
                report += f"   - GPU {gpu['id']}: {gpu['name']} - {gpu['memory_free_gb']:.1f}GB free / {gpu['memory_total_gb']:.1f}GB total{util_str}\n"
        
        report += f"""
🚀 OPTIMAL TRANSLATION CONFIGURATION:
{'=' * 50}
   Strategy: {self.optimal_config['translation_strategy']}
   Setup: {self.optimal_config['recommended_setup']}
   
   Resource Allocation:
   - GPU Workers: {self.optimal_config['gpu_workers']}
   - CPU Workers: {self.optimal_config['cpu_workers']}
   - Queue Workers: {self.optimal_config['queue_workers']}
   - Batch Size: {self.optimal_config['batch_size']}
   - Memory Limit: {self.optimal_config['memory_limit_gb']} GB
"""
        return report
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get the optimal configuration dictionary"""
        return self.optimal_config
    
    def save_report(self, filename: str = "system_hardware_report.json"):
        """Save the complete hardware analysis to a JSON file"""
        full_report = {
            'system_info': self.system_info,
            'cpu_info': self.cpu_info,
            'memory_info': self.memory_info,
            'gpu_info': self.gpu_info,
            'optimal_config': self.optimal_config,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        return filename

def main():
    """Main function to run hardware detection"""
    detector = SystemHardwareDetector()
    
    # Print system report
    print(detector.get_system_report())
    
    # Save detailed report
    report_file = detector.save_report()
    print(f"\n📁 Detailed report saved to: {report_file}")
    
    return detector.get_optimal_config()

if __name__ == "__main__":
    main()

