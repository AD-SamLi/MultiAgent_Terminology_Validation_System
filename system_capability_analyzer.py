#!/usr/bin/env python3
"""
System Capability Analyzer
Analyzes current PC hardware capabilities for multi-GPU translation processing
"""

import os
import sys
import json
import psutil
import platform
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class SystemCapabilityAnalyzer:
    """Analyze system capabilities for multi-GPU translation processing"""
    
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.memory_info = {}
        self.cpu_info = {}
        self.recommendations = {}
        
    def analyze_full_system(self) -> Dict:
        """Perform comprehensive system analysis"""
        print("🔍 Analyzing system capabilities...")
        
        self._analyze_cpu()
        self._analyze_memory()
        self._analyze_gpu()
        self._analyze_storage()
        self._analyze_network()
        self._analyze_python_environment()
        self._generate_recommendations()
        
        return {
            "system_info": self.system_info,
            "cpu_info": self.cpu_info,
            "memory_info": self.memory_info,
            "gpu_info": self.gpu_info,
            "recommendations": self.recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_cpu(self):
        """Analyze CPU capabilities"""
        print("🖥️  Analyzing CPU...")
        
        self.cpu_info = {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "architecture": platform.architecture(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
            "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else "N/A"
        }
        
        # Get more detailed CPU info on Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'model name' in cpuinfo:
                    model_line = [line for line in cpuinfo.split('\n') if 'model name' in line][0]
                    self.cpu_info['model_name'] = model_line.split(':')[1].strip()
        except:
            pass
        
        print(f"   • CPU: {self.cpu_info.get('model_name', 'Unknown')}")
        print(f"   • Cores: {self.cpu_info['physical_cores']} physical, {self.cpu_info['logical_cores']} logical")
        print(f"   • Current usage: {self.cpu_info['cpu_percent']:.1f}%")
    
    def _analyze_memory(self):
        """Analyze system memory"""
        print("🧠 Analyzing memory...")
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        self.memory_info = {
            "total_ram_gb": memory.total / (1024**3),
            "available_ram_gb": memory.available / (1024**3),
            "used_ram_gb": memory.used / (1024**3),
            "ram_percent": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent
        }
        
        print(f"   • Total RAM: {self.memory_info['total_ram_gb']:.1f} GB")
        print(f"   • Available RAM: {self.memory_info['available_ram_gb']:.1f} GB")
        print(f"   • RAM Usage: {self.memory_info['ram_percent']:.1f}%")
        print(f"   • Swap: {self.memory_info['swap_total_gb']:.1f} GB")
    
    def _analyze_gpu(self):
        """Analyze GPU capabilities"""
        print("🎮 Analyzing GPU...")
        
        self.gpu_info = {
            "torch_available": TORCH_AVAILABLE,
            "cuda_available": False,
            "gpus": [],
            "total_gpu_memory": 0,
            "cuda_version": None,
            "driver_version": None
        }
        
        if TORCH_AVAILABLE:
            self.gpu_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                self.gpu_info["cuda_version"] = torch.version.cuda
                self.gpu_info["gpu_count"] = torch.cuda.device_count()
                
                print(f"   • CUDA Available: Yes (version {self.gpu_info['cuda_version']})")
                print(f"   • GPU Count: {self.gpu_info['gpu_count']}")
                
                # Analyze each GPU
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    
                    gpu_info = {
                        "id": i,
                        "name": gpu_props.name,
                        "total_memory_gb": gpu_props.total_memory / (1024**3),
                        "major": gpu_props.major,
                        "minor": gpu_props.minor,
                        "multi_processor_count": gpu_props.multi_processor_count,
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                    }
                    
                    # Get current memory usage
                    try:
                        torch.cuda.set_device(i)
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        gpu_info.update({
                            "allocated_memory_gb": allocated,
                            "reserved_memory_gb": reserved,
                            "free_memory_gb": gpu_info["total_memory_gb"] - reserved
                        })
                    except Exception as e:
                        gpu_info.update({
                            "allocated_memory_gb": 0,
                            "reserved_memory_gb": 0,
                            "free_memory_gb": gpu_info["total_memory_gb"],
                            "memory_error": str(e)
                        })
                    
                    self.gpu_info["gpus"].append(gpu_info)
                    self.gpu_info["total_gpu_memory"] += gpu_info["total_memory_gb"]
                    
                    print(f"   • GPU {i}: {gpu_info['name']}")
                    print(f"     - Memory: {gpu_info['total_memory_gb']:.1f} GB total, {gpu_info['free_memory_gb']:.1f} GB free")
                    print(f"     - Compute: {gpu_info['compute_capability']}")
            else:
                print("   • CUDA Available: No")
        else:
            print("   • PyTorch not available")
        
        # Try NVIDIA-ML for additional GPU info
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                driver_version = nvml.nvmlSystemGetDriverVersion()
                self.gpu_info["driver_version"] = driver_version.decode('utf-8')
                print(f"   • Driver Version: {self.gpu_info['driver_version']}")
                
                # Get additional GPU details
                for i, gpu in enumerate(self.gpu_info.get("gpus", [])):
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        gpu.update({
                            "temperature_c": temp,
                            "power_usage_w": power,
                            "gpu_utilization_percent": utilization.gpu,
                            "memory_utilization_percent": utilization.memory
                        })
                        
                        print(f"     - Temperature: {temp}°C, Power: {power:.1f}W")
                        print(f"     - Utilization: GPU {utilization.gpu}%, Memory {utilization.memory}%")
                        
                    except Exception as e:
                        print(f"     - Extended info error: {e}")
                        
            except Exception as e:
                print(f"   • NVIDIA-ML error: {e}")
    
    def _analyze_storage(self):
        """Analyze storage capabilities"""
        print("💾 Analyzing storage...")
        
        self.system_info["storage"] = []
        
        # Get disk usage for all mounted filesystems
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                
                storage_info = {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent_used": (usage.used / usage.total) * 100
                }
                
                self.system_info["storage"].append(storage_info)
                
                if partition.mountpoint == '/':  # Root filesystem on Linux
                    print(f"   • Root: {storage_info['free_gb']:.1f} GB free ({storage_info['total_gb']:.1f} GB total)")
                
            except PermissionError:
                continue
    
    def _analyze_network(self):
        """Analyze network capabilities"""
        print("🌐 Analyzing network...")
        
        # Get network interfaces
        network_info = {}
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == 2:  # IPv4
                    network_info[interface] = addr.address
        
        self.system_info["network_interfaces"] = network_info
        
        # Get network I/O stats
        net_io = psutil.net_io_counters()
        self.system_info["network_io"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        print(f"   • Active interfaces: {len(network_info)}")
    
    def _analyze_python_environment(self):
        """Analyze Python environment and dependencies"""
        print("🐍 Analyzing Python environment...")
        
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "path": sys.path[:3]  # First 3 paths
        }
        
        # Check key dependencies
        dependencies = {}
        
        key_packages = [
            'torch', 'transformers', 'accelerate', 'sentencepiece', 
            'psutil', 'numpy', 'scipy', 'sklearn'
        ]
        
        for package in key_packages:
            try:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    dependencies[package] = module.__version__
                else:
                    dependencies[package] = "installed"
            except ImportError:
                dependencies[package] = "not_installed"
        
        self.system_info["python"] = python_info
        self.system_info["dependencies"] = dependencies
        
        print(f"   • Python: {python_info['version'].split()[0]}")
        print(f"   • PyTorch: {dependencies.get('torch', 'Not installed')}")
        print(f"   • Transformers: {dependencies.get('transformers', 'Not installed')}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on system analysis"""
        print("💡 Generating recommendations...")
        
        recommendations = {
            "overall_capability": "unknown",
            "recommended_config": {},
            "warnings": [],
            "optimizations": [],
            "hardware_upgrades": []
        }
        
        # Analyze GPU capability
        gpu_capable = False
        total_gpu_memory = self.gpu_info.get("total_gpu_memory", 0)
        gpu_count = len(self.gpu_info.get("gpus", []))
        
        if self.gpu_info.get("cuda_available", False) and total_gpu_memory > 0:
            gpu_capable = True
            
            if total_gpu_memory >= 16:
                recommendations["overall_capability"] = "excellent"
                recommendations["recommended_config"] = {
                    "model_size": "3.3B" if total_gpu_memory >= 24 else "1.3B",
                    "gpu_workers": min(gpu_count * 2, 4),
                    "instances_per_gpu": 2 if total_gpu_memory >= 16 else 1,
                    "batch_size": "large",
                    "cpu_workers": min(self.cpu_info.get("logical_cores", 4), 8)
                }
                recommendations["optimizations"].append("Enable large batch processing for maximum throughput")
                
            elif total_gpu_memory >= 8:
                recommendations["overall_capability"] = "good"
                recommendations["recommended_config"] = {
                    "model_size": "1.3B",
                    "gpu_workers": gpu_count,
                    "instances_per_gpu": 1,
                    "batch_size": "medium",
                    "cpu_workers": min(self.cpu_info.get("logical_cores", 4), 6)
                }
                recommendations["optimizations"].append("Use medium batch sizes to balance memory and performance")
                
            elif total_gpu_memory >= 4:
                recommendations["overall_capability"] = "limited"
                recommendations["recommended_config"] = {
                    "model_size": "small",
                    "gpu_workers": 1,
                    "instances_per_gpu": 1,
                    "batch_size": "small",
                    "cpu_workers": min(self.cpu_info.get("logical_cores", 4), 4)
                }
                recommendations["warnings"].append("Limited GPU memory - consider using CPU processing for large datasets")
                recommendations["hardware_upgrades"].append("Upgrade to GPU with 8GB+ VRAM for better performance")
            else:
                gpu_capable = False
        
        if not gpu_capable:
            recommendations["overall_capability"] = "cpu_only"
            recommendations["recommended_config"] = {
                "model_size": "small",
                "gpu_workers": 0,
                "instances_per_gpu": 0,
                "batch_size": "small",
                "cpu_workers": min(self.cpu_info.get("logical_cores", 4), 4)
            }
            recommendations["warnings"].append("No suitable GPU found - will use CPU processing (much slower)")
            recommendations["hardware_upgrades"].append("Add CUDA-compatible GPU with 8GB+ VRAM for significant speedup")
        
        # Memory recommendations
        total_ram = self.memory_info.get("total_ram_gb", 0)
        if total_ram < 16:
            recommendations["warnings"].append("Limited system RAM - may cause memory pressure during processing")
            recommendations["hardware_upgrades"].append("Upgrade to 32GB+ RAM for optimal performance")
        elif total_ram >= 32:
            recommendations["optimizations"].append("Abundant RAM allows for larger CPU worker counts and caching")
        
        # CPU recommendations
        cpu_cores = self.cpu_info.get("logical_cores", 1)
        if cpu_cores >= 16:
            recommendations["optimizations"].append("High core count allows for extensive parallel CPU processing")
        elif cpu_cores < 8:
            recommendations["warnings"].append("Limited CPU cores may bottleneck coordination tasks")
        
        # Storage recommendations
        root_storage = next((s for s in self.system_info.get("storage", []) if s["mountpoint"] == "/"), None)
        if root_storage and root_storage["free_gb"] < 50:
            recommendations["warnings"].append("Low disk space - ensure sufficient space for model caching and checkpoints")
        
        # Multi-GPU specific recommendations
        if gpu_count > 1:
            recommendations["optimizations"].append(f"Multi-GPU setup detected ({gpu_count} GPUs) - enable distributed processing")
            recommendations["recommended_config"]["enable_multi_gpu"] = True
        
        self.recommendations = recommendations
        
        # Print summary
        print(f"\n📋 SYSTEM CAPABILITY SUMMARY:")
        print(f"   • Overall Rating: {recommendations['overall_capability'].upper()}")
        print(f"   • Recommended Model: {recommendations['recommended_config'].get('model_size', 'N/A')}")
        print(f"   • GPU Workers: {recommendations['recommended_config'].get('gpu_workers', 0)}")
        print(f"   • CPU Workers: {recommendations['recommended_config'].get('cpu_workers', 4)}")
        
        if recommendations["warnings"]:
            print(f"\n⚠️  WARNINGS:")
            for warning in recommendations["warnings"]:
                print(f"   • {warning}")
        
        if recommendations["optimizations"]:
            print(f"\n💡 OPTIMIZATIONS:")
            for opt in recommendations["optimizations"]:
                print(f"   • {opt}")
    
    def save_analysis(self, output_file: str = None):
        """Save analysis to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"system_analysis_{timestamp}.json"
        
        analysis_data = {
            "system_info": self.system_info,
            "cpu_info": self.cpu_info,
            "memory_info": self.memory_info,
            "gpu_info": self.gpu_info,
            "recommendations": self.recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            print(f"💾 System analysis saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Failed to save analysis: {e}")
            return None
    
    def generate_optimal_command(self) -> str:
        """Generate optimal command line for the enhanced runner"""
        config = self.recommendations.get("recommended_config", {})
        
        cmd_parts = ["python enhanced_hybrid_gpu_cpu_runner.py"]
        
        if config.get("model_size"):
            cmd_parts.append(f"--model-size {config['model_size']}")
        
        if config.get("gpu_workers", 0) > 0:
            cmd_parts.append(f"--instances-per-gpu {config.get('instances_per_gpu', 1)}")
        
        if config.get("cpu_workers"):
            cmd_parts.append(f"--cpu-workers {config['cpu_workers']}")
        
        if config.get("batch_size") == "small":
            cmd_parts.append("--small-batches")
        
        if self.memory_info.get("total_ram_gb", 0) < 16:
            cmd_parts.append("--memory-fraction 0.6")
        
        # Add test run for initial testing
        cmd_parts.append("--test-run")
        
        return " ".join(cmd_parts)


def main():
    """Main analysis function"""
    print("🔍 System Capability Analysis for Multi-GPU Translation Processing")
    print("=" * 70)
    
    analyzer = SystemCapabilityAnalyzer()
    analysis = analyzer.analyze_full_system()
    
    print("\n" + "=" * 70)
    
    # Save analysis
    output_file = analyzer.save_analysis()
    
    # Generate optimal command
    optimal_command = analyzer.generate_optimal_command()
    print(f"\n🚀 RECOMMENDED COMMAND:")
    print(f"   {optimal_command}")
    
    print(f"\n📊 For detailed analysis, see: {output_file}")
    
    return analysis


if __name__ == "__main__":
    main()
