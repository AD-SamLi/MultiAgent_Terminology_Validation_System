#!/usr/bin/env python3
"""
Adaptive System Configuration
Automatically detects and adapts to any end user's system configuration
No manual configuration required - fully plug-and-play
"""

import psutil
import platform
import torch
import os
import subprocess
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import multi-model GPU management
try:
    from multi_model_gpu_manager import MultiModelGPUManager, create_multi_model_allocation
    MULTI_MODEL_GPU_AVAILABLE = True
except ImportError:
    MULTI_MODEL_GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SystemProfile:
    """Complete system profile for adaptive configuration"""
    # CPU Information
    cpu_cores_logical: int
    cpu_cores_physical: int
    cpu_frequency_max: float
    cpu_architecture: str
    
    # Memory Information
    total_memory_gb: float
    available_memory_gb: float
    memory_usage_percent: float
    
    # GPU Information
    gpu_count: int
    gpu_details: List[Dict[str, Any]]
    cuda_available: bool
    total_gpu_memory_gb: float
    available_gpu_memory_gb: float
    
    # System Information
    os_type: str
    os_version: str
    python_version: str
    platform_architecture: str
    
    # Performance Indicators
    system_load_1min: float
    system_load_5min: float
    disk_usage_percent: float
    
    # Capabilities
    supports_multiprocessing: bool
    supports_threading: bool
    supports_multi_model_gpu: bool
    estimated_performance_tier: str  # "low", "medium", "high", "extreme"

@dataclass
class AdaptiveConfig:
    """Adaptive configuration based on system profile"""
    # Core Processing
    gpu_workers: int
    cpu_workers: int
    cpu_translation_workers: int
    preprocessing_workers: int
    io_workers: int
    
    # Memory Management
    gpu_batch_size: int
    cpu_batch_size: int
    max_queue_size: int
    memory_limit_gb: float
    enable_memory_mapping: bool
    enable_caching: bool
    
    # Model Configuration
    model_precision: str  # "float32", "float16", "int8"
    enable_model_parallel: bool
    model_offload_strategy: str  # "none", "cpu", "disk"
    
    # Processing Strategy
    processing_strategy: str
    language_processing_tier: str  # "minimal", "core", "extended", "full"
    optimization_level: str  # "conservative", "balanced", "aggressive", "extreme"
    
    # Feature Toggles
    enable_predictive_caching: bool
    enable_dynamic_batching: bool
    enable_async_processing: bool
    enable_checkpoint_compression: bool
    enable_multi_model_gpu: bool
    
    # Multi-Model Configuration
    models_per_gpu: Dict[int, List[str]]  # GPU ID -> List of model names
    gpu_memory_allocation: Dict[int, float]  # GPU ID -> Memory allocated

class AdaptiveSystemDetector:
    """Automatically detects and adapts to any system configuration"""
    
    def __init__(self):
        self.system_profile = self._create_system_profile()
        self.adaptive_config = self._generate_adaptive_config()
        
    def _create_system_profile(self) -> SystemProfile:
        """Create comprehensive system profile"""
        logger.info("[SEARCH] Detecting system configuration...")
        
        # CPU Detection
        cpu_info = self._detect_cpu_info()
        
        # Memory Detection
        memory_info = self._detect_memory_info()
        
        # GPU Detection
        gpu_info = self._detect_gpu_info()
        
        # System Detection
        system_info = self._detect_system_info()
        
        # Performance Detection
        performance_info = self._detect_performance_info()
        
        # Capability Detection
        capability_info = self._detect_capabilities()
        
        # Determine performance tier
        performance_tier = self._determine_performance_tier(
            cpu_info, memory_info, gpu_info, performance_info
        )
        
        profile = SystemProfile(
            # CPU
            cpu_cores_logical=cpu_info['logical'],
            cpu_cores_physical=cpu_info['physical'],
            cpu_frequency_max=cpu_info['frequency_max'],
            cpu_architecture=cpu_info['architecture'],
            
            # Memory
            total_memory_gb=memory_info['total_gb'],
            available_memory_gb=memory_info['available_gb'],
            memory_usage_percent=memory_info['usage_percent'],
            
            # GPU
            gpu_count=gpu_info['count'],
            gpu_details=gpu_info['details'],
            cuda_available=gpu_info['cuda_available'],
            total_gpu_memory_gb=gpu_info['total_memory_gb'],
            available_gpu_memory_gb=gpu_info['available_memory_gb'],
            
            # System
            os_type=system_info['os_type'],
            os_version=system_info['os_version'],
            python_version=system_info['python_version'],
            platform_architecture=system_info['architecture'],
            
            # Performance
            system_load_1min=performance_info['load_1min'],
            system_load_5min=performance_info['load_5min'],
            disk_usage_percent=performance_info['disk_usage'],
            
            # Capabilities
            supports_multiprocessing=capability_info['multiprocessing'],
            supports_threading=capability_info['threading'],
            supports_multi_model_gpu=capability_info['multi_model_gpu'],
            estimated_performance_tier=performance_tier
        )
        
        self._log_system_profile(profile)
        return profile
    
    def _detect_cpu_info(self) -> Dict[str, Any]:
        """Detect CPU information"""
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                'logical': psutil.cpu_count(logical=True),
                'physical': psutil.cpu_count(logical=False),
                'frequency_max': cpu_freq.max if cpu_freq else 0.0,
                'architecture': platform.machine()
            }
        except Exception as e:
            logger.warning(f"CPU detection warning: {e}")
            return {
                'logical': 4,  # Conservative fallback
                'physical': 2,
                'frequency_max': 2000.0,
                'architecture': 'unknown'
            }
    
    def _detect_memory_info(self) -> Dict[str, Any]:
        """Detect memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent
            }
        except Exception as e:
            logger.warning(f"Memory detection warning: {e}")
            return {
                'total_gb': 8.0,  # Conservative fallback
                'available_gb': 4.0,
                'usage_percent': 50.0
            }
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information with comprehensive fallback"""
        gpu_info = {
            'count': 0,
            'details': [],
            'cuda_available': False,
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0
        }
        
        try:
            # Try PyTorch CUDA detection first
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['count']):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory / (1024**3)
                        
                        # Get current memory usage
                        torch.cuda.set_device(i)
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        available = total_memory - allocated - reserved
                        
                        gpu_detail = {
                            'id': i,
                            'name': props.name,
                            'total_memory_gb': total_memory,
                            'available_memory_gb': available,
                            'compute_capability': f"{props.major}.{props.minor}",
                            'multiprocessor_count': getattr(props, 'multiprocessor_count', 0)
                        }
                        
                        gpu_info['details'].append(gpu_detail)
                        gpu_info['total_memory_gb'] += total_memory
                        gpu_info['available_memory_gb'] += available
                        
                    except Exception as e:
                        logger.warning(f"Error detecting GPU {i}: {e}")
            
            # Fallback: Try nvidia-ml-py
            if gpu_info['count'] == 0:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    gpu_count = pynvml.nvmlDeviceGetCount()
                    
                    if gpu_count > 0:
                        gpu_info['count'] = gpu_count
                        gpu_info['cuda_available'] = True
                        
                        for i in range(gpu_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            gpu_detail = {
                                'id': i,
                                'name': name,
                                'total_memory_gb': mem_info.total / (1024**3),
                                'available_memory_gb': mem_info.free / (1024**3),
                                'compute_capability': 'unknown',
                                'multiprocessor_count': 0
                            }
                            
                            gpu_info['details'].append(gpu_detail)
                            gpu_info['total_memory_gb'] += gpu_detail['total_memory_gb']
                            gpu_info['available_memory_gb'] += gpu_detail['available_memory_gb']
                            
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"nvidia-ml-py detection failed: {e}")
            
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        return gpu_info
    
    def _detect_system_info(self) -> Dict[str, Any]:
        """Detect system information"""
        return {
            'os_type': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
    
    def _detect_performance_info(self) -> Dict[str, Any]:
        """Detect system performance indicators"""
        try:
            # System load (Unix-like systems)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:\\').percent
            
            return {
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'disk_usage': disk_usage
            }
        except Exception as e:
            logger.warning(f"Performance detection warning: {e}")
            return {
                'load_1min': 0.0,
                'load_5min': 0.0,
                'disk_usage': 50.0
            }
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect system processing capabilities"""
        try:
            # Test multiprocessing
            import multiprocessing as mp
            supports_mp = True
            try:
                mp.set_start_method('spawn', force=True)
            except:
                pass
            
            # Test threading
            import threading
            supports_threading = True
            
            # Test multi-model GPU capability
            supports_multi_model_gpu = False
            if MULTI_MODEL_GPU_AVAILABLE:
                try:
                    # Direct GPU memory check using PyTorch
                    if torch.cuda.is_available():
                        for gpu_id in range(torch.cuda.device_count()):
                            torch.cuda.set_device(gpu_id)
                            props = torch.cuda.get_device_properties(gpu_id)
                            total_memory = props.total_memory / (1024**3)
                            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                            available = total_memory - allocated - reserved - 0.5  # 0.5GB buffer
                            
                            if available > 4.0:  # Need at least 4GB for multi-model
                                supports_multi_model_gpu = True
                                logger.info(f"[GPU] GPU {gpu_id} supports multi-model: {available:.1f}GB available")
                                break
                except Exception as e:
                    logger.debug(f"Multi-model GPU detection failed: {e}")
                    supports_multi_model_gpu = False
            
            return {
                'multiprocessing': supports_mp,
                'threading': supports_threading,
                'multi_model_gpu': supports_multi_model_gpu
            }
        except Exception as e:
            logger.warning(f"Capability detection warning: {e}")
            return {
                'multiprocessing': True,  # Assume support
                'threading': True,
                'multi_model_gpu': False
            }
    
    def _determine_performance_tier(self, cpu_info: Dict, memory_info: Dict, 
                                  gpu_info: Dict, perf_info: Dict) -> str:
        """Determine system performance tier for adaptive configuration"""
        
        score = 0
        
        # CPU scoring
        if cpu_info['logical'] >= 16:
            score += 4
        elif cpu_info['logical'] >= 8:
            score += 3
        elif cpu_info['logical'] >= 4:
            score += 2
        else:
            score += 1
        
        # Memory scoring
        if memory_info['total_gb'] >= 32:
            score += 4
        elif memory_info['total_gb'] >= 16:
            score += 3
        elif memory_info['total_gb'] >= 8:
            score += 2
        else:
            score += 1
        
        # GPU scoring
        if gpu_info['count'] >= 2 and gpu_info['total_memory_gb'] >= 16:
            score += 4
        elif gpu_info['count'] >= 1 and gpu_info['total_memory_gb'] >= 8:
            score += 3
        elif gpu_info['count'] >= 1 and gpu_info['total_memory_gb'] >= 4:
            score += 2
        elif gpu_info['cuda_available']:
            score += 1
        
        # System load penalty
        if perf_info['load_1min'] > cpu_info['logical']:
            score -= 1
        
        # Memory usage penalty
        if memory_info['usage_percent'] > 80:
            score -= 1
        
        # Determine tier
        if score >= 10:
            return "extreme"
        elif score >= 8:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"
    
    def _generate_adaptive_config(self) -> AdaptiveConfig:
        """Generate adaptive configuration based on system profile"""
        profile = self.system_profile
        
        # Base configuration by performance tier
        if profile.estimated_performance_tier == "extreme":
            config = self._get_extreme_performance_config(profile)
        elif profile.estimated_performance_tier == "high":
            config = self._get_high_performance_config(profile)
        elif profile.estimated_performance_tier == "medium":
            config = self._get_medium_performance_config(profile)
        else:
            config = self._get_low_performance_config(profile)
        
        # Apply OS-specific adjustments
        config = self._apply_os_adjustments(config, profile)
        
        # Apply memory constraints
        config = self._apply_memory_constraints(config, profile)
        
        # Apply GPU optimizations
        config = self._apply_gpu_optimizations(config, profile)
        
        logger.info(f"[TARGET] Generated {profile.estimated_performance_tier.upper()} performance configuration")
        self._log_adaptive_config(config)
        
        return config
    
    def _get_extreme_performance_config(self, profile: SystemProfile) -> AdaptiveConfig:
        """Configuration for extreme performance systems - OPTIMAL RESOURCE UTILIZATION"""
        # REBALANCED CPU/GPU UTILIZATION - Optimize for better GPU usage and reduced CPU pressure
        available_cores = profile.cpu_cores_logical - 2  # Leave 2 cores for OS
        
        # AGGRESSIVE GPU UTILIZATION FIRST - Calculate optimal GPU workers
        # For multi-GPU: use all GPUs
        # For single GPU with multi-model: maximize NLLB instances we can load
        if profile.gpu_count > 1:
            gpu_workers = profile.gpu_count  # Use ALL GPUs available
            max_nllb_instances = gpu_workers * 2  # 2 NLLB per GPU
        else:
            # Single GPU - maximize multi-model capability
            gpu_workers = 1  # Default for single GPU
            max_nllb_instances = 2  # Default
            if profile.supports_multi_model_gpu:
                # Calculate how many NLLB instances we can load (more aggressive)
                estimated_memory_per_gpu = profile.available_gpu_memory_gb * 0.7  # Increase from 60% to 70%
                nllb_memory_per_instance = 3.2  # Slightly optimistic estimate (was 3.5)
                max_nllb_instances = max(2, min(4, int(estimated_memory_per_gpu / nllb_memory_per_instance)))  # Min 2, Max 4
                gpu_workers = min(max_nllb_instances, 3)  # Increase cap from 2 to 3
                logger.info(f"[GPU] OPTIMIZED Single GPU Multi-Model: {max_nllb_instances} NLLB instances -> {gpu_workers} GPU workers")
        
        # TRANSLATION-OPTIMIZED CPU ALLOCATION - CPU translation is the bottleneck
        # Prioritize CPU translation workers heavily, minimal preprocessing (queueing is fast)
        cpu_translation_workers = max(8, int(available_cores * 0.70))  # 70% for CPU translation (bottleneck)
        cpu_workers = max(2, int(available_cores * 0.20))              # 20% for main processing  
        preprocessing_workers = max(1, int(available_cores * 0.10))    # 10% for preprocessing (minimal)
        
        # Ensure total doesn't exceed available cores
        total_workers = cpu_workers + cpu_translation_workers + preprocessing_workers
        if total_workers > available_cores:
            # Scale down proportionally but prioritize CPU translation workers
            scale_factor = available_cores / total_workers
            cpu_workers = max(4, int(cpu_workers * scale_factor))
            cpu_translation_workers = max(4, int(cpu_translation_workers * scale_factor))
            preprocessing_workers = max(2, int(preprocessing_workers * scale_factor))
        
        logger.info(f"[CONFIG] EXTREME TRANSLATION-OPTIMIZED: {gpu_workers} GPU ({max_nllb_instances} NLLB) + {cpu_translation_workers} CPU-Trans + {cpu_workers} CPU-Main = {total_workers}/{available_cores} cores")
        
        # AGGRESSIVE MEMORY UTILIZATION - Use up to 85% of available RAM
        memory_limit_gb = profile.available_memory_gb * 0.85
        
        # MAXIMUM BATCH SIZES for full throughput
        gpu_batch_size = min(128, int(profile.available_gpu_memory_gb * 16))  # Scale with GPU memory
        cpu_batch_size = min(64, profile.cpu_cores_logical * 4)  # Scale with CPU cores
        
        # Configure multi-model GPU allocation
        models_per_gpu = {}
        gpu_memory_allocation = {}
        enable_multi_model = profile.supports_multi_model_gpu and MULTI_MODEL_GPU_AVAILABLE
        
        if enable_multi_model:
            # Create multi-model allocation for extreme performance
            try:
                required_models = ["nllb-200-1.3B", "gpt-4.1", "terminology-agent", "validation-agent", "dictionary-agent"]
                manager, allocations = create_multi_model_allocation(required_models)
                
                for allocation in allocations:
                    if allocation.allocated_models:
                        models_per_gpu[allocation.gpu_id] = [model.name for model in allocation.allocated_models]
                        gpu_memory_allocation[allocation.gpu_id] = allocation.memory_used_gb
                
                logger.info(f"[GPU] Multi-model GPU allocation: {len(models_per_gpu)} active GPUs")
            except Exception as e:
                logger.warning(f"Multi-model GPU allocation failed: {e}")
                enable_multi_model = False
        
        return AdaptiveConfig(
            gpu_workers=gpu_workers,
            cpu_workers=cpu_workers,
            cpu_translation_workers=cpu_translation_workers,
            preprocessing_workers=preprocessing_workers,
            io_workers=preprocessing_workers,  # Use preprocessing_workers for I/O
            gpu_batch_size=gpu_batch_size,
            cpu_batch_size=cpu_batch_size,
            max_queue_size=max(200, profile.cpu_cores_logical * 10),  # Massive queue for throughput
            memory_limit_gb=memory_limit_gb,
            enable_memory_mapping=True,
            enable_caching=True,
            model_precision="float16",
            enable_model_parallel=profile.gpu_count >= 2,
            model_offload_strategy="none",
            processing_strategy="maximum_utilization",
            language_processing_tier="full",
            optimization_level="extreme",
            enable_predictive_caching=True,
            enable_dynamic_batching=True,
            enable_async_processing=True,
            enable_checkpoint_compression=True,
            enable_multi_model_gpu=enable_multi_model,
            models_per_gpu=models_per_gpu,
            gpu_memory_allocation=gpu_memory_allocation
        )
    
    def _get_high_performance_config(self, profile: SystemProfile) -> AdaptiveConfig:
        """Configuration for high performance systems - REBALANCED RESOURCE UTILIZATION"""
        # REBALANCED CPU/GPU UTILIZATION - Optimize for better GPU usage and reduced CPU pressure
        available_cores = profile.cpu_cores_logical - 3  # Leave 3 cores for OS
        
        # AGGRESSIVE GPU UTILIZATION FIRST - Calculate optimal GPU workers
        if profile.gpu_count > 1:
            gpu_workers = profile.gpu_count  # Use ALL GPUs available
            max_nllb_instances = gpu_workers * 2  # 2 NLLB per GPU
        else:
            # Single GPU - maximize multi-model capability
            gpu_workers = 1  # Default for single GPU
            max_nllb_instances = 2  # Default
            if profile.supports_multi_model_gpu:
                # Calculate how many NLLB instances we can load (more aggressive)
                estimated_memory_per_gpu = profile.available_gpu_memory_gb * 0.65  # Increase from 60% to 65%
                nllb_memory_per_instance = 3.3  # Slightly optimistic estimate
                max_nllb_instances = max(2, min(3, int(estimated_memory_per_gpu / nllb_memory_per_instance)))  # Min 2, Max 3
                gpu_workers = min(max_nllb_instances, 3)  # Increase cap from 2 to 3
                logger.info(f"[GPU] OPTIMIZED High-Perf GPU Multi-Model: {max_nllb_instances} NLLB instances -> {gpu_workers} GPU workers")
        
        # TRANSLATION-OPTIMIZED CPU ALLOCATION - CPU translation is the bottleneck
        # Prioritize CPU translation workers heavily, minimal preprocessing (queueing is fast)
        cpu_translation_workers = max(8, int(available_cores * 0.70))  # 70% for CPU translation (bottleneck)
        cpu_workers = max(2, int(available_cores * 0.20))              # 20% for main processing  
        preprocessing_workers = max(1, int(available_cores * 0.10))    # 10% for preprocessing (minimal)
        
        # Ensure total doesn't exceed available cores
        total_workers = cpu_workers + cpu_translation_workers + preprocessing_workers
        if total_workers > available_cores:
            # Scale down proportionally but prioritize CPU translation workers
            scale_factor = available_cores / total_workers
            cpu_workers = max(4, int(cpu_workers * scale_factor))
            cpu_translation_workers = max(4, int(cpu_translation_workers * scale_factor))
            preprocessing_workers = max(2, int(preprocessing_workers * scale_factor))
        
            logger.info(f"[TRANSLATION-OPTIMIZED] {gpu_workers} GPU ({max_nllb_instances} NLLB) + {cpu_translation_workers} CPU-Trans + {cpu_workers} CPU-Main = {total_workers}/{available_cores} cores")
        
        # AGGRESSIVE MEMORY UTILIZATION - Use up to 80% of available RAM
        memory_limit_gb = profile.available_memory_gb * 0.80
        
        # OPTIMIZED BATCH SIZES for high throughput
        gpu_batch_size = min(96, int(profile.available_gpu_memory_gb * 12))  # Scale with GPU memory
        cpu_batch_size = min(48, profile.cpu_cores_logical * 3)  # Scale with CPU cores
        
        return AdaptiveConfig(
            gpu_workers=gpu_workers,
            cpu_workers=cpu_workers,
            cpu_translation_workers=cpu_translation_workers,
            preprocessing_workers=preprocessing_workers,
            io_workers=preprocessing_workers,  # Use preprocessing_workers for I/O
            gpu_batch_size=gpu_batch_size,
            cpu_batch_size=cpu_batch_size,
            max_queue_size=max(150, profile.cpu_cores_logical * 8),  # Large queue for throughput
            memory_limit_gb=memory_limit_gb,
            enable_memory_mapping=True,
            enable_caching=True,
            model_precision="float16",
            enable_model_parallel=profile.gpu_count >= 2,
            model_offload_strategy="none",
            processing_strategy="aggressive_utilization",
            language_processing_tier="extended",
            optimization_level="aggressive",
            enable_predictive_caching=True,
            enable_dynamic_batching=True,
            enable_async_processing=True,
            enable_checkpoint_compression=False,
            enable_multi_model_gpu=profile.supports_multi_model_gpu and max_nllb_instances > 1,
            models_per_gpu={0: max_nllb_instances} if profile.gpu_count > 0 else {},
            gpu_memory_allocation={0: profile.available_gpu_memory_gb} if profile.gpu_count > 0 else {}
        )
    
    def _get_medium_performance_config(self, profile: SystemProfile) -> AdaptiveConfig:
        """Configuration for medium performance systems - BALANCED OPTIMAL UTILIZATION"""
        # BALANCED CPU UTILIZATION - Use cores efficiently with good headroom
        available_cores = profile.cpu_cores_logical - 4  # Leave 4 cores for OS
        
        # Distribute cores optimally across different worker types (rebalanced)
        cpu_workers = max(4, int(available_cores * 0.50))      # 50% for main processing
        cpu_translation_workers = max(3, int(available_cores * 0.30))  # 30% for CPU translation
        preprocessing_workers = max(2, int(available_cores * 0.20))  # 20% for preprocessing
        
        # Ensure total doesn't exceed available cores
        total_workers = cpu_workers + cpu_translation_workers + preprocessing_workers
        if total_workers > available_cores:
            # Scale down proportionally
            scale_factor = available_cores / total_workers
            cpu_workers = max(3, int(cpu_workers * scale_factor))
            cpu_translation_workers = max(3, int(cpu_translation_workers * scale_factor))
            preprocessing_workers = max(2, int(preprocessing_workers * scale_factor))
        
        # USE ALL AVAILABLE GPUS - even for medium systems
        gpu_workers = profile.gpu_count  # Use ALL GPUs available
        
        # BALANCED MEMORY UTILIZATION - Use up to 70% of available RAM
        memory_limit_gb = profile.available_memory_gb * 0.70
        
        # OPTIMIZED BATCH SIZES for balanced throughput
        gpu_batch_size = min(64, int(profile.available_gpu_memory_gb * 8))  # Scale with GPU memory
        cpu_batch_size = min(32, profile.cpu_cores_logical * 2)  # Scale with CPU cores
        
        return AdaptiveConfig(
            gpu_workers=gpu_workers,
            cpu_workers=cpu_workers,
            cpu_translation_workers=cpu_translation_workers,
            preprocessing_workers=preprocessing_workers,
            io_workers=preprocessing_workers,  # Use preprocessing_workers for I/O
            gpu_batch_size=gpu_batch_size,
            cpu_batch_size=cpu_batch_size,
            max_queue_size=max(100, profile.cpu_cores_logical * 6),  # Good-sized queue
            memory_limit_gb=memory_limit_gb,
            enable_memory_mapping=True,  # Always enable for better performance
            enable_caching=True,
            model_precision="float16" if profile.gpu_count > 0 else "float32",
            enable_model_parallel=profile.gpu_count >= 2,
            model_offload_strategy="none",  # Keep models in memory for speed
            processing_strategy="balanced_full_utilization",
            language_processing_tier="core",
            optimization_level="balanced",
            enable_predictive_caching=True,
            enable_dynamic_batching=True,
            enable_async_processing=True,  # Enable async for better throughput
            enable_checkpoint_compression=False,
            enable_multi_model_gpu=False,
            models_per_gpu={0: 1} if profile.gpu_count > 0 else {},
            gpu_memory_allocation={0: profile.available_gpu_memory_gb} if profile.gpu_count > 0 else {}
        )
    
    def _get_low_performance_config(self, profile: SystemProfile) -> AdaptiveConfig:
        """Configuration for low performance systems - SMART RESOURCE UTILIZATION"""
        # SMART CPU UTILIZATION - Use available cores efficiently (rebalanced)
        available_cores = profile.cpu_cores_logical - 2  # Leave 2 cores for OS
        cpu_workers = max(4, int(available_cores * 0.60))  # 60% for main processing
        cpu_translation_workers = max(3, int(available_cores * 0.25))  # 25% for CPU translation
        preprocessing_workers = max(2, profile.cpu_cores_logical // 4)  # Use 1/4 of cores
        
        # USE AVAILABLE GPUS - even low-end systems benefit from GPU acceleration
        gpu_workers = profile.gpu_count if profile.available_gpu_memory_gb >= 2 else 0
        
        # CONSERVATIVE BUT EFFICIENT MEMORY UTILIZATION - Use up to 60% of available RAM
        memory_limit_gb = profile.available_memory_gb * 0.60
        
        # OPTIMIZED BATCH SIZES for low-end systems
        gpu_batch_size = min(32, int(profile.available_gpu_memory_gb * 4)) if gpu_workers > 0 else 16
        cpu_batch_size = min(24, profile.cpu_cores_logical * 2)  # Scale with CPU cores
        
        return AdaptiveConfig(
            gpu_workers=gpu_workers,
            cpu_workers=cpu_workers,
            cpu_translation_workers=cpu_translation_workers,
            preprocessing_workers=preprocessing_workers,
            io_workers=max(2, profile.cpu_cores_logical // 8),  # At least 2 I/O workers
            gpu_batch_size=gpu_batch_size,
            cpu_batch_size=cpu_batch_size,
            max_queue_size=max(50, profile.cpu_cores_logical * 4),  # Reasonable queue size
            memory_limit_gb=memory_limit_gb,
            enable_memory_mapping=profile.available_memory_gb > 4,  # Enable if enough RAM
            enable_caching=True,  # Enable caching for efficiency
            model_precision="float16" if gpu_workers > 0 else "float32",
            enable_model_parallel=False,  # Keep simple for low-end
            model_offload_strategy="cpu" if profile.available_memory_gb < 6 else "none",
            processing_strategy="efficient_utilization",
            language_processing_tier="minimal",
            optimization_level="balanced",  # Changed from conservative to balanced
            enable_predictive_caching=True,  # Enable for efficiency
            enable_dynamic_batching=True,   # Enable for better throughput
            enable_async_processing=True,   # Enable async for better performance
            enable_checkpoint_compression=True,
            enable_multi_model_gpu=False,
            models_per_gpu={0: 1} if gpu_workers > 0 else {},
            gpu_memory_allocation={0: profile.available_gpu_memory_gb} if gpu_workers > 0 and profile.available_gpu_memory_gb > 0 else {}
        )
    
    def _apply_os_adjustments(self, config: AdaptiveConfig, profile: SystemProfile) -> AdaptiveConfig:
        """Apply OS-specific adjustments"""
        if profile.os_type == "Windows":
            # Windows multiprocessing can be less stable
            config.cpu_workers = min(config.cpu_workers, 8)
            config.preprocessing_workers = min(config.preprocessing_workers, 4)
        elif profile.os_type == "Darwin":  # macOS
            # macOS has different memory management
            config.memory_limit_gb *= 0.8
        
        return config
    
    def _apply_memory_constraints(self, config: AdaptiveConfig, profile: SystemProfile) -> AdaptiveConfig:
        """Apply memory-based constraints with AGGRESSIVE UTILIZATION"""
        # Calculate total workers and memory allocation
        total_workers = config.gpu_workers + config.cpu_workers + config.preprocessing_workers + config.io_workers
        memory_per_worker = config.memory_limit_gb / total_workers if total_workers > 0 else config.memory_limit_gb
        
        # AGGRESSIVE MEMORY UTILIZATION - Allow lower memory per worker for higher throughput
        if memory_per_worker < 0.5:  # Less than 512MB per worker (very aggressive)
            # Slightly reduce workers but maintain high utilization
            reduction_factor = max(0.7, 0.5 / memory_per_worker)  # Less aggressive reduction
            config.cpu_workers = max(4, int(config.cpu_workers * reduction_factor))
            config.preprocessing_workers = max(2, int(config.preprocessing_workers * reduction_factor))
            logger.info(f"[WARNING] Aggressive memory allocation: {memory_per_worker:.2f}GB per worker")
        elif memory_per_worker < 1.0:  # Less than 1GB per worker
            # Minor reduction for optimal balance
            reduction_factor = max(0.8, 1.0 / memory_per_worker)
            config.cpu_workers = max(3, int(config.cpu_workers * reduction_factor))
            config.preprocessing_workers = max(2, int(config.preprocessing_workers * reduction_factor))
        
        # DYNAMIC MEMORY SCALING - Increase batch sizes if we have extra memory
        available_memory_per_worker = config.memory_limit_gb / total_workers
        if available_memory_per_worker > 2.0:  # More than 2GB per worker
            # Scale up batch sizes for better throughput
            memory_multiplier = min(2.0, available_memory_per_worker / 2.0)
            config.gpu_batch_size = min(256, int(config.gpu_batch_size * memory_multiplier))
            config.cpu_batch_size = min(128, int(config.cpu_batch_size * memory_multiplier))
            config.max_queue_size = min(500, int(config.max_queue_size * memory_multiplier))
            logger.info(f"[LAUNCH] Memory scaling applied: {memory_multiplier:.1f}x batch size increase")
        
        return config
    
    def _apply_gpu_optimizations(self, config: AdaptiveConfig, profile: SystemProfile) -> AdaptiveConfig:
        """Apply GPU-specific optimizations with MAXIMUM GPU UTILIZATION"""
        if profile.gpu_count == 0:
            # No GPU - maximize CPU utilization instead
            config.gpu_workers = 0
            config.enable_model_parallel = False
            config.model_precision = "float32"
            # Compensate with more CPU workers
            config.cpu_workers = min(config.cpu_workers + 4, profile.cpu_cores_logical - 1)
            logger.info("[SETUP] No GPU detected - boosting CPU workers for compensation")
        else:
            # AGGRESSIVE GPU UTILIZATION STRATEGIES
            total_gpu_memory = profile.available_gpu_memory_gb
            
            if total_gpu_memory >= 16:
                # HIGH GPU MEMORY - Maximum utilization
                config.gpu_batch_size = min(config.gpu_batch_size * 2, 256)  # Double batch size
                config.model_precision = "float16"  # Optimal precision
                config.enable_model_parallel = profile.gpu_count >= 2
                logger.info(f"[LAUNCH] High GPU memory ({total_gpu_memory:.1f}GB) - maximum batch sizes enabled")
                
            elif total_gpu_memory >= 8:
                # MEDIUM GPU MEMORY - Optimized utilization
                config.gpu_batch_size = min(config.gpu_batch_size, 128)
                config.model_precision = "float16"
                config.enable_model_parallel = profile.gpu_count >= 2
                logger.info(f"[FAST] Medium GPU memory ({total_gpu_memory:.1f}GB) - optimized configuration")
                
            elif total_gpu_memory >= 4:
                # LOW GPU MEMORY - Efficient utilization
                config.gpu_batch_size = min(config.gpu_batch_size, 64)
                config.model_precision = "float16"  # Still use float16 for efficiency
                logger.info(f"[SETUP] Low GPU memory ({total_gpu_memory:.1f}GB) - efficient configuration")
                
            else:
                # VERY LOW GPU MEMORY - Conservative but still utilize GPU
                config.gpu_batch_size = min(config.gpu_batch_size, 32)
                config.model_precision = "int8"  # Use quantization
                logger.info(f"[WARNING] Very low GPU memory ({total_gpu_memory:.1f}GB) - quantized models")
            
            # MULTI-GPU OPTIMIZATIONS
            if profile.gpu_count >= 2:
                # Enable advanced multi-GPU features
                config.enable_model_parallel = True
                # Distribute batch processing across GPUs
                config.gpu_batch_size = min(config.gpu_batch_size, 96)  # Prevent memory overflow
                logger.info(f"[GPU] Multi-GPU setup ({profile.gpu_count} GPUs) - parallel processing enabled")
                
                # Scale queue size with GPU count
                config.max_queue_size = min(config.max_queue_size * profile.gpu_count, 1000)
            
            # DYNAMIC GPU MEMORY SCALING
            memory_per_gpu = total_gpu_memory / profile.gpu_count if profile.gpu_count > 0 else 0
            if memory_per_gpu > 6:  # More than 6GB per GPU
                # Scale up processing for high-memory GPUs
                scaling_factor = min(1.5, memory_per_gpu / 6)
                config.gpu_batch_size = min(int(config.gpu_batch_size * scaling_factor), 256)
                logger.info(f"[PERF] GPU memory scaling: {scaling_factor:.1f}x for {memory_per_gpu:.1f}GB per GPU")
        
        return config
    
    def _log_system_profile(self, profile: SystemProfile):
        """Log detected system profile"""
        logger.info("[SYSTEM]  SYSTEM PROFILE DETECTED:")
        logger.info(f"   CPU: {profile.cpu_cores_logical} cores ({profile.cpu_cores_physical} physical)")
        logger.info(f"   Memory: {profile.available_memory_gb:.1f}GB available / {profile.total_memory_gb:.1f}GB total")
        logger.info(f"   GPU: {profile.gpu_count} devices, {profile.available_gpu_memory_gb:.1f}GB available")
        logger.info(f"   Multi-Model GPU: {'Yes' if profile.supports_multi_model_gpu else 'No'}")
        logger.info(f"   OS: {profile.os_type} {profile.os_version}")
        logger.info(f"   Performance Tier: {profile.estimated_performance_tier.upper()}")
    
    def _log_adaptive_config(self, config: AdaptiveConfig):
        """Log generated adaptive configuration"""
        logger.info("⚙️  ADAPTIVE CONFIGURATION:")
        logger.info(f"   Workers: {config.gpu_workers} GPU + {config.cpu_workers} CPU + {config.preprocessing_workers} Prep")
        logger.info(f"   Batching: GPU={config.gpu_batch_size}, CPU={config.cpu_batch_size}")
        logger.info(f"   Memory: {config.memory_limit_gb:.1f}GB limit, Mapping={config.enable_memory_mapping}")
        logger.info(f"   Strategy: {config.processing_strategy}, Tier: {config.language_processing_tier}")
        logger.info(f"   Optimization: {config.optimization_level}, Precision: {config.model_precision}")
        
        if config.enable_multi_model_gpu:
            total_models = sum(len(models) for models in config.models_per_gpu.values())
            active_gpus = len(config.models_per_gpu)
            logger.info(f"   [GPU] Multi-Model GPU: {total_models} models across {active_gpus} GPUs")
            for gpu_id, models in config.models_per_gpu.items():
                memory_gb = config.gpu_memory_allocation.get(gpu_id, 0)
                logger.info(f"     GPU {gpu_id}: {len(models)} models ({memory_gb:.1f}GB)")
    
    def get_system_profile(self) -> SystemProfile:
        """Get the detected system profile"""
        return self.system_profile
    
    def get_adaptive_config(self) -> AdaptiveConfig:
        """Get the generated adaptive configuration"""
        return self.adaptive_config
    
    def export_config(self, filepath: Optional[str] = None) -> str:
        """Export configuration to JSON file"""
        config_data = {
            'system_profile': asdict(self.system_profile),
            'adaptive_config': asdict(self.adaptive_config),
            'generation_timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        if filepath is None:
            filepath = f"adaptive_config_{self.system_profile.os_type.lower()}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[FILE] Configuration exported to: {filepath}")
        return filepath

class DynamicResourceMonitor:
    """Monitors system resources in real-time and adjusts configuration dynamically"""
    
    def __init__(self, initial_config: AdaptiveConfig, system_profile: SystemProfile):
        self.config = initial_config
        self.profile = system_profile
        self.monitoring_active = False
        self.adjustment_history = []
        
    def start_monitoring(self, adjustment_interval: float = 30.0):
        """Start real-time resource monitoring and dynamic adjustment"""
        import threading
        import time
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._check_and_adjust_resources()
                    time.sleep(adjustment_interval)
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info(f"[CONFIG] Dynamic resource monitoring started (interval: {adjustment_interval}s)")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        logger.info("⏹️ Dynamic resource monitoring stopped")
    
    def _check_and_adjust_resources(self):
        """Check current resource usage and adjust configuration if needed"""
        # Get current system state
        current_memory = psutil.virtual_memory()
        current_cpu_percent = psutil.cpu_percent(interval=1)
        
        adjustments_made = []
        
        # MEMORY PRESSURE ADJUSTMENT
        if current_memory.percent > 90:  # Very high memory usage
            if self.config.memory_limit_gb > 2:
                old_limit = self.config.memory_limit_gb
                self.config.memory_limit_gb *= 0.8  # Reduce by 20%
                self.config.gpu_batch_size = max(16, int(self.config.gpu_batch_size * 0.8))
                self.config.cpu_batch_size = max(8, int(self.config.cpu_batch_size * 0.8))
                adjustments_made.append(f"Memory limit: {old_limit:.1f}GB -> {self.config.memory_limit_gb:.1f}GB")
        elif current_memory.percent < 60:  # Low memory usage - can be more aggressive
            if self.config.memory_limit_gb < self.profile.available_memory_gb * 0.8:
                old_limit = self.config.memory_limit_gb
                self.config.memory_limit_gb = min(self.config.memory_limit_gb * 1.1, self.profile.available_memory_gb * 0.8)
                self.config.gpu_batch_size = min(256, int(self.config.gpu_batch_size * 1.1))
                self.config.cpu_batch_size = min(128, int(self.config.cpu_batch_size * 1.1))
                adjustments_made.append(f"Memory limit: {old_limit:.1f}GB -> {self.config.memory_limit_gb:.1f}GB")
        
        # CPU UTILIZATION ADJUSTMENT
        if current_cpu_percent > 95:  # Very high CPU usage
            if self.config.cpu_workers > 4:
                old_workers = self.config.cpu_workers
                self.config.cpu_workers = max(4, int(self.config.cpu_workers * 0.9))
                adjustments_made.append(f"CPU workers: {old_workers} -> {self.config.cpu_workers}")
        elif current_cpu_percent < 70:  # Low CPU usage - can be more aggressive
            if self.config.cpu_workers < self.profile.cpu_cores_logical - 2:
                old_workers = self.config.cpu_workers
                self.config.cpu_workers = min(self.profile.cpu_cores_logical - 2, self.config.cpu_workers + 1)
                adjustments_made.append(f"CPU workers: {old_workers} -> {self.config.cpu_workers}")
        
        # GPU MEMORY ADJUSTMENT (if available)
        if self.profile.gpu_count > 0:
            try:
                for gpu_id in range(self.profile.gpu_count):
                    torch.cuda.set_device(gpu_id)
                    gpu_memory_used = torch.cuda.memory_allocated(gpu_id) / torch.cuda.get_device_properties(gpu_id).total_memory
                    
                    if gpu_memory_used > 0.95:  # Very high GPU memory usage
                        if self.config.gpu_batch_size > 16:
                            old_batch = self.config.gpu_batch_size
                            self.config.gpu_batch_size = max(16, int(self.config.gpu_batch_size * 0.8))
                            adjustments_made.append(f"GPU batch size: {old_batch} -> {self.config.gpu_batch_size}")
                    elif gpu_memory_used < 0.6:  # Low GPU memory usage
                        old_batch = self.config.gpu_batch_size
                        self.config.gpu_batch_size = min(256, int(self.config.gpu_batch_size * 1.1))
                        if old_batch != self.config.gpu_batch_size:
                            adjustments_made.append(f"GPU batch size: {old_batch} -> {self.config.gpu_batch_size}")
            except Exception as e:
                logger.debug(f"GPU memory check failed: {e}")
        
        # Log adjustments if any were made
        if adjustments_made:
            timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
            logger.info(f"[SETUP] [{timestamp}] Dynamic adjustments: {', '.join(adjustments_made)}")
            self.adjustment_history.append({
                'timestamp': timestamp,
                'adjustments': adjustments_made,
                'memory_percent': current_memory.percent,
                'cpu_percent': current_cpu_percent
            })
    
    def get_current_config(self) -> AdaptiveConfig:
        """Get the current (potentially adjusted) configuration"""
        return self.config
    
    def get_adjustment_history(self) -> List[Dict]:
        """Get history of dynamic adjustments"""
        return self.adjustment_history

def get_adaptive_system_config(enable_dynamic_monitoring: bool = True) -> Tuple[SystemProfile, AdaptiveConfig, Optional[DynamicResourceMonitor]]:
    """
    Get adaptive system configuration for any end user system with optional dynamic monitoring
    
    Args:
        enable_dynamic_monitoring: Whether to enable real-time resource monitoring and adjustment
    
    Returns:
        Tuple of (SystemProfile, AdaptiveConfig, DynamicResourceMonitor) automatically configured for the current system
    """
    detector = AdaptiveSystemDetector()
    profile = detector.get_system_profile()
    config = detector.get_adaptive_config()
    
    monitor = None
    if enable_dynamic_monitoring:
        monitor = DynamicResourceMonitor(config, profile)
        monitor.start_monitoring()
        logger.info("[TARGET] Dynamic resource monitoring enabled for maximum utilization")
    
    return profile, config, monitor

if __name__ == "__main__":
    # Test adaptive configuration
    print("[SEARCH] Testing Adaptive System Configuration...")
    
    try:
        profile, config, monitor = get_adaptive_system_config()
        print(f"[OK] Successfully detected {profile.estimated_performance_tier.upper()} performance system")
        print(f"[STATS] Configuration: {config.gpu_workers} GPU + {config.cpu_workers} CPU workers")
        print(f"[LAUNCH] Resource Utilization:")
        print(f"   CPU: {config.cpu_workers + config.preprocessing_workers + config.io_workers}/{profile.cpu_cores_logical} cores ({(config.cpu_workers + config.preprocessing_workers + config.io_workers)/profile.cpu_cores_logical*100:.0f}%)")
        print(f"   GPU: {config.gpu_workers}/{profile.gpu_count} devices ({config.gpu_workers/max(1,profile.gpu_count)*100:.0f}%)")
        print(f"   RAM: {config.memory_limit_gb:.1f}GB/{profile.available_memory_gb:.1f}GB ({config.memory_limit_gb/profile.available_memory_gb*100:.0f}%)")
        print(f"   Batch Sizes: GPU={config.gpu_batch_size}, CPU={config.cpu_batch_size}")
        print(f"   Strategy: {config.processing_strategy}")
        print(f"   Optimization: {config.optimization_level}")
        
        if monitor:
            print("[CONFIG] Dynamic resource monitoring: ENABLED")
            # Stop monitoring for clean exit
            monitor.stop_monitoring()
        
        # Export configuration
        detector = AdaptiveSystemDetector()
        config_file = detector.export_config()
        print(f"[FILE] Configuration saved to: {config_file}")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
