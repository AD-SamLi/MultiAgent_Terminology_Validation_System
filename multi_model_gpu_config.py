#!/usr/bin/env python3
"""
Multi-Model GPU Configuration for 2-3 Model Setup
Optimizes GPU allocation for running multiple models simultaneously
"""

import torch
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MultiModelGPUConfig:
    """Configuration for multi-model GPU setup"""
    total_gpus: int
    models_per_gpu: Dict[int, List[str]]  # GPU ID -> List of model names
    memory_per_model: Dict[str, float]    # Model name -> Memory requirement (GB)
    gpu_memory_available: Dict[int, float] # GPU ID -> Available memory (GB)
    load_balancing_strategy: str = "memory_aware"  # "round_robin", "memory_aware", "model_specific"
    
class MultiModelGPUManager:
    """Manages GPU allocation for multiple models"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_capabilities()
        self.model_configs = {
            'nllb-200-1.3B': {'memory_gb': 3.5, 'compute_intensive': True},
            'gpt-4.1': {'memory_gb': 4.0, 'compute_intensive': True},
            'terminology_agent': {'memory_gb': 2.0, 'compute_intensive': False},
            'validation_agent': {'memory_gb': 2.5, 'compute_intensive': False},
            'dictionary_agent': {'memory_gb': 1.0, 'compute_intensive': False}
        }
    
    def _detect_gpu_capabilities(self) -> Dict[int, Dict[str, Any]]:
        """Detect detailed GPU capabilities for each device"""
        gpu_info = {}
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - falling back to CPU-only mode")
            return gpu_info
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"🔍 Detected {gpu_count} CUDA-capable GPU(s)")
        
        for gpu_id in range(gpu_count):
            props = torch.cuda.get_device_properties(gpu_id)
            
            # Get current memory usage
            torch.cuda.set_device(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            total_memory = props.total_memory / (1024**3)
            available_memory = total_memory - memory_reserved - memory_allocated
            
            gpu_info[gpu_id] = {
                'name': props.name,
                'total_memory_gb': total_memory,
                'available_memory_gb': available_memory,
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': getattr(props, 'multiprocessor_count', 'Unknown'),
                'max_threads_per_block': getattr(props, 'max_threads_per_block', 'Unknown'),
                'utilization_estimate': (memory_allocated + memory_reserved) / total_memory * 100
            }
            
            logger.info(f"   GPU {gpu_id}: {props.name}")
            logger.info(f"     Memory: {available_memory:.1f}GB available / {total_memory:.1f}GB total")
            logger.info(f"     Compute: {props.major}.{props.minor}, {props.multiprocessor_count} MPs")
        
        return gpu_info
    
    def create_optimal_allocation_plan(self, required_models: List[str]) -> MultiModelGPUConfig:
        """Create optimal GPU allocation plan for required models"""
        
        if not self.gpu_info:
            raise RuntimeError("No GPUs available for multi-model setup")
        
        total_gpus = len(self.gpu_info)
        models_per_gpu = {gpu_id: [] for gpu_id in range(total_gpus)}
        gpu_memory_available = {gpu_id: info['available_memory_gb'] for gpu_id, info in self.gpu_info.items()}
        
        # Calculate memory requirements
        model_memory_requirements = {}
        for model in required_models:
            if model in self.model_configs:
                model_memory_requirements[model] = self.model_configs[model]['memory_gb']
            else:
                # Default estimate for unknown models
                model_memory_requirements[model] = 2.0
                logger.warning(f"Unknown model '{model}', using default 2.0GB memory estimate")
        
        # Strategy 1: Memory-aware allocation (prioritize models by memory requirements)
        if total_gpus >= 2:
            # Sort models by memory requirements (largest first)
            sorted_models = sorted(required_models, 
                                 key=lambda x: model_memory_requirements[x], 
                                 reverse=True)
            
            for model in sorted_models:
                memory_needed = model_memory_requirements[model]
                
                # Find GPU with most available memory that can fit this model
                best_gpu = None
                best_available = 0
                
                for gpu_id, available_memory in gpu_memory_available.items():
                    if available_memory >= memory_needed and available_memory > best_available:
                        best_gpu = gpu_id
                        best_available = available_memory
                
                if best_gpu is not None:
                    models_per_gpu[best_gpu].append(model)
                    gpu_memory_available[best_gpu] -= memory_needed
                    logger.info(f"📍 Allocated '{model}' to GPU {best_gpu} ({memory_needed:.1f}GB)")
                else:
                    # Fallback: assign to GPU with most available memory
                    fallback_gpu = max(gpu_memory_available.keys(), 
                                     key=lambda x: gpu_memory_available[x])
                    models_per_gpu[fallback_gpu].append(model)
                    gpu_memory_available[fallback_gpu] -= memory_needed
                    logger.warning(f"⚠️ Fallback: Allocated '{model}' to GPU {fallback_gpu} (may exceed memory)")
        
        else:
            # Single GPU: load all models on GPU 0
            for model in required_models:
                models_per_gpu[0].append(model)
                gpu_memory_available[0] -= model_memory_requirements[model]
            logger.info("📍 Single GPU mode: All models allocated to GPU 0")
        
        config = MultiModelGPUConfig(
            total_gpus=total_gpus,
            models_per_gpu=models_per_gpu,
            memory_per_model=model_memory_requirements,
            gpu_memory_available=gpu_memory_available,
            load_balancing_strategy="memory_aware"
        )
        
        # Log the final allocation plan
        logger.info("🎯 Final GPU Allocation Plan:")
        for gpu_id, models in models_per_gpu.items():
            if models:
                gpu_name = self.gpu_info[gpu_id]['name']
                total_memory = sum(model_memory_requirements[m] for m in models)
                available = gpu_memory_available[gpu_id]
                logger.info(f"   GPU {gpu_id} ({gpu_name}):")
                logger.info(f"     Models: {', '.join(models)}")
                logger.info(f"     Memory: {total_memory:.1f}GB allocated, {available:.1f}GB remaining")
        
        return config
    
    def get_optimized_worker_config(self, allocation_plan: MultiModelGPUConfig) -> Dict[str, Any]:
        """Get optimized worker configuration based on allocation plan"""
        
        # Count models per GPU for worker allocation
        active_gpus = sum(1 for models in allocation_plan.models_per_gpu.values() if models)
        total_models = sum(len(models) for models in allocation_plan.models_per_gpu.values())
        
        # CPU configuration
        cpu_cores = psutil.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Calculate optimal worker distribution
        gpu_workers = min(active_gpus, 3)  # Cap at 3 as per system design
        
        # Reserve more CPU cores for multi-model coordination
        reserved_cores = max(4, active_gpus * 2)
        available_cores = max(1, cpu_cores - reserved_cores)
        
        cpu_workers = max(8, available_cores // 2)
        preprocessing_workers = max(4, available_cores // 4)
        
        config = {
            'gpu_workers': gpu_workers,
            'cpu_workers': cpu_workers,
            'preprocessing_workers': preprocessing_workers,
            'active_gpus': active_gpus,
            'total_models': total_models,
            'gpu_batch_size': 24 if active_gpus > 1 else 32,  # Reduce batch size for multi-GPU
            'memory_limit_gb': min(8, available_memory_gb * 0.6),
            'max_queue_size': max(30, active_gpus * 15),
            'model_loading_strategy': 'sequential_per_gpu',
            'allocation_plan': allocation_plan
        }
        
        logger.info("⚙️ Optimized Worker Configuration:")
        logger.info(f"   🎮 GPU Workers: {gpu_workers} (across {active_gpus} active GPUs)")
        logger.info(f"   💪 CPU Workers: {cpu_workers}")
        logger.info(f"   🔄 Preprocessing: {preprocessing_workers}")
        logger.info(f"   📊 Models: {total_models} total across {active_gpus} GPUs")
        logger.info(f"   🧠 Memory Limit: {config['memory_limit_gb']:.1f}GB")
        
        return config

def get_multi_model_gpu_config(required_models: List[str] = None) -> Dict[str, Any]:
    """
    Get optimized multi-model GPU configuration for the terminology validation system
    
    Args:
        required_models: List of model names that need to be loaded
    
    Returns:
        Dictionary with optimized configuration
    """
    
    if required_models is None:
        # Default models used by the terminology validation system
        required_models = [
            'nllb-200-1.3B',      # Main translation model
            'gpt-4.1',            # Terminology and validation agent
            'dictionary_agent'     # Fast dictionary lookup
        ]
    
    try:
        manager = MultiModelGPUManager()
        allocation_plan = manager.create_optimal_allocation_plan(required_models)
        worker_config = manager.get_optimized_worker_config(allocation_plan)
        
        return {
            'success': True,
            'allocation_plan': allocation_plan,
            'worker_config': worker_config,
            'gpu_info': manager.gpu_info,
            'recommendations': _generate_recommendations(allocation_plan, manager.gpu_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to create multi-model GPU configuration: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_config': {
                'gpu_workers': 1,
                'cpu_workers': 8,
                'preprocessing_workers': 4,
                'gpu_batch_size': 32,
                'memory_limit_gb': 4
            }
        }

def _generate_recommendations(allocation_plan: MultiModelGPUConfig, gpu_info: Dict) -> List[str]:
    """Generate optimization recommendations based on allocation plan"""
    recommendations = []
    
    # Check for memory pressure
    for gpu_id, models in allocation_plan.models_per_gpu.items():
        if models and allocation_plan.gpu_memory_available[gpu_id] < 1.0:
            recommendations.append(
                f"GPU {gpu_id} has low memory remaining (<1GB). Consider reducing batch sizes or model precision."
            )
    
    # Check for load imbalance
    model_counts = [len(models) for models in allocation_plan.models_per_gpu.values()]
    if max(model_counts) - min(model_counts) > 1:
        recommendations.append(
            "GPU load is imbalanced. Consider redistributing models for better performance."
        )
    
    # Check for underutilized GPUs
    active_gpus = sum(1 for models in allocation_plan.models_per_gpu.values() if models)
    total_gpus = allocation_plan.total_gpus
    if active_gpus < total_gpus:
        recommendations.append(
            f"Only {active_gpus}/{total_gpus} GPUs are being used. Consider loading additional models or increasing batch sizes."
        )
    
    return recommendations

if __name__ == "__main__":
    # Test the multi-model GPU configuration
    config = get_multi_model_gpu_config()
    
    if config['success']:
        print("✅ Multi-Model GPU Configuration Generated Successfully")
        print(f"Active GPUs: {config['worker_config']['active_gpus']}")
        print(f"GPU Workers: {config['worker_config']['gpu_workers']}")
        print(f"Total Models: {config['worker_config']['total_models']}")
        
        if config['recommendations']:
            print("\n💡 Recommendations:")
            for rec in config['recommendations']:
                print(f"  • {rec}")
    else:
        print(f"❌ Failed to generate configuration: {config['error']}")
        print("Using fallback configuration")
