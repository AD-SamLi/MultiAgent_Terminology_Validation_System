#!/usr/bin/env python3
"""
Multi-Model GPU Manager
Enables loading multiple models on the same GPU for optimal resource utilization
"""

import torch
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of models that can be loaded"""
    TRANSLATION = "translation"      # NLLB translation models
    TERMINOLOGY = "terminology"      # GPT-based terminology agents
    VALIDATION = "validation"        # Validation agents
    DICTIONARY = "dictionary"        # Dictionary lookup models
    EMBEDDING = "embedding"          # Embedding models

@dataclass
class ModelSpec:
    """Specification for a model to be loaded"""
    name: str
    model_type: ModelType
    memory_requirement_gb: float
    priority: int  # 1=highest priority, 10=lowest
    can_share_gpu: bool = True
    precision: str = "float16"  # "float32", "float16", "int8"
    batch_size: int = 32
    
@dataclass
class GPUAllocation:
    """Represents models allocated to a specific GPU"""
    gpu_id: int
    gpu_name: str
    total_memory_gb: float
    available_memory_gb: float
    allocated_models: List[ModelSpec]
    memory_used_gb: float
    memory_utilization_percent: float

class MultiModelGPUManager:
    """Manages loading multiple models on the same GPU optimally"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_capabilities()
        self.model_registry = self._initialize_model_registry()
        self.allocations: List[GPUAllocation] = []
        
    def _detect_gpu_capabilities(self) -> Dict[int, Dict[str, Any]]:
        """Detect detailed GPU capabilities"""
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
            available_memory = total_memory - memory_reserved - memory_allocated - 0.5  # 0.5GB buffer
            
            gpu_info[gpu_id] = {
                'name': props.name,
                'total_memory_gb': total_memory,
                'available_memory_gb': max(0, available_memory),
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'compute_capability': f"{props.major}.{props.minor}",
                'max_threads_per_block': getattr(props, 'max_threads_per_block', 1024),
                'can_handle_multiple_models': available_memory > 4.0  # Need at least 4GB for multi-model
            }
            
            logger.info(f"   GPU {gpu_id}: {props.name}")
            logger.info(f"     Memory: {available_memory:.1f}GB available / {total_memory:.1f}GB total")
            logger.info(f"     Multi-model capable: {'Yes' if gpu_info[gpu_id]['can_handle_multiple_models'] else 'No'}")
        
        return gpu_info
    
    def _initialize_model_registry(self) -> Dict[str, ModelSpec]:
        """Initialize registry of known models with their specifications"""
        return {
            # Translation Models
            "nllb-200-1.3B": ModelSpec(
                name="facebook/nllb-200-1.3B",
                model_type=ModelType.TRANSLATION,
                memory_requirement_gb=3.5,
                priority=1,
                can_share_gpu=True,
                precision="float16",
                batch_size=32
            ),
            "nllb-200-3.3B": ModelSpec(
                name="facebook/nllb-200-3.3B", 
                model_type=ModelType.TRANSLATION,
                memory_requirement_gb=7.5,
                priority=1,
                can_share_gpu=True,
                precision="float16",
                batch_size=16
            ),
            
            # Terminology Models
            "gpt-4.1": ModelSpec(
                name="gpt-4.1",
                model_type=ModelType.TERMINOLOGY,
                memory_requirement_gb=2.0,  # API-based, minimal local memory
                priority=2,
                can_share_gpu=True,
                precision="float16",
                batch_size=8
            ),
            "terminology-agent": ModelSpec(
                name="terminology-agent",
                model_type=ModelType.TERMINOLOGY,
                memory_requirement_gb=1.5,
                priority=3,
                can_share_gpu=True,
                precision="float16",
                batch_size=16
            ),
            
            # Validation Models
            "validation-agent": ModelSpec(
                name="validation-agent",
                model_type=ModelType.VALIDATION,
                memory_requirement_gb=1.8,
                priority=4,
                can_share_gpu=True,
                precision="float16",
                batch_size=12
            ),
            
            # Dictionary Models
            "dictionary-agent": ModelSpec(
                name="dictionary-agent",
                model_type=ModelType.DICTIONARY,
                memory_requirement_gb=0.8,
                priority=5,
                can_share_gpu=True,
                precision="float16",
                batch_size=64
            ),
            
            # Embedding Models
            "sentence-transformer": ModelSpec(
                name="sentence-transformers/all-MiniLM-L6-v2",
                model_type=ModelType.EMBEDDING,
                memory_requirement_gb=1.2,
                priority=6,
                can_share_gpu=True,
                precision="float16",
                batch_size=128
            )
        }
    
    def add_custom_model(self, model_spec: ModelSpec):
        """Add a custom model to the registry"""
        self.model_registry[model_spec.name] = model_spec
        logger.info(f"📝 Added custom model: {model_spec.name} ({model_spec.memory_requirement_gb:.1f}GB)")
    
    def create_optimal_allocation(self, required_models: List[str]) -> List[GPUAllocation]:
        """Create optimal allocation of models across available GPUs"""
        
        if not self.gpu_info:
            logger.error("No GPUs available for model allocation")
            return []
        
        # Get model specs for required models
        models_to_allocate = []
        for model_name in required_models:
            if model_name in self.model_registry:
                models_to_allocate.append(self.model_registry[model_name])
            else:
                logger.warning(f"Unknown model: {model_name}, creating default spec")
                # Create default spec for unknown models
                default_spec = ModelSpec(
                    name=model_name,
                    model_type=ModelType.TERMINOLOGY,
                    memory_requirement_gb=2.0,
                    priority=5,
                    can_share_gpu=True,
                    precision="float16",
                    batch_size=16
                )
                models_to_allocate.append(default_spec)
        
        # Sort models by priority (highest first)
        models_to_allocate.sort(key=lambda x: x.priority)
        
        # Initialize GPU allocations
        allocations = []
        for gpu_id, gpu_data in self.gpu_info.items():
            allocation = GPUAllocation(
                gpu_id=gpu_id,
                gpu_name=gpu_data['name'],
                total_memory_gb=gpu_data['total_memory_gb'],
                available_memory_gb=gpu_data['available_memory_gb'],
                allocated_models=[],
                memory_used_gb=0.0,
                memory_utilization_percent=0.0
            )
            allocations.append(allocation)
        
        # Allocate models using intelligent placement strategy
        self._allocate_models_intelligently(models_to_allocate, allocations)
        
        # Update memory utilization
        for allocation in allocations:
            allocation.memory_used_gb = sum(model.memory_requirement_gb for model in allocation.allocated_models)
            allocation.memory_utilization_percent = (allocation.memory_used_gb / allocation.total_memory_gb) * 100
        
        self.allocations = allocations
        self._log_allocation_plan(allocations)
        
        return allocations
    
    def _allocate_models_intelligently(self, models: List[ModelSpec], allocations: List[GPUAllocation]):
        """Intelligently allocate models to GPUs using multiple strategies"""
        
        # Strategy 1: Try to fit multiple models on high-memory GPUs
        for model in models:
            allocated = False
            
            # First, try to place on GPU with existing compatible models
            for allocation in sorted(allocations, key=lambda x: x.available_memory_gb, reverse=True):
                if self._can_fit_model(model, allocation):
                    # Check compatibility with existing models
                    if self._is_compatible_placement(model, allocation):
                        allocation.allocated_models.append(model)
                        allocation.available_memory_gb -= model.memory_requirement_gb
                        allocated = True
                        logger.info(f"🎯 Allocated {model.name} to GPU {allocation.gpu_id} (multi-model)")
                        break
            
            # If not allocated, try to place on GPU with most available memory
            if not allocated:
                best_gpu = max(allocations, key=lambda x: x.available_memory_gb)
                if self._can_fit_model(model, best_gpu):
                    best_gpu.allocated_models.append(model)
                    best_gpu.available_memory_gb -= model.memory_requirement_gb
                    allocated = True
                    logger.info(f"🎯 Allocated {model.name} to GPU {best_gpu.gpu_id} (best fit)")
            
            if not allocated:
                logger.warning(f"⚠️ Could not allocate {model.name} - insufficient GPU memory")
    
    def _can_fit_model(self, model: ModelSpec, allocation: GPUAllocation) -> bool:
        """Check if model can fit on the GPU allocation"""
        # Add 10% buffer for safety
        required_memory = model.memory_requirement_gb * 1.1
        return allocation.available_memory_gb >= required_memory
    
    def _is_compatible_placement(self, model: ModelSpec, allocation: GPUAllocation) -> bool:
        """Check if model is compatible with existing models on GPU"""
        if not allocation.allocated_models:
            return True  # Empty GPU is always compatible
        
        if not model.can_share_gpu:
            return False  # Model cannot share GPU
        
        # Check if existing models can share
        for existing_model in allocation.allocated_models:
            if not existing_model.can_share_gpu:
                return False
        
        # Check memory pressure - don't exceed 90% utilization
        total_memory_needed = sum(m.memory_requirement_gb for m in allocation.allocated_models) + model.memory_requirement_gb
        utilization = (total_memory_needed / allocation.total_memory_gb) * 100
        
        if utilization > 90:
            return False
        
        # Check model type compatibility (avoid conflicts)
        existing_types = {m.model_type for m in allocation.allocated_models}
        
        # Translation models can share with anything except other translation models
        if model.model_type == ModelType.TRANSLATION and ModelType.TRANSLATION in existing_types:
            return False
        
        return True
    
    def get_gpu_assignment(self, model_name: str) -> Optional[int]:
        """Get the GPU ID where a specific model is allocated"""
        for allocation in self.allocations:
            for model in allocation.allocated_models:
                if model.name == model_name or model_name in model.name:
                    return allocation.gpu_id
        return None
    
    def get_models_on_gpu(self, gpu_id: int) -> List[ModelSpec]:
        """Get all models allocated to a specific GPU"""
        for allocation in self.allocations:
            if allocation.gpu_id == gpu_id:
                return allocation.allocated_models
        return []
    
    def get_memory_utilization(self, gpu_id: int) -> float:
        """Get memory utilization percentage for a specific GPU"""
        for allocation in self.allocations:
            if allocation.gpu_id == gpu_id:
                return allocation.memory_utilization_percent
        return 0.0
    
    def can_load_additional_model(self, model_name: str, gpu_id: int) -> bool:
        """Check if an additional model can be loaded on a specific GPU"""
        if model_name not in self.model_registry:
            return False
        
        model = self.model_registry[model_name]
        for allocation in self.allocations:
            if allocation.gpu_id == gpu_id:
                return self._can_fit_model(model, allocation) and self._is_compatible_placement(model, allocation)
        
        return False
    
    def optimize_batch_sizes(self) -> Dict[str, int]:
        """Optimize batch sizes based on GPU memory utilization"""
        optimized_batches = {}
        
        for allocation in self.allocations:
            # Calculate available memory for batch processing
            available_for_batches = allocation.available_memory_gb
            
            for model in allocation.allocated_models:
                # Scale batch size based on available memory
                memory_factor = min(2.0, available_for_batches / len(allocation.allocated_models))
                optimized_batch = max(8, int(model.batch_size * memory_factor))
                
                # Reduce batch size for high utilization GPUs
                if allocation.memory_utilization_percent > 80:
                    optimized_batch = int(optimized_batch * 0.7)
                elif allocation.memory_utilization_percent > 60:
                    optimized_batch = int(optimized_batch * 0.85)
                
                optimized_batches[model.name] = optimized_batch
        
        return optimized_batches
    
    def _log_allocation_plan(self, allocations: List[GPUAllocation]):
        """Log the final allocation plan"""
        logger.info("🎯 MULTI-MODEL GPU ALLOCATION PLAN:")
        
        total_models = sum(len(alloc.allocated_models) for alloc in allocations)
        active_gpus = sum(1 for alloc in allocations if alloc.allocated_models)
        
        logger.info(f"   📊 Total Models: {total_models}")
        logger.info(f"   🎮 Active GPUs: {active_gpus}/{len(allocations)}")
        
        for allocation in allocations:
            if allocation.allocated_models:
                logger.info(f"   GPU {allocation.gpu_id} ({allocation.gpu_name}):")
                logger.info(f"     Models: {len(allocation.allocated_models)}")
                logger.info(f"     Memory: {allocation.memory_used_gb:.1f}GB/{allocation.total_memory_gb:.1f}GB ({allocation.memory_utilization_percent:.0f}%)")
                
                for model in allocation.allocated_models:
                    logger.info(f"       • {model.name} ({model.model_type.value}, {model.memory_requirement_gb:.1f}GB)")
    
    def export_allocation_config(self, filepath: str = None) -> str:
        """Export allocation configuration to JSON file"""
        if filepath is None:
            filepath = "multi_model_gpu_allocation.json"
        
        config_data = {
            'gpu_count': len(self.gpu_info),
            'total_models_allocated': sum(len(alloc.allocated_models) for alloc in self.allocations),
            'allocations': []
        }
        
        for allocation in self.allocations:
            alloc_data = {
                'gpu_id': allocation.gpu_id,
                'gpu_name': allocation.gpu_name,
                'memory_utilization_percent': allocation.memory_utilization_percent,
                'models': []
            }
            
            for model in allocation.allocated_models:
                model_data = {
                    'name': model.name,
                    'type': model.model_type.value,
                    'memory_gb': model.memory_requirement_gb,
                    'priority': model.priority,
                    'precision': model.precision,
                    'batch_size': model.batch_size
                }
                alloc_data['models'].append(model_data)
            
            config_data['allocations'].append(alloc_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Multi-model allocation config saved to: {filepath}")
        return filepath

def create_multi_model_allocation(required_models: List[str]) -> Tuple[MultiModelGPUManager, List[GPUAllocation]]:
    """
    Create optimal multi-model GPU allocation
    
    Args:
        required_models: List of model names to allocate
    
    Returns:
        Tuple of (MultiModelGPUManager, allocations)
    """
    manager = MultiModelGPUManager()
    allocations = manager.create_optimal_allocation(required_models)
    
    return manager, allocations

if __name__ == "__main__":
    # Test multi-model GPU allocation
    print("🔍 Testing Multi-Model GPU Allocation...")
    
    # Example models for terminology validation system
    test_models = [
        "nllb-200-1.3B",        # Translation model
        "gpt-4.1",              # Terminology agent
        "validation-agent",      # Validation agent
        "dictionary-agent",      # Dictionary lookup
        "sentence-transformer"   # Embedding model
    ]
    
    try:
        manager, allocations = create_multi_model_allocation(test_models)
        
        print(f"✅ Successfully created allocation for {len(test_models)} models")
        
        # Show allocation summary
        for allocation in allocations:
            if allocation.allocated_models:
                print(f"🎮 GPU {allocation.gpu_id}: {len(allocation.allocated_models)} models, {allocation.memory_utilization_percent:.0f}% utilization")
        
        # Export configuration
        config_file = manager.export_allocation_config()
        print(f"📄 Configuration saved to: {config_file}")
        
        # Test batch size optimization
        optimized_batches = manager.optimize_batch_sizes()
        print(f"🚀 Optimized batch sizes: {len(optimized_batches)} models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
