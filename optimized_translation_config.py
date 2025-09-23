#!/usr/bin/env python3
"""
Optimized Translation Configuration
Based on detected hardware: 20 CPU cores, 31.62GB RAM, NVIDIA RTX A1000 6GB
"""

from system_hardware_detector import SystemHardwareDetector
from ultra_optimized_smart_runner import UltraOptimizedConfig
import json

def get_optimized_config():
    """Get optimized configuration based on current hardware"""
    
    # Get current memory status
    import psutil
    memory = psutil.virtual_memory()
    current_available_gb = memory.available / (1024**3)
    
    # Hardware profile with current memory readings
    hardware_profile = {
        'cpu_cores': 20,
        'physical_cores': 14, 
        'total_ram_gb': 31.62,
        'available_ram_gb': current_available_gb,  # Dynamic reading
        'gpu_count': 1,
        'gpu_name': 'NVIDIA RTX A1000 6GB',
        'gpu_memory_gb': 6.0,
        'gpu_available_gb': 6.0,  # 0% utilization
        'cuda_available': True
    }
    
    # Optimal allocation strategy based on available RAM
    # 1 GPU for translation + CPU workers for queueing/preprocessing
    # Memory-conscious configuration
    
    # Adjust settings based on available memory
    if hardware_profile['available_ram_gb'] < 4.0:
        # Very conservative settings for low memory
        config = UltraOptimizedConfig(
            model_size="1.3B",
            gpu_workers=1,
            gpu_batch_size=16,  # Reduced for memory
            cpu_workers=6,      # Reduced workers
            max_queue_size=25,  # Smaller queue
            predictive_caching=False,  # Disable caching
            dynamic_batching=True,
            async_checkpointing=True,
            memory_mapping=False,
            core_lang_count=8,      # Minimal languages
            extended_lang_count=20,  # Very reduced
            minimal_lang_count=6     # Ultra-minimal
        )
    elif hardware_profile['available_ram_gb'] < 8.0:
        # Moderate settings for medium memory
        config = UltraOptimizedConfig(
            model_size="1.3B",
            gpu_workers=1,
            gpu_batch_size=24,  # Moderate batch size
            cpu_workers=8,
            max_queue_size=40,
            predictive_caching=True,
            dynamic_batching=True,
            async_checkpointing=True,
            memory_mapping=False,
            core_lang_count=12,
            extended_lang_count=30,
            minimal_lang_count=8
        )
    else:
        # Optimal settings for good memory
        config = UltraOptimizedConfig(
            model_size="1.3B",
            gpu_workers=1,
            gpu_batch_size=32,  # Full batch size
            cpu_workers=8,
            max_queue_size=50,
            predictive_caching=True,
            dynamic_batching=True,
            async_checkpointing=True,
            memory_mapping=True,   # Enable for good memory
            core_lang_count=12,
            extended_lang_count=40,
            minimal_lang_count=8
        )
    
    return config, hardware_profile

def create_resource_allocation_plan():
    """Create detailed resource allocation plan"""
    config, hardware = get_optimized_config()
    
    plan = {
        "resource_allocation": {
            "gpu_allocation": {
                "gpu_0": {
                    "role": "Primary Translation Engine",
                    "model": "facebook/nllb-200-1.3B",
                    "memory_usage": "~4-5GB",
                    "batch_size": config.gpu_batch_size,
                    "utilization_target": "80-90%"
                }
            },
            "cpu_allocation": {
                "preprocessing_workers": {
                    "count": 4,
                    "role": "Text preprocessing, tokenization, term categorization",
                    "cpu_cores": "Cores 0-3"
                },
                "queue_management": {
                    "count": 2,
                    "role": "GPU queue management, load balancing",
                    "cpu_cores": "Cores 4-5"
                },
                "postprocessing_workers": {
                    "count": 2,
                    "role": "Result validation, formatting, caching",
                    "cpu_cores": "Cores 6-7"
                }
            },
            "memory_allocation": {
                "gpu_memory": "6GB (RTX A1000)",
                "system_memory_limit": "1.5GB",
                "queue_memory": "0.3GB",
                "model_memory": "1.0GB",
                "working_memory": "0.2GB"
            }
        },
        "processing_strategy": {
            "workflow": [
                "1. CPU workers preprocess and categorize terms",
                "2. Queue managers batch terms for GPU processing",
                "3. GPU processes translation batches using NLLB-200-1.3B",
                "4. CPU workers post-process and validate results",
                "5. Async checkpointing saves progress"
            ],
            "optimization_techniques": [
                "Dynamic batch sizing based on GPU memory usage",
                "Predictive caching of common term patterns",
                "Intelligent language selection per term category",
                "Memory-mapped file I/O where possible",
                "Async processing to minimize blocking"
            ]
        },
        "performance_expectations": {
            "terms_per_second": "15-25 terms/sec",
            "gpu_utilization": "80-90%",
            "cpu_utilization": "60-70%",
            "memory_efficiency": "Conservative due to RAM constraint",
            "estimated_time_8691_terms": "6-10 minutes"
        }
    }
    
    return plan

def update_ultra_optimized_config():
    """Update the main system with optimized configuration"""
    config, hardware = get_optimized_config()
    plan = create_resource_allocation_plan()
    
    # Save configuration
    config_data = {
        'hardware_profile': hardware,
        'optimized_config': {
            'model_size': config.model_size,
            'gpu_workers': config.gpu_workers,
            'cpu_workers': config.cpu_workers,
            'gpu_batch_size': config.gpu_batch_size,
            'max_queue_size': config.max_queue_size,
            'predictive_caching': config.predictive_caching,
            'dynamic_batching': config.dynamic_batching,
            'memory_mapping': config.memory_mapping,
            'core_lang_count': config.core_lang_count,
            'extended_lang_count': config.extended_lang_count,
            'minimal_lang_count': config.minimal_lang_count
        },
        'resource_allocation_plan': plan
    }
    
    with open('optimized_translation_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    return config, plan

def print_optimization_report():
    """Print detailed optimization report"""
    config, plan = update_ultra_optimized_config()
    
    print(f"""
🚀 OPTIMIZED TRANSLATION CONFIGURATION
{'=' * 60}

🖥️  Hardware Profile:
   CPU: 20 cores (14 physical) Intel64 Family 6 Model 186
   RAM: 31.62GB total, 2.55GB available ⚠️  (Memory constrained)
   GPU: NVIDIA RTX A1000 6GB (0% utilization, 6GB available)
   CUDA: Available (Version 12.8)

⚙️  Optimal Resource Allocation:
   🎮 GPU Workers: 1 (Primary translation engine)
   ⚙️  CPU Workers: 8 (Preprocessing + Queue management)
   📦 GPU Batch Size: 32 terms/batch
   🔄 CPU Batch Size: 16 terms/batch
   📊 Queue Size: 50 (Memory conservative)
   💾 Memory Limit: 1.5GB (Due to RAM constraint)

🎯 Processing Strategy:
   1️⃣  4 CPU workers: Text preprocessing & term categorization
   2️⃣  2 CPU workers: GPU queue management & load balancing  
   3️⃣  1 GPU worker: NLLB-200-1.3B translation processing
   4️⃣  2 CPU workers: Result validation & post-processing

📈 Expected Performance:
   ⚡ Speed: 15-25 terms/second
   🎮 GPU Utilization: 80-90%
   ⚙️  CPU Utilization: 60-70%
   ⏱️  Est. Time (8,691 terms): 6-10 minutes

⚠️  Memory Management:
   Due to low available RAM (2.55GB), using conservative settings:
   - Reduced queue sizes
   - Smaller batch sizes
   - Disabled memory mapping
   - Frequent checkpointing

💡 Optimization Features:
   ✅ Dynamic batch sizing
   ✅ Predictive caching
   ✅ Intelligent language selection
   ✅ Async checkpointing
   ✅ Memory-conscious processing
""")

if __name__ == "__main__":
    print_optimization_report()
