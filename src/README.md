# Source Code (`src/`)

This directory contains all the modular source code for the Terms Verification System.

## 📁 Directory Structure

### 🤖 `agents/`
AI agents powered by Azure OpenAI that perform intelligent analysis:
- **`terminology_agent.py`** - Step 2: Glossary validation agent
- **`modern_terminology_review_agent.py`** - Step 7: Final review and decision agent
- **`fast_dictionary_agent.py`** - Step 3: Dictionary analysis agent

### 📊 `processors/`
Data processing modules for term extraction and analysis:
- **`direct_unified_processor.py`** - Step 1: Term collection from raw data
- **`frequency_storage.py`** - Step 4: Frequency analysis and storage
- **`step7_fixed_batch_processing.py`** - Step 7: Batch processing helpers

### 🌐 `translation/`
Translation system components using NLLB models:
- **`ultra_optimized_smart_runner.py`** - Step 5: Main translation orchestrator
- **`nllb_translation_tool.py`** - NLLB model wrapper and interface
- **`dynamic_worker_manager.py`** - Worker management for GPU/CPU coordination

### ✅ `validation/`
Validation and quality assurance modules:
- **`modern_parallel_validation.py`** - Step 7: Parallel validation system

### ⚙️ `config/`
Configuration and hardware management:
- **`adaptive_system_config.py`** - Primary adaptive configuration
- **`multi_model_gpu_config.py`** - Multi-GPU configuration
- **`multi_model_gpu_manager.py`** - GPU resource management
- **`optimized_translation_config.py`** - Translation-specific configuration

### 🛠️ `tools/`
Utility tools and helpers:
- **`terminology_tool.py`** - Glossary file management and access
- **`atomic_json_utils.py`** - Thread-safe JSON operations
- **`auth_fix_wrapper.py`** - Azure authentication wrapper

### 🔧 `utils/`
General utility functions:
- **`convert_prdsmrt_to_system_format.py`** - Format conversion utilities

## 🔄 Module Dependencies

```
agentic_terminology_validation_system.py (ROOT)
    ↓
    ├── agents/
    │   ├── terminology_agent.py
    │   ├── modern_terminology_review_agent.py
    │   └── fast_dictionary_agent.py
    │
    ├── processors/
    │   ├── direct_unified_processor.py
    │   ├── frequency_storage.py
    │   └── step7_fixed_batch_processing.py
    │
    ├── translation/
    │   ├── ultra_optimized_smart_runner.py
    │   ├── nllb_translation_tool.py
    │   └── dynamic_worker_manager.py
    │
    ├── validation/
    │   └── modern_parallel_validation.py
    │
    ├── config/
    │   ├── adaptive_system_config.py
    │   ├── multi_model_gpu_config.py
    │   ├── multi_model_gpu_manager.py
    │   └── optimized_translation_config.py
    │
    └── tools/
        ├── terminology_tool.py
        ├── atomic_json_utils.py
        └── auth_fix_wrapper.py
```

## 📝 Import Convention

All imports use the `src.` prefix:

```python
# From main file
from src.agents.terminology_agent import TerminologyAgent
from src.translation.ultra_optimized_smart_runner import UltraOptimizedSmartRunner

# Between src modules
from src.tools.terminology_tool import TerminologyTool
from src.config.adaptive_system_config import get_adaptive_system_config
```

## 🏗️ Architecture Principles

1. **Separation of Concerns** - Each module has a specific responsibility
2. **Modularity** - Components can be updated independently
3. **Clear Dependencies** - Import paths show relationships
4. **Professional Structure** - Industry-standard organization
5. **Maintainability** - Easy to navigate and extend



