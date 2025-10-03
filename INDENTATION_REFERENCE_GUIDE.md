# 🔧 Indentation Reference Guide - Multi-Agent Terminology Validation System

## 📋 Critical Indentation Rules

### **RULE 1: Consistent 4-Space Indentation**
- **ALWAYS** use 4 spaces for each indentation level
- **NEVER** mix tabs and spaces
- **NEVER** use 2 spaces or 8 spaces

### **RULE 2: Try-Except Block Patterns**
```python
# ✅ CORRECT PATTERN
try:
    # 4 spaces from try
    some_operation()
    if condition:
        # 8 spaces from try
        nested_operation()
except ImportError:
    # 4 spaces from try (same level as try content)
    try:
        # 8 spaces from original try (nested try)
        fallback_operation()
    except ImportError:
        # 8 spaces from original try (nested except)
        final_fallback()

# ❌ WRONG PATTERN (causes "Expected indented block" error)
try:
    some_operation()
except ImportError:
    # Fallback to original optimized configuration
try:  # ❌ This should be indented 4 more spaces
    fallback_operation()
```

### **RULE 3: Class and Function Definitions**
```python
# ✅ CORRECT
class MyClass:
    """Class docstring"""
    
    def __init__(self):
        # 8 spaces from class
        self.value = None
        
    def method(self):
        # 8 spaces from class
        if condition:
            # 12 spaces from class
            return True
        else:
            # 12 spaces from class
            return False
```

### **RULE 4: Dictionary and List Definitions**
```python
# ✅ CORRECT - Multi-line dictionary
config = {
    'model_size': "1.3B",
    'gpu_workers': 1,
    'cpu_workers': 8,
    'nested_config': {
        'batch_size': 32,
        'queue_size': 50
    }
}

# ✅ CORRECT - Function call with parameters
ultra_config = UltraOptimizedConfig(
    model_size="1.3B",
    gpu_workers=1,  # Conservative default
    cpu_workers=8,
    gpu_batch_size=32,
    max_queue_size=50,
    predictive_caching=True,
    dynamic_batching=True,
    async_checkpointing=True,
    memory_mapping=False  # Conservative for memory
)
```

### **RULE 5: Conditional Statements**
```python
# ✅ CORRECT
if condition:
    # 4 spaces
    action1()
    if nested_condition:
        # 8 spaces
        nested_action()
elif other_condition:
    # 4 spaces (same level as if)
    action2()
else:
    # 4 spaces (same level as if)
    action3()
```

## 🚨 Common Error Patterns to Avoid

### **ERROR 1: Missing Indentation After Colon**
```python
# ❌ WRONG - Will cause "Expected indented block"
if condition:
do_something()  # Should be indented

# ✅ CORRECT
if condition:
    do_something()  # Properly indented
```

### **ERROR 2: Inconsistent Indentation in Try-Except**
```python
# ❌ WRONG - Mixed indentation levels
try:
    operation()
except Exception:
try:  # Should be indented
    fallback()

# ✅ CORRECT
try:
    operation()
except Exception:
    try:  # Properly indented
        fallback()
```

### **ERROR 3: Wrong Indentation in Function Parameters**
```python
# ❌ WRONG - Parameters not aligned
function_call(
param1="value1",
    param2="value2"  # Inconsistent indentation
)

# ✅ CORRECT - Consistent parameter indentation
function_call(
    param1="value1",
    param2="value2"
)
```

## 🔍 Indentation Validation Checklist

Before committing code changes, verify:

1. **[ ]** All lines after `:` are indented by exactly 4 spaces
2. **[ ]** Try-except blocks have consistent indentation
3. **[ ]** Function parameters in multi-line calls are aligned
4. **[ ]** Dictionary entries are consistently indented
5. **[ ]** No mixing of tabs and spaces
6. **[ ]** Files compile without syntax errors: `python -m py_compile filename.py`

## 🛠️ Quick Fix Commands

```bash
# Test file compilation
python -m py_compile agentic_terminology_validation_system.py
python -m py_compile ultra_optimized_smart_runner.py

# Check for indentation issues with linter
# (Use your IDE's linter or external tools)
```

## 📝 Specific Fixes Applied (October 2025)

### **Critical Syntax Errors Fixed**

#### **1. Try-Except Block Indentation (agentic_terminology_validation_system.py)**
```python
# ❌ BEFORE (caused "Expected indented block" error)
except ImportError:
    # Fallback to original optimized configuration
try:  # Wrong indentation level

# ✅ AFTER (properly indented)
except ImportError:
    # Fallback to original optimized configuration
    try:  # Correctly indented 4 spaces from except
```

#### **2. Elif After Else Error (agentic_terminology_validation_system.py)**
```python
# ❌ BEFORE (caused "Expected expression" error)
else:
    step_files[file] = file_path

# Special handling for Step 7
elif step == 7 and file == 'Final_Terminology_Decisions.json':  # ❌ elif after else

# ✅ AFTER (properly structured)
# Special handling for Step 7
elif step == 7 and file == 'Final_Terminology_Decisions.json':
    # ... handling code ...
else:
    step_files[file] = file_path
```

#### **3. Misaligned Indentation (ultra_optimized_smart_runner.py)**
```python
# ❌ BEFORE (inconsistent indentation)
if config.gpu_workers > 1:
    print(f"🎮 Multi-Model Single GPU mode")
else:
print(f"🎮 Single GPU mode")  # ❌ Missing indentation
config.gpu_workers = 1        # ❌ Wrong indentation level

# ✅ AFTER (consistent 4-space indentation)
if config.gpu_workers > 1:
    print(f"🎮 Multi-Model Single GPU mode")
else:
    print(f"🎮 Single GPU mode")  # ✅ Properly indented
    config.gpu_workers = 1        # ✅ Consistent indentation
```

#### **4. Function Parameter Indentation (agentic_terminology_validation_system.py)**
```python
# ❌ BEFORE (inconsistent parameter indentation)
ultra_config = UltraOptimizedConfig(
model_size="1.3B",           # ❌ No indentation
    gpu_workers=1,           # ❌ Mixed indentation
gpu_batch_size=32,           # ❌ No indentation
)

# ✅ AFTER (consistent 4-space indentation)
ultra_config = UltraOptimizedConfig(
    model_size="1.3B",       # ✅ Consistent 4 spaces
    gpu_workers=1,           # ✅ Consistent 4 spaces
    gpu_batch_size=32,       # ✅ Consistent 4 spaces
)
```

### **Lessons Learned**

1. **Always check elif placement**: `elif` cannot follow `else` - restructure the conditional logic
2. **Nested try-except requires extra indentation**: Each nested level adds 4 more spaces
3. **Function parameters must be consistently indented**: All parameters should align at the same level
4. **Test compilation frequently**: Use `python -m py_compile filename.py` to catch syntax errors early
5. **Use IDE indentation guides**: Visual guides help maintain consistent spacing

## 🎯 Best Practices

1. **Use a consistent IDE/editor** with Python indentation settings
2. **Enable "Show whitespace"** to visualize spaces and tabs
3. **Set tab width to 4 spaces** and convert tabs to spaces
4. **Use automatic code formatters** like `black` or `autopep8`
5. **Test compilation frequently** during development
6. **Review indentation** in pull requests and code reviews

## 🔧 IDE Configuration Recommendations

### **VS Code**
```json
{
    "python.defaultInterpreterPath": "python",
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.detectIndentation": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

### **PyCharm**
- File → Settings → Editor → Code Style → Python
- Set "Tab size" to 4
- Set "Indent" to 4
- Check "Use tab character" = False

---

**Remember**: Indentation errors are the most common cause of Python syntax errors. Following these patterns will prevent 99% of indentation-related issues in the Multi-Agent Terminology Validation System.

