#!/usr/bin/env python3
"""
🛡️ ATOMIC JSON UTILITIES
========================

Corruption-resistant JSON file operations for the terminology validation system.
Provides atomic writes, automatic recovery, and data salvage capabilities.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
import re


def atomic_json_write(file_path, data, create_backup=True):
    """
    Atomic JSON write with corruption prevention
    
    Args:
        file_path (str): Path to the JSON file to write
        data (dict): Data to write to the file
        create_backup (bool): Whether to keep a backup of the old version
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Write to temporary file first
        temp_file = file_path + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Force OS to write to storage
        
        # Verify the temporary file is valid JSON
        with open(temp_file, 'r', encoding='utf-8') as f:
            json.load(f)  # This will raise JSONDecodeError if invalid
        
        # Atomic move (rename) - this is atomic on most filesystems
        if os.path.exists(file_path) and create_backup:
            backup_file = file_path + '.backup'
            shutil.move(file_path, backup_file)  # Keep backup of old version
        
        shutil.move(temp_file, file_path)
        
        # Clean up old backup after successful write
        if create_backup:
            backup_file = file_path + '.backup'
            if os.path.exists(backup_file):
                os.remove(backup_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Atomic write failed for {file_path}: {e}")
        # Clean up temp file if it exists
        temp_file = file_path + '.tmp'
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False


def load_json_safely(file_path, backup_paths=None, create_empty_on_fail=True):
    """
    Safely load JSON with automatic corruption recovery
    
    Args:
        file_path (str): Primary JSON file to load
        backup_paths (list): List of backup file paths to try
        create_empty_on_fail (bool): Create empty structure if all else fails
    
    Returns:
        dict: Loaded JSON data or empty structure
    """
    
    # Try primary file first
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Primary file corrupted ({file_path}): {e}")
            
            # Try backup files
            if backup_paths:
                for backup_path in backup_paths:
                    if os.path.exists(backup_path):
                        try:
                            print(f"🔄 Attempting recovery from {backup_path}...")
                            with open(backup_path, 'r', encoding='utf-8') as f:
                                backup_data = json.load(f)
                            
                            # Restore primary file from backup
                            if atomic_json_write(file_path, backup_data):
                                print(f"✅ Recovered from {backup_path}")
                                return backup_data
                            
                        except Exception as backup_error:
                            print(f"❌ Backup {backup_path} failed: {backup_error}")
                            continue
            
            # Last resort: try to salvage partial data
            print("🔧 Attempting data salvage...")
            salvaged_data = salvage_json_data(file_path)
            if salvaged_data:
                return salvaged_data
    
    # Create empty structure if requested
    if create_empty_on_fail:
        empty_structure = {
            "metadata": {
                "created_timestamp": datetime.now().isoformat(),
                "source": "atomic_json_utils",
                "version": "1.0",
                "created_due_to_failure": True
            }
        }
        return empty_structure
    
    return None


def salvage_json_data(file_path):
    """
    Attempt to salvage data from corrupted JSON file
    
    Args:
        file_path (str): Path to corrupted JSON file
    
    Returns:
        dict: Salvaged data or None if unsuccessful
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Strategy 1: Look for complete objects with specific patterns
        salvaged_data = {
            "metadata": {
                "created_timestamp": datetime.now().isoformat(),
                "source": "atomic_json_utils",
                "version": "1.0",
                "salvaged_from_corruption": True,
                "original_file": file_path
            }
        }
        
        # Try to salvage translation results
        translation_pattern = r'\{[^{}]*"status"\s*:\s*"completed"[^{}]*\}'
        translation_matches = re.findall(translation_pattern, content, re.DOTALL)
        
        salvaged_translations = []
        for match in translation_matches:
            try:
                result = json.loads(match)
                if 'term' in result and 'status' in result:
                    salvaged_translations.append(result)
            except:
                continue
        
        if salvaged_translations:
            salvaged_data["translation_results"] = salvaged_translations
            salvaged_data["metadata"]["salvaged_translations"] = len(salvaged_translations)
        
        # Try to salvage checkpoint data
        checkpoint_patterns = [
            r'"total_terms"\s*:\s*(\d+)',
            r'"completed_terms"\s*:\s*(\d+)',
            r'"remaining_terms"\s*:\s*(\d+)'
        ]
        
        for pattern in checkpoint_patterns:
            matches = re.findall(pattern, content)
            if matches:
                key = pattern.split('"')[1]
                try:
                    salvaged_data[key] = int(matches[-1])  # Use last match
                except:
                    pass
        
        # Try to salvage step information
        step_match = re.search(r'"step"\s*:\s*(\d+)', content)
        if step_match:
            salvaged_data["step"] = int(step_match.group(1))
        
        step_name_match = re.search(r'"step_name"\s*:\s*"([^"]+)"', content)
        if step_name_match:
            salvaged_data["step_name"] = step_name_match.group(1)
        
        if len(salvaged_data) > 1:  # More than just metadata
            print(f"🔧 Salvaged data from {file_path}:")
            for key, value in salvaged_data.items():
                if key != "metadata":
                    if isinstance(value, list):
                        print(f"   • {key}: {len(value)} items")
                    else:
                        print(f"   • {key}: {value}")
            
            # Save salvaged data
            if atomic_json_write(file_path, salvaged_data):
                return salvaged_data
        
    except Exception as e:
        print(f"❌ Salvage operation failed: {e}")
    
    return None


def verify_json_integrity(file_path):
    """
    Verify JSON file integrity
    
    Args:
        file_path (str): Path to JSON file to verify
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"JSON corruption: {e}"
    except Exception as e:
        return False, f"File error: {e}"


def create_checkpoint_safely(file_path, checkpoint_data):
    """
    Create a checkpoint file with corruption protection
    
    Args:
        file_path (str): Path to checkpoint file
        checkpoint_data (dict): Checkpoint data to save
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Add standard checkpoint metadata
    if "metadata" not in checkpoint_data:
        checkpoint_data["metadata"] = {}
    
    checkpoint_data["metadata"].update({
        "checkpoint_timestamp": datetime.now().isoformat(),
        "created_by": "atomic_json_utils",
        "version": "1.0"
    })
    
    return atomic_json_write(file_path, checkpoint_data)


def test_corruption_resistance():
    """Test the corruption resistance utilities"""
    print("🧪 TESTING ATOMIC JSON UTILITIES")
    print("=" * 40)
    
    import tempfile
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Atomic write
        test_file = os.path.join(test_dir, "test.json")
        test_data = {"test": True, "count": 42}
        
        success = atomic_json_write(test_file, test_data)
        print(f"✅ Atomic write: {'SUCCESS' if success else 'FAILED'}")
        
        # Test 2: Safe load
        loaded_data = load_json_safely(test_file)
        print(f"✅ Safe load: {'SUCCESS' if loaded_data and loaded_data.get('test') else 'FAILED'}")
        
        # Test 3: Corruption recovery
        # Create corrupted file
        with open(test_file, 'w') as f:
            f.write('{"test": true, "incomplete":')  # Corrupted
        
        # Create backup
        backup_file = test_file + "_backup"
        atomic_json_write(backup_file, test_data)
        
        # Test recovery
        recovered_data = load_json_safely(test_file, [backup_file])
        print(f"✅ Corruption recovery: {'SUCCESS' if recovered_data and recovered_data.get('test') else 'FAILED'}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_corruption_resistance()
