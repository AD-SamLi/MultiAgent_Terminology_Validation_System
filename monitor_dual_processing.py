#!/usr/bin/env python3
"""
Monitor the dual processing of dictionary and non-dictionary terms.
"""

import os
import glob
import json
from datetime import datetime

def monitor_dual_processing():
    """Monitor both dictionary and non-dictionary term processing"""
    
    print("📊 DUAL PROCESSING MONITOR")
    print("=" * 60)
    print(f"⏰ Status check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dictionary validation files
    dict_files = glob.glob("dictionary_validation_batch_*.json")
    dict_files.sort()
    
    # Check non-dictionary validation files
    non_dict_files = glob.glob("non_dictionary_validation_batch_*.json")
    non_dict_files.sort()
    
    print("📖 DICTIONARY TERMS PROCESSING:")
    print("-" * 40)
    
    if dict_files:
        dict_processed = 0
        dict_statuses = {}
        
        for file in dict_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                dict_processed += len(results)
                
                for result in results:
                    status = result.get('status', 'unknown')
                    dict_statuses[status] = dict_statuses.get(status, 0) + 1
                    
            except Exception as e:
                print(f"   ⚠️ Error reading {file}: {e}")
        
        print(f"   📁 Batch files: {len(dict_files)}")
        print(f"   ✅ Terms processed: {dict_processed}")
        
        if dict_statuses:
            print(f"   📋 Status breakdown:")
            for status, count in sorted(dict_statuses.items()):
                print(f"      • {status}: {count}")
    else:
        print("   📂 No dictionary batch files found yet")
    
    print()
    print("📚 NON-DICTIONARY TERMS PROCESSING:")
    print("-" * 40)
    
    if non_dict_files:
        non_dict_processed = 0
        non_dict_statuses = {}
        
        for file in non_dict_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                non_dict_processed += len(results)
                
                for result in results:
                    status = result.get('status', 'unknown')
                    non_dict_statuses[status] = non_dict_statuses.get(status, 0) + 1
                    
            except Exception as e:
                print(f"   ⚠️ Error reading {file}: {e}")
        
        print(f"   📁 Batch files: {len(non_dict_files)}")
        print(f"   ✅ Terms processed: {non_dict_processed}")
        
        if non_dict_statuses:
            print(f"   📋 Status breakdown:")
            for status, count in sorted(non_dict_statuses.items()):
                print(f"      • {status}: {count}")
    else:
        print("   📂 No non-dictionary batch files found yet")
    
    # Check for summary files
    dict_summaries = glob.glob("dictionary_terms_summary_*.json")
    non_dict_summaries = glob.glob("non_dictionary_terms_summary_*.json")
    
    print()
    print("📄 SUMMARY FILES:")
    print("-" * 40)
    print(f"   📖 Dictionary summaries: {len(dict_summaries)}")
    print(f"   📚 Non-dictionary summaries: {len(non_dict_summaries)}")
    
    if dict_summaries or non_dict_summaries:
        print("   ✅ Summary reports available!")
    
    # File sizes
    all_files = dict_files + non_dict_files
    if all_files:
        total_size = sum(os.path.getsize(f) for f in all_files)
        print(f"\n💾 Total output size: {total_size / (1024*1024):.1f} MB")
    
    print("=" * 60)

if __name__ == "__main__":
    monitor_dual_processing()
