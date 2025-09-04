#!/usr/bin/env python3
"""
Enhanced progress monitoring for the complete dataset processing.
Monitors all 61,371 terms being processed.
"""

import os
import time
import glob
import json
from datetime import datetime, timedelta

def estimate_completion_time(processed_count, elapsed_time, total_count=61371):
    """Estimate completion time based on current progress"""
    if processed_count <= 0:
        return "Unknown"
    
    rate = processed_count / elapsed_time.total_seconds()  # terms per second
    remaining = total_count - processed_count
    
    if rate > 0:
        remaining_seconds = remaining / rate
        eta = datetime.now() + timedelta(seconds=remaining_seconds)
        return eta.strftime("%Y-%m-%d %H:%M:%S")
    return "Unknown"

def get_processing_stats():
    """Get detailed statistics about the processing"""
    batch_files = glob.glob("term_validation_batch_*.json")
    batch_files.sort()
    
    total_processed = 0
    status_counts = {}
    latest_timestamp = None
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Count terms in this batch
            results = data.get('results', [])
            total_processed += len(results)
            
            # Count statuses
            for result in results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Track latest timestamp
            batch_timestamp = data.get('metadata', {}).get('timestamp', '')
            if batch_timestamp and (not latest_timestamp or batch_timestamp > latest_timestamp):
                latest_timestamp = batch_timestamp
                
        except Exception as e:
            print(f"   ⚠️ Error reading {batch_file}: {e}")
    
    return total_processed, status_counts, latest_timestamp, len(batch_files)

def monitor_full_dataset():
    """Monitor the complete dataset processing"""
    
    print("🚀 FULL DATASET TERMINOLOGY VALIDATION MONITOR")
    print("=" * 70)
    print(f"📊 Target: Process all 61,371 terms from New_Terms_Candidates_Clean.json")
    print(f"🤖 Model: GPT-4.1 with General industry context")
    print(f"📦 Batch size: 20 terms per batch")
    print("=" * 70)
    
    start_time = datetime.now()
    last_processed = 0
    
    while True:
        try:
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            # Get processing statistics
            total_processed, status_counts, latest_timestamp, batch_count = get_processing_stats()
            
            print(f"\n📈 PROGRESS UPDATE - {current_time.strftime('%H:%M:%S')}")
            print(f"   ⏱️  Runtime: {elapsed}")
            print(f"   📁 Batch files: {batch_count}")
            print(f"   ✅ Terms processed: {total_processed:,} / 61,371")
            
            if total_processed > 0:
                progress_pct = (total_processed / 61371) * 100
                print(f"   📊 Progress: {progress_pct:.2f}%")
                
                # Processing rate
                terms_per_hour = (total_processed / elapsed.total_seconds()) * 3600
                print(f"   🏃 Rate: {terms_per_hour:.1f} terms/hour")
                
                # ETA
                eta = estimate_completion_time(total_processed, elapsed)
                print(f"   🎯 ETA: {eta}")
                
                # Progress since last check
                if total_processed > last_processed:
                    new_terms = total_processed - last_processed
                    print(f"   🆕 New terms (last 60s): {new_terms}")
                last_processed = total_processed
            
            # Status breakdown
            if status_counts:
                print(f"   📋 Status breakdown:")
                for status, count in sorted(status_counts.items()):
                    pct = (count / total_processed) * 100 if total_processed > 0 else 0
                    print(f"      • {status}: {count:,} ({pct:.1f}%)")
            
            # File sizes
            batch_files = glob.glob("term_validation_batch_*.json")
            if batch_files:
                total_size = sum(os.path.getsize(f) for f in batch_files)
                print(f"   💾 Output size: {total_size / (1024*1024):.1f} MB")
            
            # Latest activity
            if latest_timestamp:
                print(f"   🕐 Latest batch: {latest_timestamp}")
            
            # Check if processing is still active
            import subprocess
            try:
                result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                                      capture_output=True, text=True)
                if 'python.exe' in result.stdout:
                    print(f"   🟢 Status: Processing active")
                else:
                    print(f"   🔴 Status: No Python processes - checking completion...")
                    
                    # Check if we've processed all terms
                    if total_processed >= 61371:
                        print(f"   🎉 PROCESSING COMPLETE! All {total_processed:,} terms processed.")
                        break
                    else:
                        print(f"   ⚠️ Processing stopped early at {total_processed:,} terms")
                        break
            except:
                print(f"   ❓ Could not check process status")
            
            print("-" * 70)
            
            # Wait before next check
            time.sleep(60)  # Check every minute for full dataset
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            print(f"   📊 Final count: {total_processed:,} terms processed")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_full_dataset()
