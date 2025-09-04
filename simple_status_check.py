#!/usr/bin/env python3
"""
Simple status checker for terminology validation processing.
This script only checks file status without interfering with running processes.
"""

import os
import glob
import json
from datetime import datetime

def check_processing_status():
    """Check the current status of processing without interfering with other processes"""
    
    print("📊 TERMINOLOGY VALIDATION STATUS CHECK")
    print("=" * 60)
    print(f"⏰ Status check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Find all batch files
    batch_files = glob.glob("term_validation_batch_*.json")
    batch_files.sort()
    
    if not batch_files:
        print("📂 No batch files found yet. Processing may be starting...")
        return
    
    print(f"📁 Batch files found: {len(batch_files)}")
    
    total_processed = 0
    status_counts = {}
    latest_timestamp = None
    latest_batch = None
    
    # Analyze batch files
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Count terms in this batch
            results = data.get('results', [])
            batch_count = len(results)
            total_processed += batch_count
            
            # Count statuses
            for result in results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Track latest timestamp
            batch_timestamp = data.get('metadata', {}).get('timestamp', '')
            if batch_timestamp and (not latest_timestamp or batch_timestamp > latest_timestamp):
                latest_timestamp = batch_timestamp
                latest_batch = batch_file
                
        except Exception as e:
            print(f"   ⚠️ Could not read {batch_file}: {e}")
    
    # Display results
    print(f"✅ Total terms processed: {total_processed:,}")
    
    if total_processed > 0:
        progress_pct = (total_processed / 61371) * 100
        print(f"📈 Progress: {progress_pct:.3f}% of 61,371 total terms")
        
        remaining = 61371 - total_processed
        print(f"⏳ Remaining: {remaining:,} terms")
    
    # Status breakdown
    if status_counts:
        print(f"\n📋 Status Summary:")
        for status, count in sorted(status_counts.items()):
            pct = (count / total_processed) * 100 if total_processed > 0 else 0
            print(f"   • {status}: {count:,} ({pct:.1f}%)")
    
    # Latest activity
    if latest_batch and latest_timestamp:
        print(f"\n🕐 Latest batch: {latest_batch}")
        print(f"⏰ Latest timestamp: {latest_timestamp}")
    
    # File sizes
    if batch_files:
        total_size = sum(os.path.getsize(f) for f in batch_files)
        print(f"💾 Total output size: {total_size / (1024*1024):.1f} MB")
    
    print("=" * 60)
    print("ℹ️ This is a non-intrusive status check. No processes were disturbed.")

if __name__ == "__main__":
    check_processing_status()
