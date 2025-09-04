#!/usr/bin/env python3
"""
Progress monitoring script for the full-scale terminology validation processing.
"""

import os
import time
import glob
import json
from datetime import datetime

def monitor_progress():
    """Monitor the progress of batch processing"""
    
    print("🔍 TERMINOLOGY VALIDATION PROGRESS MONITOR")
    print("=" * 60)
    
    start_time = datetime.now()
    
    while True:
        try:
            # Find all batch files
            batch_files = glob.glob("term_validation_batch_*.json")
            batch_files.sort()
            
            if batch_files:
                print(f"\n📊 Progress Update - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   📁 Batch files created: {len(batch_files)}")
                
                # Check the latest batch file
                latest_batch = batch_files[-1]
                try:
                    with open(latest_batch, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        total_results = data.get('metadata', {}).get('total_results', 0)
                        timestamp = data.get('metadata', {}).get('timestamp', '')
                        
                    print(f"   📄 Latest batch: {latest_batch}")
                    print(f"   ✅ Terms in latest batch: {total_results}")
                    print(f"   ⏰ Latest timestamp: {timestamp}")
                    
                    # Estimate total progress
                    estimated_terms_processed = len(batch_files) * 10  # Assuming 10 terms per batch
                    print(f"   🎯 Estimated total processed: ~{estimated_terms_processed} terms")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not read latest batch: {e}")
                
                # Check file sizes
                total_size = sum(os.path.getsize(f) for f in batch_files)
                print(f"   💾 Total output size: {total_size / (1024*1024):.1f} MB")
                
            else:
                print(f"\n⏳ Waiting for batch files to be created...")
            
            # Check if Python process is still running
            import subprocess
            try:
                result = subprocess.run(['tasklist', '/fi', 'imagename eq python.exe'], 
                                      capture_output=True, text=True)
                if 'python.exe' in result.stdout:
                    print(f"   🟢 Processing is active")
                else:
                    print(f"   🔴 No Python processes found - processing may be complete")
                    break
            except:
                print(f"   ❓ Could not check process status")
            
            # Runtime
            elapsed = datetime.now() - start_time
            print(f"   ⏱️ Runtime: {elapsed}")
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_progress()

