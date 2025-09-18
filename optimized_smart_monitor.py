#!/usr/bin/env python3
"""
📊 OPTIMIZED SMART RUNNER MONITOR
================================

Real-time monitoring for the optimized smart dual-model runner
- 🚀 Smart processing statistics
- 📈 Language efficiency metrics
- 🎯 Performance vs. full processing comparison
- 💾 System resource monitoring
- 🔍 Processing tier analysis
"""

import os
import json
import time
import psutil
import subprocess
from datetime import datetime
from collections import defaultdict, Counter

def get_gpu_info():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    used, total, util = parts[0], parts[1], parts[2]
                    gpu_info.append({
                        'id': i,
                        'memory_used': int(used),
                        'memory_total': int(total),
                        'memory_percent': (int(used) / int(total)) * 100,
                        'utilization': int(util)
                    })
            return gpu_info
        return []
    except:
        return []

def find_latest_optimized_session():
    """Find the latest optimized smart session"""
    checkpoint_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    optimized_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('optimized_smart_') and filename.endswith('_checkpoint.json'):
            # Extract session ID
            session_id = filename.replace('optimized_smart_', '').replace('_checkpoint.json', '')
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            results_path = os.path.join(checkpoint_dir, f'optimized_smart_{session_id}_results.json')
            
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    optimized_files.append({
                        'session_id': session_id,
                        'checkpoint_path': checkpoint_path,
                        'results_path': results_path,
                        'checkpoint_time': checkpoint_data.get('checkpoint_time', 0),
                        'processed_terms': checkpoint_data.get('processed_terms', 0),
                        'total_terms': checkpoint_data.get('total_terms', 0)
                    })
                except:
                    continue
    
    if not optimized_files:
        return None
    
    # Return the most recently updated session
    return max(optimized_files, key=lambda x: x['checkpoint_time'])

def analyze_smart_efficiency(results_data):
    """Analyze smart processing efficiency"""
    if not results_data:
        return {}
    
    # Processing tier analysis
    tier_counts = Counter()
    languages_processed_by_tier = defaultdict(list)
    languages_saved_by_tier = defaultdict(list)
    
    total_languages_processed = 0
    total_languages_saved = 0
    
    for result in results_data:
        if result.get('status') != 'completed':
            continue
            
        tier = result.get('processing_tier', 'unknown')
        tier_counts[tier] += 1
        
        langs_processed = result.get('languages_processed', 0)
        langs_saved = result.get('languages_saved', 0)
        
        languages_processed_by_tier[tier].append(langs_processed)
        languages_saved_by_tier[tier].append(langs_saved)
        
        total_languages_processed += langs_processed
        total_languages_saved += langs_saved
    
    # Calculate averages by tier
    tier_analysis = {}
    for tier, count in tier_counts.items():
        if count > 0:
            avg_processed = sum(languages_processed_by_tier[tier]) / count
            avg_saved = sum(languages_saved_by_tier[tier]) / count
            tier_analysis[tier] = {
                'count': count,
                'avg_languages_processed': avg_processed,
                'avg_languages_saved': avg_saved,
                'total_languages_processed': sum(languages_processed_by_tier[tier]),
                'total_languages_saved': sum(languages_saved_by_tier[tier])
            }
    
    # Overall efficiency
    total_terms = len([r for r in results_data if r.get('status') == 'completed'])
    full_processing_estimate = total_terms * 202  # 202 languages per term
    efficiency_gain = (total_languages_saved / full_processing_estimate * 100) if full_processing_estimate > 0 else 0
    
    return {
        'tier_analysis': tier_analysis,
        'overall_stats': {
            'total_terms': total_terms,
            'total_languages_processed': total_languages_processed,
            'total_languages_saved': total_languages_saved,
            'efficiency_gain_percent': efficiency_gain,
            'avg_languages_per_term': total_languages_processed / total_terms if total_terms > 0 else 0
        }
    }

def display_optimized_monitor():
    """Display real-time optimized smart runner monitoring"""
    print("🚀 OPTIMIZED SMART RUNNER MONITOR")
    print("=" * 50)
    
    # Find latest session
    session_info = find_latest_optimized_session()
    
    if not session_info:
        print("❌ No optimized smart sessions found")
        return
    
    session_id = session_info['session_id']
    checkpoint_path = session_info['checkpoint_path']
    results_path = session_info['results_path']
    
    print(f"📊 Monitoring Session: {session_id}")
    print(f"🕒 Started: {datetime.fromtimestamp(session_info['checkpoint_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load checkpoint data
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load results if available
        results_data = []
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_data = json.load(f)
        
        # Basic progress info
        processed_terms = checkpoint_data.get('processed_terms', 0)
        failed_terms = checkpoint_data.get('failed_terms', 0)
        total_terms = checkpoint_data.get('total_terms', 0)
        processing_rate = checkpoint_data.get('processing_rate', 0)
        
        # Smart processing stats
        core_only_terms = checkpoint_data.get('core_only_terms', 0)
        extended_terms = checkpoint_data.get('extended_terms', 0)
        full_terms = checkpoint_data.get('full_terms', 0)
        language_savings = checkpoint_data.get('language_savings', 0)
        
        # Calculate progress metrics
        total_processed = processed_terms + failed_terms
        progress_percent = (total_processed / total_terms * 100) if total_terms > 0 else 0
        success_rate = (processed_terms / total_processed * 100) if total_processed > 0 else 0
        
        # Estimate completion
        if processing_rate > 0:
            remaining_terms = total_terms - total_processed
            eta_seconds = remaining_terms / processing_rate
            eta_hours = eta_seconds / 3600
        else:
            eta_hours = 0
        
        # Display progress
        print(f"📈 PROCESSING PROGRESS:")
        print(f"   • Total Progress: {total_processed:,}/{total_terms:,} ({progress_percent:.1f}%)")
        print(f"   • Success Rate: {success_rate:.1f}% ({processed_terms:,} success, {failed_terms:,} failed)")
        print(f"   • Processing Rate: {processing_rate:.3f} terms/sec")
        print(f"   • ETA: {eta_hours:.1f} hours")
        print()
        
        # Smart processing breakdown
        print(f"🚀 SMART PROCESSING BREAKDOWN:")
        print(f"   • Core Only: {core_only_terms:,} terms")
        print(f"   • Extended: {extended_terms:,} terms")
        print(f"   • Full Set: {full_terms:,} terms")
        print(f"   • Languages Saved: {language_savings:,}")
        print()
        
        # Efficiency analysis
        if results_data:
            efficiency_analysis = analyze_smart_efficiency(results_data)
            
            print(f"⚡ EFFICIENCY ANALYSIS:")
            overall_stats = efficiency_analysis.get('overall_stats', {})
            print(f"   • Total Terms Analyzed: {overall_stats.get('total_terms', 0):,}")
            print(f"   • Languages Processed: {overall_stats.get('total_languages_processed', 0):,}")
            print(f"   • Languages Saved: {overall_stats.get('total_languages_saved', 0):,}")
            print(f"   • Efficiency Gain: {overall_stats.get('efficiency_gain_percent', 0):.1f}%")
            print(f"   • Avg Languages/Term: {overall_stats.get('avg_languages_per_term', 0):.1f}")
            print()
            
            # Tier analysis
            tier_analysis = efficiency_analysis.get('tier_analysis', {})
            if tier_analysis:
                print(f"🎯 PROCESSING TIER ANALYSIS:")
                for tier, stats in tier_analysis.items():
                    print(f"   • {tier.upper()}: {stats['count']} terms")
                    print(f"     - Avg Languages: {stats['avg_languages_processed']:.1f}")
                    print(f"     - Avg Saved: {stats['avg_languages_saved']:.1f}")
                print()
        
        # System resources
        print(f"💻 SYSTEM RESOURCES:")
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        print(f"   • CPU Usage: {cpu_percent:.1f}%")
        print(f"   • RAM Usage: {memory_info.percent:.1f}% ({memory_info.used/1024**3:.1f}GB/{memory_info.total/1024**3:.1f}GB)")
        
        # GPU info
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"   • GPU Info:")
            for gpu in gpu_info:
                print(f"     - GPU {gpu['id']}: {gpu['memory_percent']:.1f}% VRAM ({gpu['memory_used']}MB/{gpu['memory_total']}MB) | {gpu['utilization']}% Util")
        else:
            print(f"   • GPU: Not available or not detected")
        print()
        
        # Configuration
        config = checkpoint_data.get('config', {})
        language_sets = checkpoint_data.get('language_set_sizes', {})
        
        print(f"⚙️  CONFIGURATION:")
        print(f"   • Model: {config.get('model_size', 'Unknown')}")
        print(f"   • GPU Workers: {config.get('gpu_workers', 0)}")
        print(f"   • CPU Workers: {config.get('cpu_workers', 0)}")
        print(f"   • Batch Size: {config.get('gpu_batch_size', 0)}")
        print(f"   • Core Threshold: {config.get('core_language_threshold', 0)}")
        print(f"   • Extended Threshold: {config.get('extended_language_threshold', 0)}")
        print()
        
        print(f"🌍 LANGUAGE SETS:")
        print(f"   • Core Languages: {language_sets.get('core', 0)}")
        print(f"   • Extended Languages: {language_sets.get('extended', 0)}")
        print(f"   • Full Languages: {language_sets.get('full', 0)}")
        print()
        
        # Performance comparison
        if total_terms > 0 and processing_rate > 0:
            # Estimate full processing time
            full_processing_rate_estimate = processing_rate * (overall_stats.get('avg_languages_per_term', 60) / 202)
            full_processing_eta = total_terms / full_processing_rate_estimate / 3600 if full_processing_rate_estimate > 0 else 0
            
            current_eta = total_terms / processing_rate / 3600 if processing_rate > 0 else 0
            speedup_factor = full_processing_eta / current_eta if current_eta > 0 else 0
            
            print(f"🏆 PERFORMANCE COMPARISON:")
            print(f"   • Smart Processing ETA: {current_eta:.1f} hours")
            print(f"   • Full Processing ETA: {full_processing_eta:.1f} hours")
            print(f"   • Speedup Factor: {speedup_factor:.1f}x faster")
            print(f"   • Time Saved: {full_processing_eta - current_eta:.1f} hours")
        
        print(f"\n🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error reading session data: {e}")

def main():
    """Main monitoring loop"""
    try:
        while True:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            display_optimized_monitor()
            
            print(f"\n⏱️  Refreshing in 30 seconds... (Ctrl+C to exit)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    main()
