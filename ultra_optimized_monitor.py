#!/usr/bin/env python3
"""
⚡ ULTRA-OPTIMIZED SMART RUNNER MONITOR
======================================

Real-time monitoring for the ultra-optimized smart dual-model runner
- ⚡ Ultra processing statistics with performance tracking
- 🚀 5-7x speedup metrics vs full processing
- 🧠 Predictive optimization analysis
- 💾 System resource monitoring with efficiency focus
- 📊 Ultra-aggressive language reduction analytics
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
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 4:
                    used, total, util, temp = parts[0], parts[1], parts[2], parts[3]
                    gpu_info.append({
                        'id': i,
                        'memory_used': int(used),
                        'memory_total': int(total),
                        'memory_percent': (int(used) / int(total)) * 100,
                        'utilization': int(util),
                        'temperature': int(temp)
                    })
            return gpu_info
        return []
    except:
        return []

def find_latest_ultra_optimized_session():
    """Find the latest ultra-optimized session"""
    checkpoint_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    ultra_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('ultra_optimized_') and filename.endswith('_checkpoint.json'):
            # Extract session ID
            session_id = filename.replace('ultra_optimized_', '').replace('_checkpoint.json', '')
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            results_path = os.path.join(checkpoint_dir, f'ultra_optimized_{session_id}_results.json')
            
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    ultra_files.append({
                        'session_id': session_id,
                        'checkpoint_path': checkpoint_path,
                        'results_path': results_path,
                        'checkpoint_time': checkpoint_data.get('checkpoint_time', 0),
                        'processed_terms': checkpoint_data.get('processed_terms', 0),
                        'total_terms': checkpoint_data.get('total_terms', 0)
                    })
                except:
                    continue
    
    if not ultra_files:
        return None
    
    # Return the most recently updated session
    return max(ultra_files, key=lambda x: x['checkpoint_time'])

def analyze_ultra_efficiency(results_data):
    """Analyze ultra processing efficiency with tier breakdown"""
    if not results_data:
        return {}
    
    # Processing tier analysis
    tier_counts = Counter()
    languages_by_tier = defaultdict(list)
    processing_times_by_tier = defaultdict(list)
    
    total_languages_processed = 0
    total_languages_saved = 0
    total_processing_time = 0
    
    for result in results_data:
        if result.get('status') != 'completed':
            continue
            
        tier = result.get('processing_tier', 'unknown')
        tier_counts[tier] += 1
        
        langs_processed = result.get('languages_processed', 0)
        langs_saved = result.get('languages_saved', 0)
        proc_time = result.get('processing_time_seconds', 0)
        
        languages_by_tier[tier].append(langs_processed)
        processing_times_by_tier[tier].append(proc_time)
        
        total_languages_processed += langs_processed
        total_languages_saved += langs_saved
        total_processing_time += proc_time
    
    # Calculate ultra metrics by tier
    tier_analysis = {}
    for tier, count in tier_counts.items():
        if count > 0:
            avg_langs = sum(languages_by_tier[tier]) / count
            avg_time = sum(processing_times_by_tier[tier]) / count
            total_langs = sum(languages_by_tier[tier])
            total_saved = count * (202 - avg_langs)  # Estimate saved
            
            tier_analysis[tier] = {
                'count': count,
                'avg_languages': avg_langs,
                'avg_processing_time': avg_time,
                'total_languages_processed': total_langs,
                'estimated_languages_saved': int(total_saved),
                'efficiency_percent': ((202 - avg_langs) / 202) * 100
            }
    
    # Overall ultra efficiency
    total_terms = len([r for r in results_data if r.get('status') == 'completed'])
    full_processing_estimate = total_terms * 202
    ultra_efficiency = (total_languages_saved / full_processing_estimate * 100) if full_processing_estimate > 0 else 0
    
    # Speed metrics
    avg_processing_time = total_processing_time / total_terms if total_terms > 0 else 0
    estimated_full_time = avg_processing_time * (202 / (total_languages_processed / total_terms)) if total_terms > 0 else 0
    speedup_factor = estimated_full_time / avg_processing_time if avg_processing_time > 0 else 0
    
    return {
        'tier_analysis': tier_analysis,
        'ultra_metrics': {
            'total_terms': total_terms,
            'total_languages_processed': total_languages_processed,
            'total_languages_saved': total_languages_saved,
            'ultra_efficiency_percent': ultra_efficiency,
            'avg_languages_per_term': total_languages_processed / total_terms if total_terms > 0 else 0,
            'avg_processing_time_per_term': avg_processing_time,
            'estimated_speedup_factor': speedup_factor,
            'time_per_language': avg_processing_time / (total_languages_processed / total_terms) if total_terms > 0 else 0
        }
    }

def display_ultra_monitor():
    """Display real-time ultra-optimized runner monitoring"""
    print("⚡ ULTRA-OPTIMIZED SMART RUNNER MONITOR")
    print("=" * 55)
    
    # Find latest ultra session
    session_info = find_latest_ultra_optimized_session()
    
    if not session_info:
        print("❌ No ultra-optimized sessions found")
        return
    
    session_id = session_info['session_id']
    checkpoint_path = session_info['checkpoint_path']
    results_path = session_info['results_path']
    
    print(f"⚡ Monitoring Ultra Session: {session_id}")
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
        
        # Ultra processing stats
        ultra_minimal_terms = checkpoint_data.get('ultra_minimal_terms', 0)
        core_terms = checkpoint_data.get('core_terms', 0)
        extended_terms = checkpoint_data.get('extended_terms', 0)
        language_savings = checkpoint_data.get('language_savings', 0)
        gpu_performance = checkpoint_data.get('gpu_performance', [0.0, 0.0])
        
        # Calculate ultra progress metrics
        total_processed = processed_terms + failed_terms
        progress_percent = (total_processed / total_terms * 100) if total_terms > 0 else 0
        success_rate = (processed_terms / total_processed * 100) if total_processed > 0 else 0
        
        # Ultra ETA calculation
        if processing_rate > 0:
            remaining_terms = total_terms - total_processed
            eta_seconds = remaining_terms / processing_rate
            eta_hours = eta_seconds / 3600
        else:
            eta_hours = 0
        
        # Display ultra progress
        print(f"⚡ ULTRA PROCESSING PROGRESS:")
        print(f"   • Total Progress: {total_processed:,}/{total_terms:,} ({progress_percent:.1f}%)")
        print(f"   • Success Rate: {success_rate:.1f}% ({processed_terms:,} success, {failed_terms:,} failed)")
        print(f"   • Ultra Rate: {processing_rate:.3f} terms/sec")
        print(f"   • Ultra ETA: {eta_hours:.1f} hours")
        print()
        
        # Ultra processing tier breakdown
        print(f"⚡ ULTRA PROCESSING TIER BREAKDOWN:")
        print(f"   • Ultra-Minimal: {ultra_minimal_terms:,} terms (~15 langs, ~92% efficiency)")
        print(f"   • Core: {core_terms:,} terms (~40 langs, ~80% efficiency)")
        print(f"   • Extended: {extended_terms:,} terms (~80 langs, ~60% efficiency)")
        print(f"   • Total Languages Saved: {language_savings:,}")
        print()
        
        # GPU performance tracking
        print(f"🎮 ULTRA GPU PERFORMANCE:")
        print(f"   • GPU Worker 1: {gpu_performance[0]:.2f} items/sec")
        print(f"   • GPU Worker 2: {gpu_performance[1]:.2f} items/sec")
        print(f"   • Combined Throughput: {sum(gpu_performance):.2f} items/sec")
        print(f"   • Load Balance: {abs(gpu_performance[0] - gpu_performance[1]):.2f} difference")
        print()
        
        # Ultra efficiency analysis
        if results_data:
            efficiency_analysis = analyze_ultra_efficiency(results_data)
            
            print(f"⚡ ULTRA EFFICIENCY ANALYSIS:")
            ultra_metrics = efficiency_analysis.get('ultra_metrics', {})
            print(f"   • Terms Analyzed: {ultra_metrics.get('total_terms', 0):,}")
            print(f"   • Languages Processed: {ultra_metrics.get('total_languages_processed', 0):,}")
            print(f"   • Languages Saved: {ultra_metrics.get('total_languages_saved', 0):,}")
            print(f"   • Ultra Efficiency: {ultra_metrics.get('ultra_efficiency_percent', 0):.1f}%")
            print(f"   • Avg Languages/Term: {ultra_metrics.get('avg_languages_per_term', 0):.1f}")
            print(f"   • Estimated Speedup: {ultra_metrics.get('estimated_speedup_factor', 0):.1f}x")
            print(f"   • Time per Language: {ultra_metrics.get('time_per_language', 0):.3f}s")
            print()
            
            # Ultra tier performance breakdown
            tier_analysis = efficiency_analysis.get('tier_analysis', {})
            if tier_analysis:
                print(f"⚡ ULTRA TIER PERFORMANCE:")
                for tier, stats in tier_analysis.items():
                    count = stats['count']
                    avg_langs = stats['avg_languages']
                    efficiency = stats['efficiency_percent']
                    avg_time = stats['avg_processing_time']
                    
                    print(f"   • {tier.upper()}: {count} terms")
                    print(f"     - Avg Languages: {avg_langs:.1f} | Efficiency: {efficiency:.1f}%")
                    print(f"     - Avg Time: {avg_time:.2f}s | Total Saved: {stats['estimated_languages_saved']:,}")
                print()
        
        # Ultra system resources
        print(f"💻 ULTRA SYSTEM RESOURCES:")
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        print(f"   • CPU Usage: {cpu_percent:.1f}%")
        print(f"   • RAM Usage: {memory_info.percent:.1f}% ({memory_info.used/1024**3:.1f}GB/{memory_info.total/1024**3:.1f}GB)")
        
        # Ultra GPU info
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"   • Ultra GPU Status:")
            for gpu in gpu_info:
                print(f"     - GPU {gpu['id']}: {gpu['memory_percent']:.1f}% VRAM ({gpu['memory_used']}MB/{gpu['memory_total']}MB)")
                print(f"       Utilization: {gpu['utilization']}% | Temperature: {gpu['temperature']}°C")
        else:
            print(f"   • GPU: Not available or not detected")
        print()
        
        # Ultra configuration
        config = checkpoint_data.get('config', {})
        language_sets = checkpoint_data.get('language_set_sizes', {})
        
        print(f"⚙️  ULTRA CONFIGURATION:")
        print(f"   • Model: {config.get('model_size', 'Unknown')}")
        print(f"   • GPU Workers: {config.get('gpu_workers', 0)} (Ultra-optimized)")
        print(f"   • CPU Workers: {config.get('cpu_workers', 0)} (Ultra-parallel)")
        print(f"   • Batch Size: {config.get('gpu_batch_size', 0)} (Ultra-efficient)")
        print(f"   • Ultra-Minimal Threshold: {config.get('ultra_minimal_threshold', 0)}")
        print(f"   • Ultra-Core Threshold: {config.get('ultra_core_threshold', 0)}")
        print()
        
        print(f"🌍 ULTRA LANGUAGE SETS:")
        print(f"   • Ultra-Minimal: {language_sets.get('ultra_minimal', 0)} languages (15 typical)")
        print(f"   • Ultra-Core: {language_sets.get('ultra_core', 0)} languages (40 typical)")
        print(f"   • Ultra-Extended: {language_sets.get('ultra_extended', 0)} languages (80 typical)")
        print(f"   • Full Set: {language_sets.get('full', 0)} languages")
        print()
        
        # Ultra performance comparison
        if total_terms > 0 and processing_rate > 0:
            # Estimate performance vs different processing modes
            current_eta = total_terms / processing_rate / 3600 if processing_rate > 0 else 0
            
            # Estimates based on language counts
            avg_langs_current = ultra_metrics.get('avg_languages_per_term', 25) if results_data else 25
            
            # Comparison estimates
            optimized_eta = current_eta * (60 / avg_langs_current)  # Optimized smart uses ~60 langs
            fixed_dual_eta = current_eta * (202 / avg_langs_current)  # Fixed dual uses all 202
            
            print(f"🏆 ULTRA PERFORMANCE COMPARISON:")
            print(f"   • Ultra-Optimized ETA: {current_eta:.1f} hours")
            print(f"   • Optimized Smart ETA: {optimized_eta:.1f} hours")
            print(f"   • Fixed Dual ETA: {fixed_dual_eta:.1f} hours")
            print(f"   • Ultra vs Optimized: {optimized_eta/current_eta:.1f}x faster")
            print(f"   • Ultra vs Fixed Dual: {fixed_dual_eta/current_eta:.1f}x faster")
            print(f"   • Time Saved vs Full: {fixed_dual_eta - current_eta:.1f} hours")
        
        print(f"\n🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error reading ultra session data: {e}")

def main():
    """Main ultra monitoring loop"""
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            display_ultra_monitor()
            
            print(f"\n⏱️  Ultra refresh in 25 seconds... (Ctrl+C to exit)")
            time.sleep(25)
            
    except KeyboardInterrupt:
        print(f"\n👋 Ultra monitoring stopped by user")

if __name__ == "__main__":
    main()
