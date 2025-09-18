#!/usr/bin/env python3
"""
🔄 UNIVERSAL SMART RESUME
========================

Universal resume script that can automatically detect and resume from:
- Optimized Smart Runner sessions
- Fixed Dual Model Runner sessions  
- Ultra Fast Runner sessions
- Any other compatible session format

USAGE:
  python universal_smart_resume.py --best          # Resume most processed session
  python universal_smart_resume.py --latest        # Resume latest session
  python universal_smart_resume.py --from=ID       # Resume specific session
  python universal_smart_resume.py --list          # List all sessions
  python universal_smart_resume.py --optimized     # Resume best optimized smart session
"""

import os
import json
import sys
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Optional

def find_all_sessions() -> List[Dict]:
    """Find all available sessions across all runner types"""
    checkpoint_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    sessions = []
    
    # Session type patterns (priority order - ultra optimized first)
    session_patterns = [
        ('ultra_optimized_', 'ultra_optimized_smart_runner.py', 'Ultra-Optimized Smart'),
        ('optimized_smart_', 'optimized_smart_runner.py', 'Optimized Smart'),
        ('fixed_dual_', 'fixed_dual_model_runner.py', 'Fixed Dual Model'),
        ('ultra_fast_', 'ultra_fast_runner.py', 'Ultra Fast'),
        ('dual_model_', 'dual_model_ultra_fast_runner.py', 'Dual Model'),
        ('hybrid_', 'hybrid_gpu_cpu_runner.py', 'Hybrid GPU-CPU')
    ]
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('_checkpoint.json'):
            for prefix, script_name, display_name in session_patterns:
                if filename.startswith(prefix):
                    # Extract session ID
                    session_id = filename.replace(prefix, '').replace('_checkpoint.json', '')
                    checkpoint_path = os.path.join(checkpoint_dir, filename)
                    
                    try:
                        with open(checkpoint_path, 'r') as f:
                            checkpoint_data = json.load(f)
                        
                        # Get basic info
                        processed_terms = checkpoint_data.get('processed_terms', 0)
                        total_terms = checkpoint_data.get('total_terms', 0)
                        checkpoint_time = checkpoint_data.get('checkpoint_time', 0)
                        processing_rate = checkpoint_data.get('processing_rate', 0)
                        
                        # Calculate progress percentage
                        progress = (processed_terms / total_terms * 100) if total_terms > 0 else 0
                        
                        # Get smart processing stats if available
                        smart_stats = {}
                        if prefix == 'ultra_optimized_':
                            smart_stats = {
                                'ultra_minimal_terms': checkpoint_data.get('ultra_minimal_terms', 0),
                                'core_terms': checkpoint_data.get('core_terms', 0),
                                'extended_terms': checkpoint_data.get('extended_terms', 0),
                                'language_savings': checkpoint_data.get('language_savings', 0),
                                'type': 'ultra'
                            }
                        elif prefix == 'optimized_smart_':
                            smart_stats = {
                                'core_only_terms': checkpoint_data.get('core_only_terms', 0),
                                'extended_terms': checkpoint_data.get('extended_terms', 0),
                                'full_terms': checkpoint_data.get('full_terms', 0),
                                'language_savings': checkpoint_data.get('language_savings', 0),
                                'type': 'optimized'
                            }
                        
                        sessions.append({
                            'session_id': session_id,
                            'runner_type': prefix.rstrip('_'),
                            'script_name': script_name,
                            'display_name': display_name,
                            'checkpoint_path': checkpoint_path,
                            'processed_terms': processed_terms,
                            'total_terms': total_terms,
                            'progress_percent': progress,
                            'checkpoint_time': checkpoint_time,
                            'processing_rate': processing_rate,
                            'smart_stats': smart_stats
                        })
                        
                    except Exception as e:
                        print(f"⚠️  Could not read {filename}: {e}")
                        continue
                    
                    break  # Found matching pattern, move to next file
    
    return sessions

def display_sessions(sessions: List[Dict], highlight_best: bool = False):
    """Display available sessions in a formatted table"""
    if not sessions:
        print("❌ No sessions found")
        return
    
    # Sort by progress (most processed first)
    sessions_sorted = sorted(sessions, key=lambda x: x['processed_terms'], reverse=True)
    
    print(f"📋 AVAILABLE SESSIONS ({len(sessions)} found):")
    print("=" * 100)
    print(f"{'Session ID':<20} {'Type':<18} {'Progress':<12} {'Rate':<12} {'Last Updated':<20} {'Smart Stats'}")
    print("-" * 100)
    
    for i, session in enumerate(sessions_sorted):
        session_id = session['session_id']
        display_name = session['display_name']
        progress = f"{session['processed_terms']:,}/{session['total_terms']:,}"
        progress_pct = f"({session['progress_percent']:.1f}%)"
        rate = f"{session['processing_rate']:.3f}/s" if session['processing_rate'] > 0 else "N/A"
        
        # Format timestamp
        if session['checkpoint_time'] > 0:
            last_updated = datetime.fromtimestamp(session['checkpoint_time']).strftime('%m-%d %H:%M')
        else:
            last_updated = "Unknown"
        
        # Smart stats for optimized runners
        smart_info = ""
        if session['smart_stats']:
            stats = session['smart_stats']
            stats_type = stats.get('type', 'unknown')
            
            if stats_type == 'ultra':
                ultra_min = stats.get('ultra_minimal_terms', 0)
                core = stats.get('core_terms', 0)
                extended = stats.get('extended_terms', 0)
                saved = stats.get('language_savings', 0)
                smart_info = f"UM:{ultra_min} C:{core} E:{extended} S:{saved:,}"
            elif stats_type == 'optimized':
                core = stats.get('core_only_terms', 0)
                extended = stats.get('extended_terms', 0)
                saved = stats.get('language_savings', 0)
                smart_info = f"C:{core} E:{extended} S:{saved:,}"
        
        # Highlight best session
        marker = "🏆" if highlight_best and i == 0 else "  "
        
        print(f"{marker}{session_id:<20} {display_name:<18} {progress:<7} {progress_pct:<5} {rate:<12} {last_updated:<20} {smart_info}")

def get_best_session(sessions: List[Dict], prefer_optimized: bool = False) -> Optional[Dict]:
    """Get the best session to resume (most processed terms)"""
    if not sessions:
        return None
    
    if prefer_optimized:
        # Prefer ultra-optimized first, then optimized smart sessions
        ultra_sessions = [s for s in sessions if s['runner_type'] == 'ultra_optimized']
        if ultra_sessions:
            return max(ultra_sessions, key=lambda x: x['processed_terms'])
        
        optimized_sessions = [s for s in sessions if s['runner_type'] == 'optimized_smart']
        if optimized_sessions:
            return max(optimized_sessions, key=lambda x: x['processed_terms'])
    
    # Return most processed session overall
    return max(sessions, key=lambda x: x['processed_terms'])

def get_latest_session(sessions: List[Dict]) -> Optional[Dict]:
    """Get the latest session (most recent checkpoint time)"""
    if not sessions:
        return None
    
    return max(sessions, key=lambda x: x['checkpoint_time'])

def get_session_by_id(sessions: List[Dict], session_id: str) -> Optional[Dict]:
    """Get specific session by ID"""
    for session in sessions:
        if session['session_id'] == session_id:
            return session
    return None

def resume_session(session: Dict) -> bool:
    """Resume a specific session"""
    script_name = session['script_name']
    session_id = session['session_id']
    script_path = f"/home/samli/Documents/Python/Term_Verify/{script_name}"
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"🚀 Resuming {session['display_name']} session: {session_id}")
    print(f"📊 Progress: {session['processed_terms']:,}/{session['total_terms']:,} ({session['progress_percent']:.1f}%)")
    
    if session['smart_stats']:
        stats = session['smart_stats']
        print(f"🧠 Smart Stats: Core:{stats.get('core_only_terms', 0)} Extended:{stats.get('extended_terms', 0)} Saved:{stats.get('language_savings', 0):,}")
    
    print(f"⚡ Starting: python {script_name} --resume-from {session_id}")
    print()
    
    try:
        # Execute the resume command
        os.chdir("/home/samli/Documents/Python/Term_Verify")
        subprocess.run([sys.executable, script_name, "--resume-from", session_id])
        return True
    except KeyboardInterrupt:
        print(f"\n⏹️  Session interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Failed to resume session: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Universal Smart Resume - Resume any translation session")
    parser.add_argument('--best', action='store_true', help='Resume the most processed session')
    parser.add_argument('--latest', action='store_true', help='Resume the latest session')
    parser.add_argument('--optimized', action='store_true', help='Resume the best optimized smart session')
    parser.add_argument('--from', dest='session_id', help='Resume specific session by ID')
    parser.add_argument('--list', action='store_true', help='List all available sessions')
    
    args = parser.parse_args()
    
    print("🔄 UNIVERSAL SMART RESUME")
    print("=" * 40)
    
    # Find all sessions
    print("🔍 Scanning for sessions...")
    sessions = find_all_sessions()
    
    if not sessions:
        print("❌ No sessions found")
        return
    
    print(f"✅ Found {len(sessions)} sessions")
    print()
    
    # Handle different commands
    if args.list:
        display_sessions(sessions, highlight_best=True)
        return
    
    target_session = None
    
    if args.session_id:
        target_session = get_session_by_id(sessions, args.session_id)
        if not target_session:
            print(f"❌ Session '{args.session_id}' not found")
            print("\n📋 Available sessions:")
            display_sessions(sessions)
            return
    
    elif args.optimized:
        target_session = get_best_session(sessions, prefer_optimized=True)
        if not target_session:
            print("❌ No optimized smart sessions found")
            return
    
    elif args.latest:
        target_session = get_latest_session(sessions)
    
    elif args.best:
        target_session = get_best_session(sessions)
    
    else:
        # Default behavior - show options
        display_sessions(sessions, highlight_best=True)
        print()
        print("💡 Usage:")
        print("   --best      Resume most processed session")
        print("   --latest    Resume latest session")
        print("   --optimized Resume best optimized smart session")
        print("   --from=ID   Resume specific session")
        print("   --list      List all sessions")
        return
    
    if target_session:
        success = resume_session(target_session)
        if success:
            print("✅ Session completed successfully")
        else:
            print("⚠️  Session ended with issues")
    else:
        print("❌ No suitable session found")

if __name__ == "__main__":
    main()
