#!/usr/bin/env python3
"""
🔄 UNIVERSAL RESUME SCRIPT
==========================

Automatically detects and resumes from the most recent session,
whether it's from single-model (ultra_fast) or dual-model runner.

Finds the session with the most processed terms and resumes appropriately.
"""

import os
import json
import sys
from datetime import datetime
from typing import Optional, Tuple, Dict

def find_all_sessions() -> Dict[str, dict]:
    """Find all available sessions from both runners"""
    checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
    sessions = {}
    
    if not os.path.exists(checkpoints_dir):
        return sessions
    
    # Find ultra_fast sessions
    ultra_fast_files = [f for f in os.listdir(checkpoints_dir) 
                       if f.startswith("ultra_fast_") and f.endswith("_checkpoint.json")]
    
    for filename in ultra_fast_files:
        try:
            session_id = filename.replace("ultra_fast_", "").replace("_checkpoint.json", "")
            filepath = os.path.join(checkpoints_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sessions[session_id] = {
                'type': 'ultra_fast',
                'session_id': session_id,
                'processed_terms': data.get('processed_terms', 0),
                'failed_terms': data.get('failed_terms', 0),
                'total_terms': data.get('total_terms', 0),
                'processing_rate': data.get('processing_rate', 0),
                'checkpoint_time': data.get('checkpoint_time', 0),
                'file_mtime': os.path.getmtime(filepath),
                'checkpoint_file': filename
            }
        except Exception as e:
            print(f"⚠️  Could not read ultra_fast session {filename}: {e}")
    
    # Find dual_model sessions
    dual_model_files = [f for f in os.listdir(checkpoints_dir) 
                       if f.startswith("dual_model_") and f.endswith("_checkpoint.json")]
    
    for filename in dual_model_files:
        try:
            session_id = filename.replace("dual_model_", "").replace("_checkpoint.json", "")
            filepath = os.path.join(checkpoints_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sessions[session_id] = {
                'type': 'dual_model',
                'session_id': session_id,
                'processed_terms': data.get('processed_terms', 0),
                'failed_terms': data.get('failed_terms', 0),
                'total_terms': data.get('total_terms', 0),
                'processing_rate': data.get('processing_rate', 0),
                'checkpoint_time': data.get('checkpoint_time', 0),
                'file_mtime': os.path.getmtime(filepath),
                'checkpoint_file': filename
            }
        except Exception as e:
            print(f"⚠️  Could not read dual_model session {filename}: {e}")
    
    return sessions

def find_best_session(sessions: Dict[str, dict]) -> Optional[Tuple[str, dict]]:
    """Find the session with the most processed terms"""
    if not sessions:
        return None
    
    best_session_id = None
    best_session = None
    max_processed = -1
    
    for session_id, session_data in sessions.items():
        processed = session_data.get('processed_terms', 0)
        if processed > max_processed:
            max_processed = processed
            best_session_id = session_id
            best_session = session_data
    
    return best_session_id, best_session

def find_latest_session(sessions: Dict[str, dict]) -> Optional[Tuple[str, dict]]:
    """Find the most recently modified session"""
    if not sessions:
        return None
    
    latest_session_id = None
    latest_session = None
    latest_time = 0
    
    for session_id, session_data in sessions.items():
        mtime = session_data.get('file_mtime', 0)
        if mtime > latest_time:
            latest_time = mtime
            latest_session_id = session_id
            latest_session = session_data
    
    return latest_session_id, latest_session

def list_all_sessions(sessions: Dict[str, dict]):
    """List all available sessions"""
    print("📋 ALL AVAILABLE SESSIONS:")
    print("=" * 80)
    
    if not sessions:
        print("   No sessions found")
        return
    
    # Sort by processed terms (descending)
    sorted_sessions = sorted(sessions.items(), 
                           key=lambda x: x[1].get('processed_terms', 0), 
                           reverse=True)
    
    for session_id, session_data in sorted_sessions:
        session_type = session_data['type']
        processed = session_data.get('processed_terms', 0)
        failed = session_data.get('failed_terms', 0)
        total = session_data.get('total_terms', 0)
        rate = session_data.get('processing_rate', 0)
        
        progress = (processed / total * 100) if total > 0 else 0
        
        # Format session type
        type_display = "🚀 DUAL-MODEL" if session_type == "dual_model" else "⚡ SINGLE-MODEL"
        
        print(f"   {type_display} • {session_id}")
        print(f"     → Progress: {processed:,}/{total:,} ({progress:.1f}%)")
        print(f"     → Failed: {failed:,} | Rate: {rate:.3f} terms/sec")
        
        # Show age
        age_seconds = datetime.now().timestamp() - session_data.get('checkpoint_time', 0)
        if age_seconds < 3600:
            age_str = f"{age_seconds/60:.0f} minutes ago"
        elif age_seconds < 86400:
            age_str = f"{age_seconds/3600:.1f} hours ago"
        else:
            age_str = f"{age_seconds/86400:.1f} days ago"
        
        print(f"     → Last updated: {age_str}")
        print()

def resume_session(session_id: str, session_data: dict):
    """Resume the specified session"""
    session_type = session_data['type']
    processed = session_data.get('processed_terms', 0)
    
    print(f"🔄 RESUMING SESSION: {session_id}")
    print(f"   • Type: {session_type.upper().replace('_', '-')}")
    print(f"   • Progress: {processed:,} terms processed")
    print("=" * 60)
    
    if session_type == "ultra_fast":
        print("🚀 Starting ultra-fast runner...")
        os.system(f"python ultra_fast_runner.py --resume-from {session_id}")
    elif session_type == "dual_model":
        print("🎮 Starting dual-model runner...")
        os.system(f"python dual_model_ultra_fast_runner.py --resume-from {session_id}")
    else:
        print(f"❌ Unknown session type: {session_type}")
        sys.exit(1)

def main():
    print("🔄 UNIVERSAL SESSION RESUME")
    print("=" * 50)
    
    # Find all sessions
    sessions = find_all_sessions()
    
    if not sessions:
        print("❌ No sessions found to resume")
        print("\nAvailable options:")
        print("  • python ultra_fast_runner.py              (start single-model)")
        print("  • python dual_model_ultra_fast_runner.py   (start dual-model)")
        sys.exit(1)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--list":
            list_all_sessions(sessions)
            return
        
        elif command == "--best":
            result = find_best_session(sessions)
            if result:
                session_id, session_data = result
                print(f"🏆 Resuming BEST session (most progress): {session_id}")
                resume_session(session_id, session_data)
            else:
                print("❌ No sessions found")
                sys.exit(1)
        
        elif command == "--latest":
            result = find_latest_session(sessions)
            if result:
                session_id, session_data = result
                print(f"🕒 Resuming LATEST session (most recent): {session_id}")
                resume_session(session_id, session_data)
            else:
                print("❌ No sessions found")
                sys.exit(1)
        
        elif command.startswith("--from="):
            target_session_id = command.split("=", 1)[1]
            if target_session_id in sessions:
                print(f"🎯 Resuming SPECIFIC session: {target_session_id}")
                resume_session(target_session_id, sessions[target_session_id])
            else:
                print(f"❌ Session not found: {target_session_id}")
                list_all_sessions(sessions)
                sys.exit(1)
        
        else:
            print("❌ Unknown command")
            print("\nUsage:")
            print("  python universal_resume.py --best     # Resume session with most progress")
            print("  python universal_resume.py --latest   # Resume most recent session")
            print("  python universal_resume.py --list     # List all sessions")
            print("  python universal_resume.py --from=ID  # Resume specific session")
            sys.exit(1)
    
    else:
        # Default: resume best session
        result = find_best_session(sessions)
        if result:
            session_id, session_data = result
            
            print("🤖 AUTO-SELECTING BEST SESSION:")
            processed = session_data.get('processed_terms', 0)
            session_type = session_data['type'].upper().replace('_', '-')
            print(f"   • Session: {session_id} ({session_type})")
            print(f"   • Progress: {processed:,} terms processed")
            
            # Ask for confirmation
            print(f"\nPress Enter to resume, or 'list' to see all options: ", end="")
            choice = input().strip().lower()
            
            if choice == "list":
                print()
                list_all_sessions(sessions)
                print("\nTo resume a specific session:")
                print("  python universal_resume.py --from=SESSION_ID")
            else:
                print()
                resume_session(session_id, session_data)
        else:
            print("❌ No sessions found")
            sys.exit(1)

if __name__ == "__main__":
    main()
