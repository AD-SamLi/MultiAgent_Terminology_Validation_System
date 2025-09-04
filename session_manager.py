#!/usr/bin/env python3
"""
Session Management Utility
Manage translation processing sessions, checkpoints, and resuming
"""

import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from enhanced_translation_processor import ProcessingCheckpoint


class SessionManager:
    """Utility for managing translation processing sessions"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def list_sessions(self) -> List[Dict]:
        """List all available sessions"""
        sessions = []
        checkpoint_files = list(Path(self.checkpoint_dir).glob("*_checkpoint.pkl"))
        
        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                progress_percent = (checkpoint.processed_terms / checkpoint.total_terms) * 100
                
                sessions.append({
                    "session_id": checkpoint.session_id,
                    "file_type": checkpoint.file_type,
                    "total_terms": checkpoint.total_terms,
                    "processed_terms": checkpoint.processed_terms,
                    "failed_terms": checkpoint.failed_terms,
                    "progress_percent": progress_percent,
                    "start_time": checkpoint.start_time.isoformat(),
                    "last_update": checkpoint.last_update_time.isoformat(),
                    "checkpoint_file": str(checkpoint_file),
                    "processing_config": checkpoint.processing_config
                })
            except Exception as e:
                print(f"❌ Failed to read checkpoint {checkpoint_file}: {e}")
        
        # Sort by last update time (most recent first)
        sessions.sort(key=lambda x: x['last_update'], reverse=True)
        return sessions
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get detailed information about a specific session"""
        sessions = self.list_sessions()
        for session in sessions:
            if session['session_id'] == session_id:
                return session
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its checkpoints"""
        try:
            checkpoint_files = list(Path(self.checkpoint_dir).glob(f"{session_id}_*_checkpoint.pkl"))
            
            if not checkpoint_files:
                print(f"❌ No checkpoints found for session: {session_id}")
                return False
            
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
                print(f"🗑️  Deleted: {checkpoint_file}")
            
            print(f"✅ Session {session_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete session {session_id}: {e}")
            return False
    
    def export_session_info(self, output_file: str = "session_info.json"):
        """Export session information to JSON file"""
        sessions = self.list_sessions()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Session info exported to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export session info: {e}")
            return False
    
    def cleanup_old_sessions(self, keep_recent: int = 5) -> int:
        """Clean up old sessions, keeping only the most recent ones"""
        sessions = self.list_sessions()
        
        if len(sessions) <= keep_recent:
            print(f"✅ No cleanup needed. Found {len(sessions)} sessions, keeping {keep_recent}")
            return 0
        
        sessions_to_delete = sessions[keep_recent:]
        deleted_count = 0
        
        for session in sessions_to_delete:
            if self.delete_session(session['session_id']):
                deleted_count += 1
        
        print(f"🧹 Cleaned up {deleted_count} old sessions")
        return deleted_count
    
    def print_session_summary(self):
        """Print a formatted summary of all sessions"""
        sessions = self.list_sessions()
        
        if not sessions:
            print("📋 No sessions found")
            return
        
        print("📋 SESSION SUMMARY")
        print("=" * 80)
        
        for i, session in enumerate(sessions, 1):
            print(f"{i}. Session: {session['session_id']}")
            print(f"   📁 Type: {session['file_type']} terms")
            print(f"   📊 Progress: {session['processed_terms']:,}/{session['total_terms']:,} "
                  f"({session['progress_percent']:.1f}%)")
            
            if session['failed_terms'] > 0:
                print(f"   ❌ Failed: {session['failed_terms']:,} terms")
            
            print(f"   🕐 Started: {session['start_time']}")
            print(f"   🕐 Updated: {session['last_update']}")
            
            config = session.get('processing_config', {})
            if config:
                parallel_config = config.get('parallel_config', {})
                print(f"   ⚙️  Config: {config.get('model_size', 'unknown')} model, "
                      f"{parallel_config.get('max_workers', 'unknown')} workers")
            
            print()


def main():
    """Main CLI for session management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translation Session Manager")
    parser.add_argument("--list", action="store_true", help="List all sessions")
    parser.add_argument("--info", type=str, help="Get info for specific session ID")
    parser.add_argument("--delete", type=str, help="Delete specific session ID")
    parser.add_argument("--export", type=str, help="Export session info to JSON file")
    parser.add_argument("--cleanup", type=int, default=5, help="Clean up old sessions, keep N recent")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.list:
        manager.print_session_summary()
    
    elif args.info:
        session_info = manager.get_session_info(args.info)
        if session_info:
            print(f"📋 SESSION INFO: {args.info}")
            print("=" * 50)
            for key, value in session_info.items():
                print(f"{key}: {value}")
        else:
            print(f"❌ Session not found: {args.info}")
    
    elif args.delete:
        manager.delete_session(args.delete)
    
    elif args.export:
        manager.export_session_info(args.export)
    
    elif args.cleanup:
        manager.cleanup_old_sessions(args.cleanup)
    
    else:
        # Default: show summary
        manager.print_session_summary()


if __name__ == "__main__":
    main()

