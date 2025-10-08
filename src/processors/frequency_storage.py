#!/usr/bin/env python3
"""
Frequency Storage System for Terms with Frequency = 1
Stores terms that appear only once for future reference and reconsideration
"""

import os
import json
import time
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class FrequencyStorageSystem:
    """
    Storage system for managing terms with frequency = 1
    Implements the requirement: "Store the 1 frequency terms for next time if it appear again"
    """
    
    def __init__(self, storage_dir: str = "frequency_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # SQLite database for efficient storage and retrieval
        self.db_path = os.path.join(storage_dir, "frequency_storage.db")
        self.json_backup_path = os.path.join(storage_dir, "frequency_storage_backup.json")
        
        self._init_database()
        print(f"[OK] Frequency Storage System initialized at: {storage_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for frequency storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frequency_terms (
                    term_id TEXT PRIMARY KEY,
                    term TEXT NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    occurrence_count INTEGER DEFAULT 1,
                    source_files TEXT,  -- JSON array of source files
                    original_contexts TEXT,  -- JSON array of original contexts
                    pos_tags TEXT,  -- JSON array of POS tags
                    promoted_to_processing BOOLEAN DEFAULT FALSE,
                    promoted_at TIMESTAMP,
                    status TEXT DEFAULT 'stored',  -- stored, promoted, archived
                    metadata TEXT  -- JSON for additional metadata
                )
            """)
            
            # Index for efficient lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_term ON frequency_terms(term)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON frequency_terms(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_occurrence_count ON frequency_terms(occurrence_count)")
            
            print("[STATS] Frequency storage database initialized")
    
    def _get_term_id(self, term: str) -> str:
        """Generate unique ID for a term"""
        return hashlib.md5(term.lower().encode()).hexdigest()[:16]
    
    def store_frequency_one_term(self, term: str, source_file: str = None, 
                                original_contexts: List[str] = None, 
                                pos_tags: List[str] = None,
                                metadata: Dict = None) -> bool:
        """
        Store a term with frequency = 1 for future reference
        
        Args:
            term: The term to store
            source_file: Source file where term was found
            original_contexts: List of original text contexts
            pos_tags: List of POS tags for the term
            metadata: Additional metadata
            
        Returns:
            True if stored/updated, False if already promoted
        """
        term_id = self._get_term_id(term)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if term already exists
            cursor = conn.execute(
                "SELECT term, occurrence_count, promoted_to_processing, source_files, original_contexts, pos_tags FROM frequency_terms WHERE term_id = ?",
                (term_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                existing_term, count, promoted, existing_sources, existing_contexts, existing_pos_tags = existing
                
                if promoted:
                    print(f"ℹ️  Term '{term}' already promoted to processing")
                    return False
                
                # Update existing term
                new_count = count + 1
                
                # Merge source files
                try:
                    sources_list = json.loads(existing_sources) if existing_sources else []
                except:
                    sources_list = []
                
                if source_file and source_file not in sources_list:
                    sources_list.append(source_file)
                
                # Merge contexts
                try:
                    contexts_list = json.loads(existing_contexts) if existing_contexts else []
                except:
                    contexts_list = []
                
                if original_contexts:
                    for context in original_contexts:
                        if context not in contexts_list:
                            contexts_list.append(context)
                
                # Merge POS tags
                try:
                    pos_list = json.loads(existing_pos_tags) if existing_pos_tags else []
                except:
                    pos_list = []
                
                if pos_tags:
                    for tag in pos_tags:
                        if tag not in pos_list:
                            pos_list.append(tag)
                
                # Update the record
                conn.execute("""
                    UPDATE frequency_terms 
                    SET occurrence_count = ?, last_seen = CURRENT_TIMESTAMP,
                        source_files = ?, original_contexts = ?, pos_tags = ?, metadata = ?
                    WHERE term_id = ?
                """, (
                    new_count,
                    json.dumps(sources_list),
                    json.dumps(contexts_list),
                    json.dumps(pos_list),
                    json.dumps(metadata) if metadata else None,
                    term_id
                ))
                
                print(f"[PROGRESS] Updated term '{term}': occurrence count {count} -> {new_count}")
                
                # Check if term should be promoted (frequency > 1)
                if new_count >= 2:
                    self._promote_term_to_processing(term_id, conn)
                    return True
                
            else:
                # Insert new term
                conn.execute("""
                    INSERT INTO frequency_terms 
                    (term_id, term, source_files, original_contexts, pos_tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    term_id,
                    term,
                    json.dumps([source_file] if source_file else []),
                    json.dumps(original_contexts) if original_contexts else None,
                    json.dumps(pos_tags) if pos_tags else None,
                    json.dumps(metadata) if metadata else None
                ))
                
                print(f"[SAVE] Stored new frequency=1 term: '{term}'")
        
        return True
    
    def _promote_term_to_processing(self, term_id: str, conn: sqlite3.Connection):
        """Promote a term to processing when frequency >= 2"""
        conn.execute("""
            UPDATE frequency_terms 
            SET promoted_to_processing = TRUE, promoted_at = CURRENT_TIMESTAMP, status = 'promoted'
            WHERE term_id = ?
        """, (term_id,))
        
        # Get term details
        cursor = conn.execute("SELECT term, occurrence_count FROM frequency_terms WHERE term_id = ?", (term_id,))
        term, count = cursor.fetchone()
        
        print(f"[START] PROMOTED: Term '{term}' now has frequency {count} - ready for processing!")
    
    def get_terms_ready_for_processing(self) -> List[Dict]:
        """Get terms that have reached frequency >= 2 and are ready for processing"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT term_id, term, occurrence_count, first_seen, last_seen, 
                       source_files, original_contexts, pos_tags, metadata
                FROM frequency_terms 
                WHERE promoted_to_processing = TRUE AND status = 'promoted'
                ORDER BY occurrence_count DESC, last_seen DESC
            """)
            
            ready_terms = []
            for row in cursor.fetchall():
                term_id, term, count, first_seen, last_seen, sources, contexts, pos_tags, metadata = row
                
                ready_terms.append({
                    'term_id': term_id,
                    'term': term,
                    'frequency': count,
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'source_files': json.loads(sources) if sources else [],
                    'original_contexts': json.loads(contexts) if contexts else [],
                    'pos_tags': json.loads(pos_tags) if pos_tags else [],
                    'metadata': json.loads(metadata) if metadata else {}
                })
            
            return ready_terms
    
    def mark_term_as_processed(self, term: str) -> bool:
        """Mark a term as processed to avoid reprocessing"""
        term_id = self._get_term_id(term)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE frequency_terms SET status = 'processed' WHERE term_id = ?",
                (term_id,)
            )
            
            if cursor.rowcount > 0:
                print(f"[OK] Marked term '{term}' as processed")
                return True
            
            return False
    
    def get_storage_statistics(self) -> Dict:
        """Get statistics about stored terms"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total terms
            cursor = conn.execute("SELECT COUNT(*) FROM frequency_terms")
            stats['total_terms'] = cursor.fetchone()[0]
            
            # By status
            cursor = conn.execute("SELECT status, COUNT(*) FROM frequency_terms GROUP BY status")
            stats['by_status'] = dict(cursor.fetchall())
            
            # By occurrence count
            cursor = conn.execute("SELECT occurrence_count, COUNT(*) FROM frequency_terms GROUP BY occurrence_count ORDER BY occurrence_count")
            stats['by_frequency'] = dict(cursor.fetchall())
            
            # Ready for processing
            cursor = conn.execute("SELECT COUNT(*) FROM frequency_terms WHERE promoted_to_processing = TRUE AND status = 'promoted'")
            stats['ready_for_processing'] = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cursor = conn.execute("SELECT COUNT(*) FROM frequency_terms WHERE last_seen > datetime('now', '-1 day')")
            stats['recent_activity'] = cursor.fetchone()[0]
            
            return stats
    
    def search_stored_terms(self, pattern: str = None, min_frequency: int = None, 
                           status: str = None) -> List[Dict]:
        """Search stored terms with filters"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM frequency_terms WHERE 1=1"
            params = []
            
            if pattern:
                query += " AND term LIKE ?"
                params.append(f"%{pattern}%")
            
            if min_frequency:
                query += " AND occurrence_count >= ?"
                params.append(min_frequency)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY occurrence_count DESC, last_seen DESC"
            
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    'term_id': row[0],
                    'term': row[1],
                    'first_seen': row[2],
                    'last_seen': row[3],
                    'occurrence_count': row[4],
                    'source_files': json.loads(row[5]) if row[5] else [],
                    'original_contexts': json.loads(row[6]) if row[6] else [],
                    'pos_tags': json.loads(row[7]) if row[7] else [],
                    'promoted_to_processing': bool(row[8]),
                    'promoted_at': row[9],
                    'status': row[10],
                    'metadata': json.loads(row[11]) if row[11] else {}
                })
            
            return results
    
    def export_to_json(self, output_file: str = None) -> str:
        """Export all stored terms to JSON format"""
        if not output_file:
            output_file = self.json_backup_path
        
        all_terms = self.search_stored_terms()
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_storage_statistics(),
            'terms': all_terms
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Exported {len(all_terms)} terms to: {output_file}")
        return output_file
    
    def import_from_existing_data(self, input_file: str, source_name: str = None):
        """Import terms from existing data files (CSV, JSON)"""
        print(f"[INPUT] Importing frequency=1 terms from: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            return
        
        imported_count = 0
        
        try:
            if input_file.endswith('.json'):
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    if 'terms' in data:
                        terms_data = data['terms']
                    elif 'dictionary_terms' in data:
                        terms_data = data['dictionary_terms']
                    elif 'non_dictionary_terms' in data:
                        terms_data = data['non_dictionary_terms']
                    else:
                        terms_data = data
                else:
                    terms_data = data
                
                # Process each term
                for term_entry in terms_data:
                    if isinstance(term_entry, dict):
                        term = term_entry.get('term', '')
                        frequency = term_entry.get('frequency', 1)
                        
                        # Only import frequency=1 terms
                        if frequency == 1 and term:
                            original_texts = term_entry.get('original_texts', {})
                            contexts = original_texts.get('texts', []) if isinstance(original_texts, dict) else []
                            
                            pos_tags = term_entry.get('pos_tag_variations', {})
                            tags = pos_tags.get('tags', []) if isinstance(pos_tags, dict) else []
                            
                            self.store_frequency_one_term(
                                term=term,
                                source_file=source_name or input_file,
                                original_contexts=contexts,
                                pos_tags=tags,
                                metadata={'imported_from': input_file}
                            )
                            imported_count += 1
            
            elif input_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(input_file)
                
                for _, row in df.iterrows():
                    if 'Terms' in row and 'Frequency' in row:
                        if row['Frequency'] == 1:
                            contexts = [row.get('Original Text', '')] if 'Original Text' in row else []
                            pos_tags = [row.get('pos_tags', '')] if 'pos_tags' in row else []
                            
                            self.store_frequency_one_term(
                                term=row['Terms'],
                                source_file=source_name or input_file,
                                original_contexts=contexts,
                                pos_tags=pos_tags,
                                metadata={'imported_from': input_file}
                            )
                            imported_count += 1
            
            print(f"[OK] Imported {imported_count} frequency=1 terms from {input_file}")
            
        except Exception as e:
            print(f"[ERROR] Error importing from {input_file}: {e}")
    
    def cleanup_old_entries(self, days_old: int = 90):
        """Clean up old entries that haven't been seen recently"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM frequency_terms 
                WHERE status = 'stored' 
                AND occurrence_count = 1 
                AND last_seen < datetime('now', '-{} days')
            """.format(days_old))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"[CLEAN] Cleaned up {deleted_count} old frequency=1 entries (older than {days_old} days)")


def main():
    """Demo and test the frequency storage system"""
    print("🗄️  FREQUENCY STORAGE SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize storage system
    storage = FrequencyStorageSystem()
    
    # Demo: Store some frequency=1 terms
    demo_terms = [
        {
            'term': 'vertex_optimization',
            'contexts': ['The vertex_optimization algorithm improves mesh quality'],
            'pos_tags': ['NN'],
            'source': 'demo_file_1.txt'
        },
        {
            'term': 'autocad_plugin',
            'contexts': ['Install the autocad_plugin for enhanced functionality'],
            'pos_tags': ['NN'],
            'source': 'demo_file_2.txt'
        }
    ]
    
    print("\n[INPUT] Storing frequency=1 terms...")
    for term_data in demo_terms:
        storage.store_frequency_one_term(
            term=term_data['term'],
            source_file=term_data['source'],
            original_contexts=term_data['contexts'],
            pos_tags=term_data['pos_tags']
        )
    
    # Demo: Add the same term again (should increase frequency)
    print("\n[PROGRESS] Adding same term again (frequency increase)...")
    storage.store_frequency_one_term(
        term='vertex_optimization',
        source_file='demo_file_3.txt',
        original_contexts=['Another usage of vertex_optimization in the code'],
        pos_tags=['NN']
    )
    
    # Show statistics
    print("\n[STATS] Storage Statistics:")
    stats = storage.get_storage_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show terms ready for processing
    print("\n[START] Terms Ready for Processing:")
    ready_terms = storage.get_terms_ready_for_processing()
    for term in ready_terms:
        print(f"   • {term['term']} (frequency: {term['frequency']})")
    
    # Export to JSON
    print("\n📤 Exporting to JSON...")
    export_file = storage.export_to_json()
    print(f"   Exported to: {export_file}")


if __name__ == "__main__":
    main()
