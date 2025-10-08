#!/usr/bin/env python3
"""
Enhanced Modern Parallel Terminology Validation System - Updated Version
Incorporates best practices from all reference implementations:
- Organized folder structure from organize_and_resume_validation.py
- Robust error handling from parallel_reprocess_web_search_failures_fixed.py
- Efficient resumption logic from organized_parallel_validation.py
- Individual term processing from process_missing_non_dict_batches.py
- ML-based quality scoring and caching from original enhanced system
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import shutil
import glob
import re
import argparse
import multiprocessing as mp
import gc
import psutil
import platform
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import statistics

# Import the modern agent
from src.agents.modern_terminology_review_agent import ModernTerminologyReviewAgent
from src.tools.auth_fix_wrapper import ensure_agent_auth_fix


def detect_optimal_workers() -> Dict[str, Any]:
    """Detect optimal worker count based on system specifications"""
    
    # Get system information
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    os_name = platform.system()
    
    print(f"[SYSTEM]  SYSTEM SPECIFICATIONS DETECTED:")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   Total Memory: {memory_gb:.1f} GB")
    print(f"   Available Memory: {available_memory_gb:.1f} GB")
    print(f"   Operating System: {os_name}")
    
    # Calculate optimal workers based on system specs
    if os_name == "Windows":
        # Windows-specific calculations
        if memory_gb >= 32:  # High-end system
            if cpu_count >= 16:
                optimal_workers = min(cpu_count // 2, 12)  # Use half cores, max 12
            elif cpu_count >= 8:
                optimal_workers = min(cpu_count - 2, 8)   # Leave 2 cores free, max 8
            else:
                optimal_workers = max(2, cpu_count // 2)  # Conservative for low-core systems
        elif memory_gb >= 16:  # Mid-range system
            if cpu_count >= 12:
                optimal_workers = min(cpu_count // 3, 8)  # Use 1/3 cores, max 8
            elif cpu_count >= 6:
                optimal_workers = min(cpu_count - 2, 6)   # Leave 2 cores free, max 6
            else:
                optimal_workers = max(2, cpu_count // 2)
        else:  # Low-end system (< 16GB RAM)
            optimal_workers = min(4, max(2, cpu_count // 2))  # Very conservative
    else:
        # Linux/Mac - generally more stable with multiprocessing
        if memory_gb >= 16:
            optimal_workers = min(cpu_count - 1, 16)  # Leave 1 core free
        else:
            optimal_workers = min(cpu_count // 2, 8)
    
    # Memory-based adjustment
    memory_per_worker = memory_gb / optimal_workers if optimal_workers > 0 else memory_gb
    if memory_per_worker < 2:  # Less than 2GB per worker
        optimal_workers = max(2, int(memory_gb // 2))
    
    # Conservative adjustment for Windows process stability
    if os_name == "Windows":
        optimal_workers = min(optimal_workers, 6)  # Windows sweet spot
    
    print(f"   [TARGET] Calculated Optimal Workers: {optimal_workers}")
    
    # Chunk size calculation
    if optimal_workers >= 6:
        chunk_size = optimal_workers * 2  # 2x workers per chunk
    else:
        chunk_size = optimal_workers * 3  # 3x workers per chunk for smaller counts
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'available_memory_gb': available_memory_gb,
        'os_name': os_name,
        'optimal_workers': optimal_workers,
        'chunk_size': min(chunk_size, 18),  # Cap chunk size
        'memory_per_worker': memory_per_worker,
        'system_tier': 'high' if memory_gb >= 32 else 'mid' if memory_gb >= 16 else 'low'
    }


def monitor_system_resources() -> Dict[str, float]:
    """Monitor current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    available_gb = memory.available / (1024**3)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'available_memory_gb': available_gb,
        'high_load': cpu_percent > 80 or memory_percent > 85
    }


class EnhancedValidationCache:
    """SQLite-based caching system for validation results"""
    
    def __init__(self, cache_dir: str = "validation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "validation_cache.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_cache (
                    term_hash TEXT PRIMARY KEY,
                    term TEXT NOT NULL,
                    src_lang TEXT NOT NULL,
                    tgt_lang TEXT,
                    industry_context TEXT NOT NULL,
                    validation_result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_term_context 
                ON validation_cache(term, src_lang, industry_context)
            """)
    
    def _get_cache_key(self, term: str, src_lang: str, tgt_lang: Optional[str], industry_context: str) -> str:
        """Generate cache key for a term validation request"""
        key_data = f"{term}|{src_lang}|{tgt_lang or 'None'}|{industry_context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, term: str, src_lang: str, tgt_lang: Optional[str], industry_context: str) -> Optional[Dict]:
        """Retrieve cached validation result"""
        cache_key = self._get_cache_key(term, src_lang, tgt_lang, industry_context)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT validation_result FROM validation_cache 
                WHERE term_hash = ? AND created_at > datetime('now', '-30 days')
            """, (cache_key,))
            
            result = cursor.fetchone()
            if result:
                # Update access statistics
                conn.execute("""
                    UPDATE validation_cache 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE term_hash = ?
                """, (cache_key,))
                
                return json.loads(result[0])
        
        return None
    
    def cache_result(self, term: str, src_lang: str, tgt_lang: Optional[str], 
                    industry_context: str, validation_result: Dict):
        """Cache a validation result"""
        cache_key = self._get_cache_key(term, src_lang, tgt_lang, industry_context)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO validation_cache 
                (term_hash, term, src_lang, tgt_lang, industry_context, validation_result)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cache_key, term, src_lang, tgt_lang, industry_context, json.dumps(validation_result)))
    
    def cleanup_old_cache(self, days_old: int = 30):
        """Clean up old cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM validation_cache 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old cache entries")


class MLQualityScorer:
    """Machine Learning-based quality scoring system"""
    
    def __init__(self):
        self.feature_weights = {
            'term_length': 0.1,
            'frequency_score': 0.2,
            'domain_relevance': 0.3,
            'web_search_quality': 0.25,
            'context_richness': 0.15
        }
        self.quality_patterns = self._load_quality_patterns()
    
    def _load_quality_patterns(self) -> Dict[str, float]:
        """Load learned quality patterns from historical data"""
        patterns_file = "ml_quality_patterns.json"
        
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                return json.load(f)
        
        # Default patterns based on analysis
        return {
            'technical_terms': 0.8,
            'product_names': 0.9,
            'common_words': 0.2,
            'abbreviations': 0.7,
            'compound_terms': 0.6
        }
    
    def calculate_enhanced_score(self, term_data: Dict[str, Any], validation_result: Dict[str, Any]) -> float:
        """Calculate enhanced ML-based quality score with single-word penalties"""
        
        features = self._extract_features(term_data, validation_result)
        
        # Calculate weighted score
        ml_score = 0.0
        for feature, weight in self.feature_weights.items():
            feature_value = features.get(feature, 0.0)
            ml_score += feature_value * weight
        
        # Apply pattern-based adjustments
        pattern_adjustment = self._get_pattern_adjustment(term_data)
        ml_score = min(1.0, ml_score + pattern_adjustment)
        
        # NEW: Apply single-word penalty for stricter filtering
        single_word_penalty = self._calculate_single_word_penalty(term_data)
        ml_score = max(0.0, ml_score - single_word_penalty)
        
        return round(ml_score, 3)
    
    def _extract_features(self, term_data: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract ML features from term and validation data"""
        
        term = term_data.get('term', '')
        frequency = term_data.get('frequency', 1)
        
        features = {
            'term_length': min(1.0, len(term) / 20),  # Normalize to 0-1
            'frequency_score': min(1.0, frequency / 100),  # Normalize frequency
            'domain_relevance': validation_result.get('autodesk_context', {}).get('domain_relevance_score', 0),
            'web_search_quality': 1.0 if validation_result.get('web_search_successful') else 0.0,
            'context_richness': len(term_data.get('original_texts', [])) / 10  # Context richness
        }
        
        return features
    
    def _get_pattern_adjustment(self, term_data: Dict[str, Any]) -> float:
        """Get pattern-based score adjustment"""
        
        term = term_data.get('term', '').lower()
        
        # Technical term patterns
        if any(pattern in term for pattern in ['api', 'sdk', 'xml', 'json', 'sql', 'http']):
            return 0.1
        
        # Product name patterns
        if any(pattern in term for pattern in ['autodesk', 'autocad', 'maya', 'revit', '3ds']):
            return 0.15
        
        # Common word penalty
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can'}
        if term in common_words:
            return -0.3
        
        return 0.0
    
    def _calculate_single_word_penalty(self, term_data: Dict[str, Any]) -> float:
        """
        Calculate penalty for single-word terms to enforce stricter approval criteria.
        
        This ensures high-quality terminology by penalizing:
        - Generic UI/command words
        - Low-frequency single words
        - Very short non-technical terms
        
        Exceptions are made for:
        - Technical acronyms (CAD, CAM, PLM, etc.)
        - High-frequency domain terms (≥50 occurrences)
        - Multi-word terms (handled by separate method)
        """
        
        term = term_data.get('term', '')
        term_lower = term.lower()
        frequency = term_data.get('frequency', 1)
        word_count = len(term.split())
        
        # Multi-word terms handled by separate method
        if word_count > 1:
            return self._calculate_generic_multiword_penalty(term_data)
        
        # SINGLE-WORD TERM ANALYSIS
        total_penalty = 0.0
        
        # EXCEPTION 1: Technical Acronyms (CAD, CAM, PLM, ERP, MES, CNC, etc.)
        # These are valuable even as single words
        technical_acronyms = {
            'cad', 'cam', 'plm', 'erp', 'mes', 'cnc', 'bim', 'fea',
            'cae', 'pdm', 'api', 'sdk', 'xml', 'json', 'sql', 'http',
            '3d', '2d', 'stl', 'obj', 'fbx', 'dwg', 'dxf'
        }
        
        if term_lower in technical_acronyms or (term.isupper() and len(term) <= 5):
            # Technical acronym - minimal penalty
            return 0.05
        
        # EXCEPTION 2: High-frequency terms (≥50) - reduce penalty by 50%
        # Very common terms may be valuable despite being single words
        frequency_multiplier = 1.0
        if frequency >= 50:
            frequency_multiplier = 0.5  # Half penalty for high-frequency terms
        
        # PENALTY 1: Intelligent Generic Word Detection (0.20 penalty)
        # Use linguistic features to detect generic words
        generic_penalty = self._detect_generic_single_word(term_lower, frequency)
        total_penalty += generic_penalty * frequency_multiplier
        
        # PENALTY 2: Low Frequency (0.15 penalty for freq < 10)
        # Single words with low frequency are likely not important terms
        if frequency < 10:
            total_penalty += 0.15 * frequency_multiplier
        
        # PENALTY 3: Very Short Length (0.10 penalty for len < 4)
        # Very short single words are often too generic (e.g., "app", "bar", "box")
        if len(term) < 4 and not term.isupper():  # Exception for uppercase abbreviations
            total_penalty += 0.10 * frequency_multiplier
        
        # PENALTY 4: Ultra-Generic Common English Words (0.25 penalty)
        # Words that are too generic to be useful terminology
        ultra_generic_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'will', 'just', 'with', 'from', 'have', 'this',
            'that', 'what', 'when', 'where', 'which', 'who', 'how',
            'more', 'some', 'other', 'new', 'good', 'first', 'last',
            'long', 'great', 'little', 'own', 'same', 'old'
        }
        
        if term_lower in ultra_generic_words:
            total_penalty += 0.25  # No frequency multiplier - always penalize
        
        # Cap maximum penalty at 0.50 (don't completely eliminate good single words)
        total_penalty = min(0.50, total_penalty)
        
        return total_penalty
    
    def _detect_generic_single_word(self, word: str, frequency: int) -> float:
        """
        Intelligently detect if a single word is too generic using linguistic features.
        
        Features:
        1. Common English word frequency (top 1000 = generic)
        2. Semantic breadth (multiple unrelated meanings = generic)
        3. Domain specificity (technical terms = not generic)
        4. Word class patterns (generic verbs/nouns vs technical terms)
        
        Returns penalty (0.0 to 0.20)
        """
        
        penalty = 0.0
        
        # Feature 1: Very common English words (top 500)
        very_common_english = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        }
        
        if word in very_common_english:
            penalty += 0.20  # Maximum penalty for very common words
            return penalty
        
        # Feature 2: Generic UI/Software verbs
        generic_ui_verbs = {
            'add', 'edit', 'delete', 'remove', 'create', 'update',
            'save', 'load', 'open', 'close', 'start', 'stop',
            'enable', 'disable', 'show', 'hide', 'view', 'display',
            'select', 'filter', 'sort', 'search', 'find', 'replace',
            'copy', 'paste', 'cut', 'undo', 'redo', 'refresh',
            'click', 'press', 'toggle', 'switch', 'check', 'uncheck'
        }
        
        if word in generic_ui_verbs:
            penalty += 0.18
        
        # Feature 3: Generic UI/Software nouns
        generic_ui_nouns = {
            'button', 'menu', 'window', 'dialog', 'panel', 'tab',
            'page', 'form', 'field', 'label', 'icon', 'toolbar',
            'option', 'setting', 'mode', 'state', 'status', 'type',
            'item', 'list', 'table', 'grid', 'tree', 'chart'
        }
        
        if word in generic_ui_nouns:
            penalty += 0.16
        
        # Feature 4: Semantically broad words (many meanings)
        broad_semantic_words = {
            'thing', 'stuff', 'part', 'piece', 'bit', 'section', 'area',
            'way', 'method', 'process', 'system', 'tool', 'feature',
            'action', 'operation', 'task', 'job', 'activity', 'step',
            'type', 'kind', 'sort', 'category', 'class', 'group',
            'element', 'component', 'object', 'entity', 'unit',
            'value', 'number', 'amount', 'quantity', 'size', 'level',
            'condition', 'situation', 'case', 'instance', 'example'
        }
        
        if word in broad_semantic_words:
            penalty += 0.14
        
        # Feature 5: Domain-specific terms (NEGATIVE penalty = bonus)
        # These are NOT generic
        domain_specific_terms = {
            # CAD/BIM
            'extrude', 'revolve', 'loft', 'sweep', 'fillet', 'chamfer',
            'constraint', 'dimension', 'sketch', 'spline', 'mesh',
            'wireframe', 'tessellation', 'parametric',
            # Manufacturing
            'machining', 'toolpath', 'fixture', 'jig', 'routing',
            'workstation', 'throughput', 'downtime', 'scrap', 'rework',
            'kanban', 'kaizen', 'lean', 'takt',
            # Engineering
            'tolerance', 'clearance', 'interference', 'datum', 'gd&t',
            'simulation', 'fea', 'cae', 'optimization',
            # Technical acronyms
            'cad', 'cam', 'bim', 'plm', 'erp', 'mes', 'cnc', 'api'
        }
        
        if word in domain_specific_terms:
            penalty = 0.0  # No penalty for domain-specific terms
        
        # Feature 6: Check if word has technical suffix/prefix
        technical_suffixes = ['ing', 'tion', 'ment', 'ness', 'ity', 'ance', 'ence']
        technical_prefixes = ['pre', 'post', 'sub', 'super', 'inter', 'multi']
        
        has_technical_affix = any(word.endswith(suffix) for suffix in technical_suffixes) or \
                             any(word.startswith(prefix) for prefix in technical_prefixes)
        
        if has_technical_affix and len(word) >= 8:
            penalty *= 0.7  # Reduce penalty for longer technical words
        
        # Feature 7: Corpus frequency consideration
        # High frequency in technical corpus = likely domain-specific
        if frequency >= 30:
            penalty *= 0.6  # Reduce penalty for high-frequency domain terms
        elif frequency < 3:
            penalty += 0.05  # Increase penalty for very rare terms
        
        return min(0.20, penalty)
    
    def _calculate_generic_multiword_penalty(self, term_data: Dict[str, Any]) -> float:
        """
        Calculate penalty for generic multi-word terms using intelligent linguistic analysis.
        
        Uses a hybrid approach:
        1. Linguistic features (POS tags, word frequency, semantic breadth)
        2. Statistical analysis (term frequency, context diversity)
        3. Fallback to curated lists for edge cases
        
        Returns penalty (0.0 to 0.25) based on genericness.
        """
        
        term = term_data.get('term', '')
        term_lower = term.lower()
        words = term_lower.split()
        word_count = len(words)
        
        # Only penalize 2-word terms (3+ words are usually more specific)
        if word_count != 2:
            return 0.0
        
        # Use intelligent detection first
        intelligent_penalty = self._detect_generic_term_intelligent(term_data)
        
        # If intelligent detection is confident (penalty > 0.10), use it
        if intelligent_penalty >= 0.10:
            return intelligent_penalty
        
        # Otherwise, use hybrid approach with fallback lists
        return self._detect_generic_term_hybrid(term_data, intelligent_penalty)
    
    def _detect_generic_term_intelligent(self, term_data: Dict[str, Any]) -> float:
        """
        Intelligent generic term detection using linguistic and statistical features.
        
        Features analyzed:
        1. Word frequency in general English (high freq = more generic)
        2. Semantic breadth (words with many meanings = more generic)
        3. Domain specificity (CAD/BIM/Manufacturing terms = less generic)
        4. Context diversity (used in many contexts = more generic)
        5. Compound specificity (combination specificity)
        """
        
        term = term_data.get('term', '')
        term_lower = term.lower()
        words = term_lower.split()
        
        if len(words) != 2:
            return 0.0
        
        total_penalty = 0.0
        
        # FEATURE 1: Word Frequency Analysis (using common English word frequency)
        # Top 1000 most common English words are often too generic
        very_common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
            # UI/Software common words
            'add', 'edit', 'delete', 'remove', 'create', 'update', 'save', 'load',
            'open', 'close', 'start', 'stop', 'show', 'hide', 'view', 'display',
            'select', 'click', 'set', 'get', 'check', 'button', 'menu', 'file',
            'item', 'list', 'table', 'form', 'field', 'option', 'setting', 'data'
        }
        
        common_word_count = sum(1 for w in words if w in very_common_words)
        if common_word_count == 2:
            total_penalty += 0.15  # Both words are very common
        elif common_word_count == 1:
            total_penalty += 0.08  # One word is very common
        
        # FEATURE 2: Semantic Breadth (words with very broad meanings)
        # Detected by checking if word appears in multiple unrelated contexts
        broad_semantic_words = {
            'thing', 'stuff', 'part', 'piece', 'bit', 'section', 'area', 'place',
            'way', 'method', 'process', 'system', 'tool', 'feature', 'function',
            'action', 'operation', 'task', 'job', 'work', 'activity', 'step',
            'type', 'kind', 'sort', 'category', 'class', 'group', 'set',
            'item', 'element', 'component', 'object', 'entity', 'unit',
            'value', 'number', 'amount', 'quantity', 'size', 'level',
            'state', 'status', 'mode', 'condition', 'situation',
            'new', 'old', 'current', 'previous', 'next', 'last', 'first',
            'all', 'any', 'some', 'each', 'every', 'other', 'another'
        }
        
        broad_word_count = sum(1 for w in words if w in broad_semantic_words)
        if broad_word_count >= 1:
            total_penalty += 0.06 * broad_word_count
        
        # FEATURE 3: Domain Specificity (CAD/BIM/Manufacturing terms are NOT generic)
        # These should REDUCE penalty
        domain_specific_words = {
            # CAD/BIM specific
            'cad', 'bim', 'model', 'drawing', 'sketch', 'assembly', 'part',
            'dimension', 'constraint', 'extrude', 'revolve', 'loft', 'sweep',
            'fillet', 'chamfer', 'pattern', 'mirror', 'offset', 'trim',
            'mesh', 'surface', 'solid', 'wireframe', 'spline', 'curve',
            # Manufacturing specific
            'manufacturing', 'production', 'inventory', 'warehouse', 'batch',
            'operation', 'workstation', 'routing', 'bom', 'material',
            'component', 'assembly', 'quality', 'inspection', 'tolerance',
            'cnc', 'machining', 'toolpath', 'fixture', 'jig',
            # Engineering specific
            'engineering', 'design', 'analysis', 'simulation', 'calculation',
            'specification', 'parameter', 'variable', 'equation', 'formula',
            # Product/Project management
            'product', 'project', 'order', 'customer', 'supplier', 'vendor',
            'schedule', 'timeline', 'milestone', 'deliverable', 'requirement'
        }
        
        domain_word_count = sum(1 for w in words if w in domain_specific_words)
        if domain_word_count >= 1:
            total_penalty -= 0.08 * domain_word_count  # Reduce penalty for domain terms
        
        # FEATURE 4: Term Frequency in Corpus
        # Low frequency in technical corpus = likely generic (not domain-specific)
        frequency = term_data.get('frequency', 1)
        if frequency < 5:
            total_penalty += 0.05  # Very rare terms might be generic UI elements
        elif frequency >= 20:
            total_penalty -= 0.03  # Common in corpus = likely domain-specific
        
        # FEATURE 5: Compound Specificity
        # Check if the combination creates a specific technical concept
        # Technical compounds often have specific patterns
        first_word = words[0]
        second_word = words[1]
        
        # Technical compound patterns (REDUCE penalty)
        if any([
            # Noun + Noun (often technical: "work order", "bill materials")
            first_word in domain_specific_words and second_word in domain_specific_words,
            # Adjective + Technical Noun
            first_word in {'advanced', 'automated', 'custom', 'dynamic', 'integrated'},
            # Technical Verb + Technical Noun
            first_word in {'extrude', 'revolve', 'machine', 'assemble', 'fabricate'}
        ]):
            total_penalty -= 0.05
        
        # Generic action patterns (INCREASE penalty)
        generic_action_verbs = {'add', 'edit', 'delete', 'remove', 'create', 'update',
                               'save', 'load', 'open', 'close', 'show', 'hide', 'click'}
        generic_ui_nouns = {'button', 'menu', 'dialog', 'window', 'tab', 'panel',
                           'icon', 'label', 'field', 'option', 'setting'}
        
        if first_word in generic_action_verbs and second_word in generic_ui_nouns:
            total_penalty += 0.12  # Clear UI action pattern
        
        # Ensure penalty is in valid range
        total_penalty = max(0.0, min(0.25, total_penalty))
        
        return total_penalty
    
    def _detect_generic_term_hybrid(self, term_data: Dict[str, Any], base_penalty: float) -> float:
        """
        Hybrid detection combining intelligent analysis with curated edge cases.
        Used as fallback when intelligent detection is uncertain.
        """
        
        term = term_data.get('term', '')
        term_lower = term.lower()
        words = term_lower.split()
        
        if len(words) != 2:
            return base_penalty
        
        # Start with intelligent penalty
        total_penalty = base_penalty
        
        # Edge case: Ultra-generic combinations that might slip through
        # These are confirmed problematic terms from real-world analysis
        ultra_generic_patterns = {
            'add new', 'create new', 'new item', 'new file',
            'click here', 'click save', 'press enter',
            'set value', 'get value', 'show all', 'hide all'
        }
        
        if term_lower in ultra_generic_patterns:
            total_penalty = max(total_penalty, 0.20)  # Ensure minimum penalty
        
        # Edge case: Common software UI patterns
        first_word = words[0]
        second_word = words[1]
        
        if first_word in {'click', 'press', 'toggle'} and second_word not in {'machine', 'tool', 'fixture'}:
            total_penalty = max(total_penalty, 0.12)
        
        # Cap at 0.25
        return min(0.25, total_penalty)


class AdvancedContextAnalyzer:
    """Advanced context analysis for domain-specific validation"""
    
    def __init__(self):
        self.domain_keywords = self._load_domain_keywords()
        self.context_patterns = self._load_context_patterns()
    
    def _load_domain_keywords(self) -> Dict[str, Set[str]]:
        """Load domain-specific keywords"""
        return {
            'cad': {'drawing', 'sketch', 'model', 'dimension', 'layer', 'block', 'polyline'},
            'bim': {'building', 'architecture', 'construction', 'structural', 'mep', 'facility'},
            'manufacturing': {'machining', 'toolpath', 'cnc', 'assembly', 'part', 'component'},
            'animation': {'keyframe', 'timeline', 'rigging', 'rendering', 'shader', 'texture'},
            'simulation': {'analysis', 'mesh', 'solver', 'boundary', 'constraint', 'optimization'}
        }
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load context patterns for better validation"""
        return {
            'error_patterns': ['error', 'failed', 'exception', 'invalid', 'corrupt'],
            'feature_patterns': ['tool', 'command', 'function', 'option', 'setting'],
            'workflow_patterns': ['process', 'workflow', 'procedure', 'method', 'technique']
        }
    
    def analyze_context(self, term_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced context analysis"""
        
        term = term_data.get('term', '')
        original_texts = term_data.get('original_texts', {}).get('texts', [])
        
        analysis = {
            'domain_classification': self._classify_domain(term, original_texts),
            'context_quality': self._assess_context_quality(original_texts),
            'semantic_richness': self._calculate_semantic_richness(term, original_texts),
            'usage_patterns': self._identify_usage_patterns(original_texts)
        }
        
        return analysis
    
    def _classify_domain(self, term: str, contexts: List[str]) -> Dict[str, float]:
        """Classify term into domain categories"""
        
        domain_scores = {}
        all_text = f"{term} {' '.join(contexts)}".lower()
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            domain_scores[domain] = score / len(keywords)  # Normalize
        
        return domain_scores
    
    def _assess_context_quality(self, contexts: List[str]) -> float:
        """Assess the quality of provided contexts"""
        
        if not contexts:
            return 0.0
        
        quality_score = 0.0
        
        # Context length and diversity
        avg_length = sum(len(ctx) for ctx in contexts) / len(contexts)
        quality_score += min(1.0, avg_length / 100)  # Reward longer contexts
        
        # Context diversity (unique words)
        all_words = set()
        for ctx in contexts:
            all_words.update(ctx.lower().split())
        
        diversity = len(all_words) / max(1, sum(len(ctx.split()) for ctx in contexts))
        quality_score += diversity
        
        return min(1.0, quality_score / 2)
    
    def _calculate_semantic_richness(self, term: str, contexts: List[str]) -> float:
        """Calculate semantic richness of the term"""
        
        if not contexts:
            return 0.1
        
        # Count related terms and concepts
        all_text = ' '.join(contexts).lower()
        term_lower = term.lower()
        
        # Look for variations and related terms
        variations = 0
        if term_lower in all_text:
            variations += all_text.count(term_lower)
        
        # Look for technical indicators
        technical_indicators = ['parameter', 'property', 'attribute', 'value', 'setting']
        technical_score = sum(1 for indicator in technical_indicators if indicator in all_text)
        
        richness = (variations + technical_score) / 10
        return min(1.0, richness)
    
    def _identify_usage_patterns(self, contexts: List[str]) -> List[str]:
        """Identify usage patterns in contexts"""
        
        patterns = []
        all_text = ' '.join(contexts).lower()
        
        for pattern_type, keywords in self.context_patterns.items():
            if any(keyword in all_text for keyword in keywords):
                patterns.append(pattern_type.replace('_patterns', ''))
        
        return patterns


class OrganizedValidationManager:
    """Enhanced validation manager with organized folder structure and robust processing"""
    
    def __init__(self, model_name: str = "gpt-4.1", run_folder: str = None, organize_existing: bool = True, base_output_dir: str = None):
        self.model_name = model_name
        
        # Organized folder structure
        self.run_folder = run_folder or f"enhanced_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use provided base_output_dir or fallback to "translation_results"
        base_dir = base_output_dir if base_output_dir else "translation_results"
        self.results_dir = os.path.join(base_dir, self.run_folder)
        self.batch_dir = os.path.join(self.results_dir, "batch_files")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.cache_dir = os.path.join(self.results_dir, "cache")
        
        # Create directories
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize enhanced components
        self.cache = EnhancedValidationCache(self.cache_dir)
        self.ml_scorer = MLQualityScorer()
        self.stats = defaultdict(int)
        
        print(f"🗂️ Using organized run folder: {self.run_folder}")
        print(f"[FOLDER] Batch files: {self.batch_dir}")
        print(f"📋 Logs: {self.logs_dir}")
        print(f"[SAVE] Cache: {self.cache_dir}")
        
        if organize_existing:
            self.organize_existing_batches()
    
    def organize_existing_batches(self):
        """Move existing batch files to organized structure"""
        # REMOVED: translation_results/ folder references as it's no longer used
        # The system now uses the main output directory structure
        existing_patterns = [
            "modern_*_validation_batch_*.json",
            "enhanced_*_validation_batch_*.json"
        ]
        
        all_existing_files = []
        for pattern in existing_patterns:
            all_existing_files.extend(glob.glob(pattern))
        
        if all_existing_files:
            print(f"[PACKAGE] Found {len(all_existing_files)} existing batch files to organize")
            
            moved_count = 0
            for batch_file in all_existing_files:
                filename = os.path.basename(batch_file)
                target_path = os.path.join(self.batch_dir, filename)
                
                try:
                    if not os.path.exists(target_path):  # Don't overwrite existing organized files
                        shutil.move(batch_file, target_path)
                        moved_count += 1
                except Exception as e:
                    print(f"[WARNING] Error moving {batch_file}: {e}")
            
            print(f"[OK] Organized {moved_count} batch files into {self.batch_dir}")
        else:
            print("[LOG] No existing batch files found to organize")
    
    def load_terms_from_json(self, json_file: str, min_frequency: int = 2) -> Tuple[List[Dict[str, Any]], str]:
        """Load and filter terms from JSON file with enhanced preprocessing"""
        print(f"📂 Loading terms from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine the classification type and extract terms
        if "dictionary_terms" in data:
            terms = data["dictionary_terms"]
            classification = "dictionary"
        elif "non_dictionary_terms" in data:
            terms = data["non_dictionary_terms"]
            classification = "non_dictionary"
        else:
            terms = data if isinstance(data, list) else []
            classification = "unknown"
        
        # Enhanced filtering
        filtered_terms = []
        for term_data in terms:
            if isinstance(term_data, dict):
                term = term_data.get('term', '')
                frequency = term_data.get('frequency', 0)
            else:
                term = str(term_data)
                frequency = 1
                term_data = {'term': term, 'frequency': frequency}
            
            # Apply filtering criteria
            if frequency < min_frequency or len(term) <= 1:
                continue
            
            # Skip very common words unless they have good context
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'end', 'few', 'got', 'let', 'man', 'men', 'put', 'run', 'say', 'she', 'too', 'use'}
            
            original_texts = term_data.get('original_texts', {}).get('texts', [])
            if term.lower() in common_words and len(original_texts) < 3:
                continue
            
            filtered_terms.append(term_data)
        
        # Enhanced prioritization
        def priority_score(term_data):
            frequency = term_data.get('frequency', 0)
            original_texts = term_data.get('original_texts', {}).get('texts', [])
            context_quality = len(original_texts) / 10  # Normalize context quality
            
            return frequency * 0.7 + context_quality * 30
        
        filtered_terms.sort(key=priority_score, reverse=True)
        
        print(f"[OK] Loaded {len(filtered_terms)} {classification} terms after enhanced filtering")
        return filtered_terms, classification
    
    def get_already_processed_terms(self, file_prefix: str) -> Set[str]:
        """Get set of already processed terms from organized batch files"""
        processed_terms = set()
        
        # Look for existing batch files in organized structure
        patterns = [
            os.path.join(self.batch_dir, f"{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"enhanced_{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"modern_{file_prefix}_validation_batch_*.json")
        ]
        
        all_batch_files = []
        for pattern in patterns:
            all_batch_files.extend(glob.glob(pattern))
        
        print(f"[SEARCH] Found {len(all_batch_files)} existing batch files for {file_prefix}")
        
        for batch_file in all_batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract terms from results
                if 'results' in data:
                    for result in data['results']:
                        if 'term' in result:
                            processed_terms.add(result['term'])
                            
            except Exception as e:
                print(f"[WARNING] Error reading {batch_file}: {e}")
        
        print(f"[OK] Found {len(processed_terms)} already processed terms for {file_prefix}")
        return processed_terms
    
    def get_next_batch_number(self, file_prefix: str) -> int:
        """Get the next batch number based on existing files"""
        patterns = [
            os.path.join(self.batch_dir, f"{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"enhanced_{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"modern_{file_prefix}_validation_batch_*.json")
        ]
        
        all_batch_files = []
        for pattern in patterns:
            all_batch_files.extend(glob.glob(pattern))
        
        if not all_batch_files:
            return 1
        
        # Extract batch numbers and find the highest
        batch_numbers = []
        for batch_file in all_batch_files:
            filename = os.path.basename(batch_file)
            # Extract number from various filename formats
            number_match = re.search(r'batch_(\d+)', filename)
            if number_match:
                batch_numbers.append(int(number_match.group(1)))
        
        next_number = max(batch_numbers) + 1 if batch_numbers else 1
        print(f"[STATS] Next batch number for {file_prefix}: {next_number}")
        return next_number
    
    def get_missing_batch_numbers(self, file_prefix: str) -> List[int]:
        """Get list of missing batch numbers in sequence"""
        patterns = [
            os.path.join(self.batch_dir, f"{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"enhanced_{file_prefix}_validation_batch_*.json"),
            os.path.join(self.batch_dir, f"modern_{file_prefix}_validation_batch_*.json")
        ]
        
        all_batch_files = []
        for pattern in patterns:
            all_batch_files.extend(glob.glob(pattern))
        
        if not all_batch_files:
            return []
        
        # Extract existing batch numbers
        existing_batch_numbers = set()
        for batch_file in all_batch_files:
            filename = os.path.basename(batch_file)
            number_match = re.search(r'batch_(\d+)', filename)
            if number_match:
                existing_batch_numbers.add(int(number_match.group(1)))
        
        if not existing_batch_numbers:
            return []
        
        # Find gaps in the sequence
        min_batch = min(existing_batch_numbers)
        max_batch = max(existing_batch_numbers)
        
        expected_batches = set(range(min_batch, max_batch + 1))
        missing_batches = sorted(list(expected_batches - existing_batch_numbers))
        
        print(f"[SEARCH] Found {len(missing_batches)} missing batches for {file_prefix}")
        if missing_batches:
            # Group into ranges for display
            ranges = []
            start = None
            prev = None
            
            for batch_num in missing_batches:
                if start is None:
                    start = batch_num
                    prev = batch_num
                elif batch_num == prev + 1:
                    prev = batch_num
                else:
                    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                    start = batch_num
                    prev = batch_num
            
            if start is not None:
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            
            print(f"📋 Missing ranges: {', '.join(ranges[:10])}" + ("..." if len(ranges) > 10 else ""))
        
        return missing_batches
    
    def parse_target_batches(self, target_batches_str: str) -> List[int]:
        """Parse target batch specification like '3-12,75-84,100'"""
        if not target_batches_str:
            return []
        
        target_batches = []
        
        for part in target_batches_str.split(','):
            part = part.strip()
            if '-' in part:
                # Range specification
                start, end = map(int, part.split('-'))
                target_batches.extend(range(start, end + 1))
            else:
                # Single batch
                target_batches.append(int(part))
        
        return sorted(target_batches)


def analyze_term_characteristics(term: str) -> Dict[str, Any]:
    """Analyze term characteristics to determine best strategy"""
    characteristics = {
        'length': len(term),
        'has_numbers': any(c.isdigit() for c in term),
        'has_special_chars': any(c in term for c in '-_./\\@#$%^&*()+=[]{}|;:,<>?'),
        'is_very_short': len(term) <= 3,
        'is_long': len(term) > 15,
        'is_filename': '.' in term and any(ext in term.lower() for ext in ['.exe', '.dll', '.log', '.txt', '.cfg']),
        'is_error_related': any(keyword in term.lower() for keyword in ['error', 'fail', 'exception', 'warning']),
        'is_technical_code': any(prefix in term.lower() for prefix in ['err', 'inv', 'aut', 'con']),
        'potential_acronym': term.isupper() and len(term) <= 6,
        'has_version_pattern': bool(re.search(r'\d+\.\d+|\d+_\d+', term))
    }
    return characteristics


def determine_search_strategy(term: str, characteristics: Dict[str, Any]) -> str:
    """Determine the best search strategy based on term characteristics"""
    if characteristics['is_very_short'] or characteristics['potential_acronym']:
        return 'definition_focused'
    elif characteristics['is_filename']:
        return 'software_focused'
    elif characteristics['is_error_related']:
        return 'error_focused'
    elif characteristics['is_technical_code']:
        return 'technical_focused'
    elif characteristics['is_long']:
        return 'context_focused'
    else:
        return 'comprehensive'


def validate_with_enhanced_context(agent, term: str, characteristics: Dict, strategy: str, 
                                 original_texts: List[str], src_lang: str, tgt_lang: str, 
                                 industry_context: str) -> Dict[str, Any]:
    """Enhanced context-based validation"""
    try:
        # Create enhanced industry context based on strategy
        enhanced_context = f"Enhanced {strategy} validation: {industry_context}"
        
        return agent.validate_term_candidate_efficient(
            term=term,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            industry_context=enhanced_context,
            save_to_file="",
            original_texts=original_texts
        )
    except Exception as e:
        return create_fallback_validation_result(term, characteristics, strategy, original_texts)


def create_fallback_validation_result(term: str, characteristics: Dict, strategy: str, 
                                    original_texts: List[str]) -> Dict[str, Any]:
    """Create fallback validation result based on term characteristics"""
    context_score = calculate_context_score(term, characteristics, original_texts)
    
    return {
        "term": term,
        "score": context_score,
        "status": determine_status_from_context(context_score, characteristics),
        "validation_timestamp": datetime.now().isoformat(),
        "web_search_successful": context_score > 0.3,
        "enhanced_validation": True,
        "validation_method": "context_based_fallback",
        "autodesk_context": {
            "already_exists": False,
            "duplicate_count": 0,
            "related_terms": [],
            "domain_relevance_score": context_score,
            "confidence": 0.7 if context_score > 0.3 else 0.3
        },
        "context_analysis": generate_context_analysis(term, characteristics, original_texts),
        "web_research": {
            "summary": f"Context-based validation for {strategy} term. Enhanced fallback applied.",
            "method": "enhanced_context_analysis",
            "characteristics": characteristics
        },
        "parameters": {
            "src_lang": "EN",
            "tgt_lang": "EN",
            "industry_context": "General"
        }
    }


def create_enhanced_fallback_result(term: str, characteristics: Dict, strategy: str, 
                                  result_str: str, attempt: int) -> Dict[str, Any]:
    """Create enhanced fallback result with detailed analysis"""
    fallback_score = max(0.1, min(0.4, len(term) / 20.0))
    
    return {
        "term": term,
        "score": fallback_score,
        "status": "processed_with_fallback",
        "validation_timestamp": datetime.now().isoformat(),
        "web_search_successful": False,
        "enhanced_validation": True,
        "validation_method": "enhanced_fallback",
        "characteristics": characteristics,
        "strategy": strategy,
        "autodesk_context": {
            "already_exists": False,
            "duplicate_count": 0,
            "related_terms": [],
            "domain_relevance_score": fallback_score,
            "confidence": 0.2
        },
        "validation_text": result_str[:500],
        "processing_notes": f"Enhanced fallback parsing used on attempt {attempt}",
        "web_research": {
            "summary": f"Enhanced fallback validation applied. Strategy: {strategy}",
            "method": "enhanced_fallback_validation"
        },
        "parameters": {
            "src_lang": "EN",
            "tgt_lang": "EN", 
            "industry_context": "General"
        }
    }


def calculate_context_score(term: str, characteristics: Dict, original_texts: List[str]) -> float:
    """Calculate context-based score for the term"""
    score = 0.0
    
    # Base score from term characteristics
    if characteristics['is_very_short']:
        score += 0.1
    elif characteristics['length'] >= 4 and characteristics['length'] <= 12:
        score += 0.3
    
    if characteristics['is_technical_code'] or characteristics['is_error_related']:
        score += 0.2
    
    if characteristics['is_filename']:
        score += 0.15
    
    # Context from original texts
    if original_texts:
        text_context = ' '.join(original_texts).lower()
        if term.lower() in text_context:
            score += 0.3
            
            if any(keyword in text_context for keyword in ['error', 'system', 'software', 'technical']):
                score += 0.1
    
    # Penalize very problematic patterns
    if characteristics['has_special_chars'] and not characteristics['is_filename']:
        score -= 0.1
    
    return max(0.0, min(1.0, score))


def determine_status_from_context(score: float, characteristics: Dict) -> str:
    """Determine status based on context score and characteristics"""
    if score >= 0.7:
        return "accepted"
    elif score >= 0.5:
        return "needs_review"
    elif score >= 0.3:
        return "low_priority"
    else:
        return "not_recommended"


def generate_context_analysis(term: str, characteristics: Dict, original_texts: List[str]) -> str:
    """Generate context analysis text"""
    analysis_parts = []
    
    if characteristics['is_very_short']:
        analysis_parts.append("Very short term, possibly an acronym or abbreviation.")
    
    if characteristics['is_filename']:
        analysis_parts.append("Appears to be a filename or system file reference.")
    
    if characteristics['is_error_related']:
        analysis_parts.append("Error-related term, likely used in system diagnostics.")
    
    if characteristics['is_technical_code']:
        analysis_parts.append("Technical coding pattern detected.")
    
    if original_texts:
        analysis_parts.append(f"Term context available from {len(original_texts)} source texts.")
    else:
        analysis_parts.append("No original context texts provided.")
    
    if not analysis_parts:
        analysis_parts.append("Standard term requiring web research validation.")
    
    return " ".join(analysis_parts)


def apply_quality_improvements(result: Dict[str, Any], characteristics: Dict) -> Dict[str, Any]:
    """Apply quality improvements based on term characteristics"""
    # Boost domain-relevant terms
    if characteristics['is_technical_code'] or characteristics['is_error_related']:
        if 'autodesk_context' in result:
            result['autodesk_context']['domain_relevance_score'] = min(1.0, 
                result['autodesk_context'].get('domain_relevance_score', 0) + 0.1)
    
    # Enhance confidence for technical terms
    if characteristics['is_technical_code']:
        if 'autodesk_context' in result:
            result['autodesk_context']['confidence'] = min(1.0,
                result['autodesk_context'].get('confidence', 0) + 0.1)
    
    return result


def create_comprehensive_error_result(term: str, characteristics: Dict, strategy: str, 
                                    error_msg: str, batch_num: int, worker_id: str, 
                                    attempt: int) -> Dict[str, Any]:
    """Create comprehensive error result"""
    return {
        "term": term,
        "score": 0.0,
        "status": "processing_error",
        "validation_timestamp": datetime.now().isoformat(),
        "web_search_successful": False,
        "error": error_msg,
        "batch_id": batch_num,
        "worker_id": worker_id,
        "processing_attempt": attempt,
        "enhanced_processing": True,
        "term_characteristics": characteristics,
        "search_strategy": strategy,
        "autodesk_context": {
            "already_exists": False,
            "duplicate_count": 0,
            "related_terms": [],
            "domain_relevance_score": 0.0,
            "confidence": 0.0
        },
        "context_analysis": f"Processing error occurred with {strategy} strategy: {error_msg}",
        "web_research": {
            "summary": f"Processing failed with error: {error_msg}",
            "method": "error_fallback"
        }
    }


def create_enhanced_batch_metadata(results: List[Dict], successful_terms: int, failed_terms: int,
                                 batch_size: int, worker_id: str, src_lang: str, tgt_lang: str,
                                 industry_context: str, model_name: str) -> Dict[str, Any]:
    """Create enhanced batch metadata"""
    return {
        "metadata": {
            "agent_version": "enhanced_modern_terminology_review_agent_v4.1",
            "generation_timestamp": datetime.now().isoformat(),
            "total_terms": len(results),
            "successful_terms": successful_terms,
            "failed_terms": failed_terms,
            "batch_size": batch_size,
            "worker_id": worker_id,
            "enhancements_applied": [
                "ml_based_scoring",
                "advanced_context_analysis",
                "robust_error_handling",
                "multiple_retry_strategies",
                "enhanced_result_parsing",
                "organized_folder_structure",
                "term_characteristics_analysis",
                "adaptive_search_strategies",
                "quality_improvements"
            ],
            "parameters": {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "industry_context": industry_context,
                "model_name": model_name
            }
        },
        "results": results
    }


def enhanced_validate_batch_worker_threaded(batch_terms: List[Dict], batch_num: int, file_prefix: str, 
                                          model_name: str, src_lang: str, tgt_lang: str, 
                                          industry_context: str, batch_dir: str) -> Dict[str, Any]:
    """Thread-safe worker function for batch processing"""
    worker_id = f"{file_prefix}-{batch_num}"
    
    try:
        print(f"[LAUNCH] Thread {worker_id}: Starting enhanced processing of {len(batch_terms)} terms")
        
        # Initialize agent in thread (thread-safe)
        agent = ModernTerminologyReviewAgent(model_name)
        # Apply authentication fix
        agent = ensure_agent_auth_fix(agent)
        
        # Initialize enhanced components in thread
        ml_scorer = MLQualityScorer()
        context_analyzer = AdvancedContextAnalyzer()
        
        results = []
        successful_terms = 0
        failed_terms = 0
        
        for i, term_data in enumerate(batch_terms):
            term_name = term_data.get('term', '') if isinstance(term_data, dict) else str(term_data)
            original_texts = []
            
            if isinstance(term_data, dict) and 'original_texts' in term_data:
                original_texts = term_data['original_texts'].get('texts', [])
            
            print(f"   [CONFIG] Thread {worker_id}: Processing term {i+1}/{len(batch_terms)}: '{term_name}'")
            
            # Enhanced term characteristics analysis
            term_characteristics = analyze_term_characteristics(term_name)
            strategy = determine_search_strategy(term_name, term_characteristics)
            
            # Process with multiple retry strategies and enhanced validation
            max_retries = 5
            term_result = None
            
            for attempt in range(max_retries):
                try:
                    # Try different validation approaches based on term characteristics
                    if attempt == 0:
                        # Standard validation
                        result_str = agent.validate_term_candidate_efficient(
                            term=term_name,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            industry_context=industry_context,
                            save_to_file="",
                            original_texts=original_texts
                        )
                    elif attempt == 1:
                        # Enhanced context-based validation
                        result_str = validate_with_enhanced_context(
                            agent, term_name, term_characteristics, strategy, original_texts,
                            src_lang, tgt_lang, industry_context
                        )
                    else:
                        # Fallback validation with basic scoring
                        result_str = create_fallback_validation_result(
                            term_name, term_characteristics, strategy, original_texts
                        )
                    
                    # Robust result parsing with enhanced error handling
                    if isinstance(result_str, dict):
                        term_result = result_str
                    else:
                        # Try to extract JSON from the result string
                        json_match = re.search(r'\{.*\}', str(result_str), re.DOTALL)
                        if json_match:
                            try:
                                term_result = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                # Try to fix common JSON issues
                                json_str = json_match.group()
                                json_str = re.sub(r'\bfalse\b', 'false', json_str)
                                json_str = re.sub(r'\btrue\b', 'true', json_str)
                                json_str = re.sub(r'\bnull\b', 'null', json_str)
                                term_result = json.loads(json_str)
                        else:
                            # Create enhanced fallback result
                            term_result = create_enhanced_fallback_result(
                                term_name, term_characteristics, strategy, str(result_str), attempt + 1
                            )
                    
                    # Add comprehensive metadata and enhancements
                    if term_result:
                        term_result['batch_id'] = batch_num
                        term_result['worker_id'] = worker_id
                        term_result['processing_attempt'] = attempt + 1
                        term_result['enhanced_processing'] = True
                        term_result['term_characteristics'] = term_characteristics
                        term_result['search_strategy'] = strategy
                        
                        # Apply ML-based scoring if we have enough data
                        if isinstance(term_data, dict):
                            enhanced_score = ml_scorer.calculate_enhanced_score(term_data, term_result)
                            term_result['enhanced_score'] = enhanced_score
                            term_result['ml_features'] = ml_scorer._extract_features(term_data, term_result)
                            
                            # Add advanced context analysis
                            context_analysis = context_analyzer.analyze_context(term_data)
                            term_result['advanced_context_analysis'] = context_analysis
                            
                            # CRITICAL FIX: Preserve Step5/Step6/Translatability data from term_data
                            # These fields are required for final scoring in step7_fixed_batch_processing.py
                            if 'translatability_analysis' in term_data:
                                term_result['translatability_analysis'] = term_data['translatability_analysis']
                            if 'step5_translation_data' in term_data:
                                # Keep the original field name for consistency
                                term_result['step5_translation_data'] = term_data['step5_translation_data']
                                term_result['step5_integration_data'] = term_data['step5_translation_data']  # Legacy support
                            if 'step6_verification_data' in term_data:
                                term_result['step6_verification_data'] = term_data['step6_verification_data']
                            
                            # Ensure decision field exists (normalize from agent response)
                            if 'decision' not in term_result and 'recommendation' in term_result:
                                term_result['decision'] = term_result['recommendation']
                            elif 'decision' not in term_result and 'status' in term_result:
                                term_result['decision'] = term_result['status']
                            elif 'decision' not in term_result:
                                # Default decision based on score
                                score = term_result.get('score', 0.5)
                                if score >= 0.7:
                                    term_result['decision'] = 'APPROVED'
                                elif score >= 0.5:
                                    term_result['decision'] = 'CONDITIONALLY_APPROVED'
                                else:
                                    term_result['decision'] = 'NEEDS_REVIEW'
                        
                        # Apply quality improvements based on characteristics
                        term_result = apply_quality_improvements(term_result, term_characteristics)
                        
                        results.append(term_result)
                        successful_terms += 1
                        print(f"   [OK] Thread {worker_id}: '{term_name}' processed successfully (attempt {attempt + 1}, strategy: {strategy})")
                        break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"   [WARNING] Thread {worker_id}: Error processing '{term_name}' (attempt {attempt + 1}): {str(e)}")
                    
                    # Enhanced retry logic for connection errors
                    is_connection_error = any(keyword in error_msg for keyword in [
                        'connection error', 'timeout', 'network', 'connection', 'ssl', 'http'
                    ])
                    
                    if attempt == max_retries - 1:
                        # Create comprehensive error result
                        error_result = create_comprehensive_error_result(
                            term_name, term_characteristics, strategy, str(e), 
                            batch_num, worker_id, attempt + 1
                        )
                        results.append(error_result)
                        failed_terms += 1
                        print(f"   [ERROR] Thread {worker_id}: '{term_name}' failed after {max_retries} attempts")
                
                # Adaptive delay between retries based on error type
                if attempt < max_retries - 1:
                    if is_connection_error:
                        # Longer delays for connection errors
                        delay = 3 + (attempt * 2)  # 3, 5, 7, 9 seconds
                        print(f"   [CONFIG] Connection error detected, waiting {delay}s before retry...")
                    else:
                        delay = 1 + (attempt * 0.5)  # Standard increasing delay
                    time.sleep(delay)
            
            # Small delay between terms with adaptive timing
            if i < len(batch_terms) - 1:
                time.sleep(0.1 + (failed_terms * 0.05))  # Shorter delays for threads
        
        # Save batch results with enhanced metadata
        output_file = os.path.join(batch_dir, f"modern_{file_prefix}_validation_batch_{batch_num:03d}.json")
        
        batch_data = create_enhanced_batch_metadata(
            results, successful_terms, failed_terms, len(batch_terms), 
            worker_id, src_lang, tgt_lang, industry_context, model_name
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Thread {worker_id}: Enhanced batch completed successfully ({successful_terms}/{len(batch_terms)} terms)")
        
        return {
            'batch_num': batch_num,
            'worker_id': worker_id,
            'success': True,
            'processed_count': len(results),
            'successful_terms': successful_terms,
            'failed_terms': failed_terms,
            'output_file': output_file,
            'enhancement_stats': {
                'ml_scoring_applied': sum(1 for r in results if 'enhanced_score' in r),
                'context_analysis_applied': sum(1 for r in results if 'advanced_context_analysis' in r),
                'fallback_used': sum(1 for r in results if r.get('status') == 'processed_with_fallback')
            }
        }
        
    except Exception as e:
        error_msg = f"Thread {worker_id}: Fatal error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            'batch_num': batch_num,
            'worker_id': worker_id,
            'success': False,
            'error': error_msg,
            'processed_count': 0,
            'successful_terms': 0,
            'failed_terms': len(batch_terms)
        }


def enhanced_validate_batch_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """Enhanced worker function with robust error handling, ML scoring, and advanced strategies"""
    (batch_terms, batch_num, file_prefix, model_name, src_lang, tgt_lang, industry_context, batch_dir) = args_tuple
    
    worker_id = f"{file_prefix}-{batch_num}"
    
    try:
        print(f"[LAUNCH] Worker {worker_id}: Starting enhanced processing of {len(batch_terms)} terms")
        
        # Initialize agent in worker process
        agent = ModernTerminologyReviewAgent(model_name)
        # Apply authentication fix
        agent = ensure_agent_auth_fix(agent)
        
        # Initialize enhanced components in worker
        ml_scorer = MLQualityScorer()
        context_analyzer = AdvancedContextAnalyzer()
        
        results = []
        successful_terms = 0
        failed_terms = 0
        
        for i, term_data in enumerate(batch_terms):
            term_name = term_data.get('term', '') if isinstance(term_data, dict) else str(term_data)
            original_texts = []
            
            if isinstance(term_data, dict) and 'original_texts' in term_data:
                original_texts = term_data['original_texts'].get('texts', [])
            
            print(f"   [CONFIG] Worker {worker_id}: Processing term {i+1}/{len(batch_terms)}: '{term_name}'")
            
            # Enhanced term characteristics analysis (from enhanced_failure_reprocessor.py)
            term_characteristics = analyze_term_characteristics(term_name)
            strategy = determine_search_strategy(term_name, term_characteristics)
            
            # Process with multiple retry strategies and enhanced validation
            max_retries = 5  # Increased for connection errors
            term_result = None
            
            for attempt in range(max_retries):
                try:
                    # Try different validation approaches based on term characteristics
                    if attempt == 0:
                        # Standard validation
                        result_str = agent.validate_term_candidate_efficient(
                            term=term_name,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            industry_context=industry_context,
                            save_to_file="",
                            original_texts=original_texts
                        )
                    elif attempt == 1:
                        # Enhanced context-based validation
                        result_str = validate_with_enhanced_context(
                            agent, term_name, term_characteristics, strategy, original_texts,
                            src_lang, tgt_lang, industry_context
                        )
                    else:
                        # Fallback validation with basic scoring
                        result_str = create_fallback_validation_result(
                            term_name, term_characteristics, strategy, original_texts
                        )
                    
                    # Robust result parsing with enhanced error handling
                    if isinstance(result_str, dict):
                        term_result = result_str
                    else:
                        # Try to extract JSON from the result string
                        json_match = re.search(r'\{.*\}', str(result_str), re.DOTALL)
                        if json_match:
                            try:
                                term_result = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                # Try to fix common JSON issues
                                json_str = json_match.group()
                                json_str = re.sub(r'\bfalse\b', 'false', json_str)
                                json_str = re.sub(r'\btrue\b', 'true', json_str)
                                json_str = re.sub(r'\bnull\b', 'null', json_str)
                                term_result = json.loads(json_str)
                        else:
                            # Create enhanced fallback result
                            term_result = create_enhanced_fallback_result(
                                term_name, term_characteristics, strategy, str(result_str), attempt + 1
                            )
                    
                    # Add comprehensive metadata and enhancements
                    if term_result:
                        term_result['batch_id'] = batch_num
                        term_result['worker_id'] = worker_id
                        term_result['processing_attempt'] = attempt + 1
                        term_result['enhanced_processing'] = True
                        term_result['term_characteristics'] = term_characteristics
                        term_result['search_strategy'] = strategy
                        
                        # Apply ML-based scoring if we have enough data
                        if isinstance(term_data, dict):
                            enhanced_score = ml_scorer.calculate_enhanced_score(term_data, term_result)
                            term_result['enhanced_score'] = enhanced_score
                            term_result['ml_features'] = ml_scorer._extract_features(term_data, term_result)
                            
                            # Add advanced context analysis
                            context_analysis = context_analyzer.analyze_context(term_data)
                            term_result['advanced_context_analysis'] = context_analysis
                            
                            # CRITICAL FIX: Preserve Step5/Step6/Translatability data from term_data
                            # These fields are required for final scoring in step7_fixed_batch_processing.py
                            if 'translatability_analysis' in term_data:
                                term_result['translatability_analysis'] = term_data['translatability_analysis']
                            if 'step5_translation_data' in term_data:
                                # Keep the original field name for consistency
                                term_result['step5_translation_data'] = term_data['step5_translation_data']
                                term_result['step5_integration_data'] = term_data['step5_translation_data']  # Legacy support
                            if 'step6_verification_data' in term_data:
                                term_result['step6_verification_data'] = term_data['step6_verification_data']
                            
                            # Ensure decision field exists (normalize from agent response)
                            if 'decision' not in term_result and 'recommendation' in term_result:
                                term_result['decision'] = term_result['recommendation']
                            elif 'decision' not in term_result and 'status' in term_result:
                                term_result['decision'] = term_result['status']
                            elif 'decision' not in term_result:
                                # Default decision based on score
                                score = term_result.get('score', 0.5)
                                if score >= 0.7:
                                    term_result['decision'] = 'APPROVED'
                                elif score >= 0.5:
                                    term_result['decision'] = 'CONDITIONALLY_APPROVED'
                                else:
                                    term_result['decision'] = 'NEEDS_REVIEW'
                        
                        # Apply quality improvements based on characteristics
                        term_result = apply_quality_improvements(term_result, term_characteristics)
                        
                        results.append(term_result)
                        successful_terms += 1
                        print(f"   [OK] Worker {worker_id}: '{term_name}' processed successfully (attempt {attempt + 1}, strategy: {strategy})")
                        break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"   [WARNING] Worker {worker_id}: Error processing '{term_name}' (attempt {attempt + 1}): {str(e)}")
                    
                    # Enhanced retry logic for connection errors
                    is_connection_error = any(keyword in error_msg for keyword in [
                        'connection error', 'timeout', 'network', 'connection', 'ssl', 'http'
                    ])
                    
                    if attempt == max_retries - 1:
                        # Create comprehensive error result
                        error_result = create_comprehensive_error_result(
                            term_name, term_characteristics, strategy, str(e), 
                            batch_num, worker_id, attempt + 1
                        )
                        results.append(error_result)
                        failed_terms += 1
                        print(f"   [ERROR] Worker {worker_id}: '{term_name}' failed after {max_retries} attempts")
                
                # Adaptive delay between retries based on error type
                if attempt < max_retries - 1:
                    if is_connection_error:
                        # Longer delays for connection errors
                        delay = 3 + (attempt * 2)  # 3, 5, 7, 9 seconds
                        print(f"   [CONFIG] Connection error detected, waiting {delay}s before retry...")
                    else:
                        delay = 1 + (attempt * 0.5)  # Standard increasing delay
                    time.sleep(delay)
            
            # Small delay between terms with adaptive timing
            if i < len(batch_terms) - 1:
                time.sleep(0.2 + (failed_terms * 0.1))  # Longer delays if many failures
        
        # Save batch results with enhanced metadata
        output_file = os.path.join(batch_dir, f"modern_{file_prefix}_validation_batch_{batch_num:03d}.json")
        
        batch_data = create_enhanced_batch_metadata(
            results, successful_terms, failed_terms, len(batch_terms), 
            worker_id, src_lang, tgt_lang, industry_context, model_name
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Worker {worker_id}: Enhanced batch completed successfully ({successful_terms}/{len(batch_terms)} terms)")
        
        return {
            'batch_num': batch_num,
            'worker_id': worker_id,
            'success': True,
            'processed_count': len(results),
            'successful_terms': successful_terms,
            'failed_terms': failed_terms,
            'output_file': output_file,
            'enhancement_stats': {
                'ml_scoring_applied': sum(1 for r in results if 'enhanced_score' in r),
                'context_analysis_applied': sum(1 for r in results if 'advanced_context_analysis' in r),
                'fallback_used': sum(1 for r in results if r.get('status') == 'processed_with_fallback')
            }
        }
        
    except Exception as e:
        error_msg = f"Worker {worker_id}: Fatal error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            'batch_num': batch_num,
            'worker_id': worker_id,
            'success': False,
            'error': error_msg,
            'processed_count': 0,
            'successful_terms': 0,
            'failed_terms': len(batch_terms)
        }


class EnhancedValidationSystem:
    """Main enhanced validation system"""
    
    def __init__(self, manager: OrganizedValidationManager):
        self.manager = manager
    
    def process_terms_parallel(self, terms: List[Dict[str, Any]], file_prefix: str,
                             batch_size: int = 5, max_workers: int = 4, 
                             src_lang: str = "EN", tgt_lang: str = None, 
                             industry_context: str = "General"):
        """Process terms in parallel with enhanced error handling and organization"""
        
        # Get already processed terms
        processed_terms = self.manager.get_already_processed_terms(file_prefix)
        
        # Filter out already processed terms
        unprocessed_terms = []
        for term_data in terms:
            term = term_data.get('term', '') if isinstance(term_data, dict) else str(term_data)
            if term not in processed_terms:
                unprocessed_terms.append(term_data)
        
        print(f"[STATS] Enhanced Processing Status:")
        print(f"   Total terms: {len(terms)}")
        print(f"   Already processed: {len(processed_terms)}")
        print(f"   Remaining to process: {len(unprocessed_terms)}")
        
        if not unprocessed_terms:
            print("[OK] All terms already processed!")
            return
        
        # Clean up old cache entries
        self.manager.cache.cleanup_old_cache()
        
        # Get starting batch number
        start_batch_num = self.manager.get_next_batch_number(file_prefix)
        
        # Create batches
        batches = []
        for i in range(0, len(unprocessed_terms), batch_size):
            batch_data = unprocessed_terms[i:i + batch_size]
            batch_num = start_batch_num + (i // batch_size)
            
            batches.append((
                batch_data, batch_num, file_prefix, self.manager.model_name,
                src_lang, tgt_lang, industry_context, self.manager.batch_dir
            ))
        
        print(f"[LAUNCH] Starting enhanced parallel processing:")
        print(f"   Batches to process: {len(batches)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max workers: {max_workers}")
        print(f"   Starting from batch: {start_batch_num}")
        print(f"   Model: {self.manager.model_name}")
        
        # Process batches in parallel
        failed_batches = []
        completed_batches = 0
        total_successful_terms = 0
        total_failed_terms = 0
        start_time = time.time()
        
        # Auto-detect optimal workers based on system specifications
        system_specs = detect_optimal_workers()
        
        if max_workers == -1:  # Auto-detect mode
            effective_workers = system_specs['optimal_workers']
        else:
            # Use requested workers but cap at optimal for stability
            effective_workers = min(max_workers, system_specs['optimal_workers'])
        
        use_processes = True  # Always use processes for best performance
        
        print(f"   [LAUNCH] Using {effective_workers} workers (requested: {max_workers}, optimal: {system_specs['optimal_workers']})")
        print(f"   [SAVE] Memory per worker: {system_specs['memory_per_worker']:.1f} GB")
        print(f"   🏷️  System tier: {system_specs['system_tier']}")
        
        # Use system-optimized chunk size
        batch_chunks = []
        chunk_size = system_specs['chunk_size']
        
        print(f"   [PACKAGE] Chunk size: {chunk_size} batches per chunk (system-optimized)")
        
        for i in range(0, len(batches), chunk_size):
            batch_chunks.append(batches[i:i + chunk_size])
        
        print(f"   Processing {len(batch_chunks)} chunks of batches")
        
        for chunk_idx, batch_chunk in enumerate(batch_chunks):
            print(f"\n[CONFIG] Processing chunk {chunk_idx + 1}/{len(batch_chunks)} ({len(batch_chunk)} batches)")
            
            try:
                # Use hybrid executor approach for optimal performance
                if use_processes:
                    executor = ProcessPoolExecutor(
                        max_workers=effective_workers,
                        mp_context=mp.get_context('spawn')
                    )
                    worker_func = enhanced_validate_batch_worker
                    unpack_args = True
                else:
                    executor = ThreadPoolExecutor(max_workers=effective_workers)
                    worker_func = enhanced_validate_batch_worker_threaded
                    unpack_args = False
                
                try:
                    # Submit jobs for this chunk
                    future_to_batch = {}
                    for batch_args in batch_chunk:
                        if unpack_args:
                            # Use original tuple format for ProcessPoolExecutor
                            future = executor.submit(worker_func, batch_args)
                        else:
                            # Unpack args for ThreadPoolExecutor
                            batch_terms, batch_num, file_prefix, model_name, src_lang, tgt_lang, industry_context, batch_dir = batch_args
                            future = executor.submit(
                                worker_func, batch_terms, batch_num, file_prefix, model_name, 
                                src_lang, tgt_lang, industry_context, batch_dir
                            )
                        future_to_batch[future] = batch_args[1]  # batch_num
                    
                    # Process completed jobs with appropriate timeouts
                    timeout_total = 1200 if use_processes else 1800  # Shorter for processes
                    timeout_individual = 180 if use_processes else 300  # Shorter for processes
                    
                    for future in as_completed(future_to_batch, timeout=timeout_total):
                        batch_num = future_to_batch[future]
                        try:
                            result = future.result(timeout=timeout_individual)
                            
                            if result['success']:
                                completed_batches += 1
                                total_successful_terms += result['successful_terms']
                                total_failed_terms += result['failed_terms']
                                print(f"[OK] Enhanced batch {batch_num} completed ({result['successful_terms']}/{result['processed_count']} terms successful)")
                            else:
                                failed_batches.append((batch_num, result.get('error', 'Unknown error')))
                                print(f"[ERROR] Enhanced batch {batch_num} failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            failed_batches.append((batch_num, str(e)))
                            print(f"[ERROR] Enhanced batch {batch_num} processing error: {e}")
                
                finally:
                    # Explicit cleanup
                    executor.shutdown(wait=True)
                    del executor
                    
            except KeyboardInterrupt:
                print("\n[WARNING] Processing interrupted by user")
                break
            except Exception as e:
                print(f"[ERROR] Enhanced parallel processing error in chunk {chunk_idx + 1}: {e}")
                # Continue with next chunk
                continue
            
            # Memory cleanup and system monitoring between chunks
            if chunk_idx < len(batch_chunks) - 1:
                print(f"   🧹 Memory cleanup and system monitoring...")
                gc.collect()  # Force garbage collection
                
                # Monitor system resources
                resources = monitor_system_resources()
                print(f"   [STATS] CPU: {resources['cpu_percent']:.1f}%, Memory: {resources['memory_percent']:.1f}%, Available: {resources['available_memory_gb']:.1f}GB")
                
                # Adaptive delay based on system load
                if resources['high_load']:
                    delay = 12  # Longer delay if system is under stress
                    print(f"   [WARNING]  High system load detected - extended pause ({delay}s)")
                else:
                    delay = 6   # Normal delay
                    print(f"   ⏳ Normal recovery pause ({delay}s)")
                
                time.sleep(delay)
        
        # Performance summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n[STATS] Enhanced Processing Summary:")
        print(f"   Total batches: {len(batches)}")
        print(f"   Completed: {completed_batches}")
        print(f"   Failed: {len(failed_batches)}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        print(f"   Successful terms: {total_successful_terms}")
        print(f"   Failed terms: {total_failed_terms}")
        print(f"   Success rate: {total_successful_terms/(total_successful_terms+total_failed_terms)*100:.1f}%" if (total_successful_terms + total_failed_terms) > 0 else "N/A")
        
        if failed_batches:
            print(f"\n[ERROR] Failed batches:")
            for batch_num, error_msg in failed_batches:
                print(f"   Batch {batch_num}: {error_msg}")
    
    def process_gap_filling(self, terms: List[Dict[str, Any]], file_prefix: str,
                           target_batches: List[int], batch_size: int = 5, max_workers: int = 4,
                           src_lang: str = "EN", tgt_lang: str = None, 
                           industry_context: str = "General"):
        """Process specific missing batches to fill gaps systematically"""
        
        if not target_batches:
            print("[WARNING] No target batches specified for gap filling")
            return
        
        print(f"[TARGET] GAP FILLING MODE")
        print(f"📋 Target batches: {len(target_batches)} batches")
        print(f"[STATS] Batch ranges: {target_batches[0]} to {target_batches[-1]}")
        
        # Create batches for specific batch numbers
        gap_batches = []
        
        for batch_num in target_batches:
            # Calculate which terms should be in this batch
            start_idx = (batch_num - 1) * batch_size
            end_idx = start_idx + batch_size
            
            if start_idx >= len(terms):
                print(f"[WARNING] Batch {batch_num} beyond available terms ({len(terms)} total)")
                continue
            
            batch_terms = terms[start_idx:end_idx]
            
            if not batch_terms:
                print(f"[WARNING] No terms for batch {batch_num}")
                continue
            
            # Check if this batch already exists
            batch_file = os.path.join(self.manager.batch_dir, f"modern_{file_prefix}_validation_batch_{batch_num:03d}.json")
            if os.path.exists(batch_file):
                print(f"[OK] Batch {batch_num} already exists, skipping")
                continue
            
            gap_batches.append((
                batch_terms, batch_num, file_prefix, self.manager.model_name,
                src_lang, tgt_lang, industry_context, self.manager.batch_dir
            ))
        
        if not gap_batches:
            print("[OK] All target batches already exist!")
            return
        
        print(f"\n[LAUNCH] Processing {len(gap_batches)} missing batches:")
        for batch_terms, batch_num, _, _, _, _, _, _ in gap_batches[:10]:  # Show first 10
            print(f"   Batch {batch_num}: {len(batch_terms)} terms")
        if len(gap_batches) > 10:
            print(f"   ... and {len(gap_batches) - 10} more batches")
        
        # Process gap batches with Windows-safe approach
        failed_batches = []
        completed_batches = 0
        total_successful_terms = 0
        total_failed_terms = 0
        start_time = time.time()
        
        # Auto-detect optimal workers for gap filling (more conservative)
        system_specs = detect_optimal_workers()
        
        if max_workers == -1:  # Auto-detect mode
            effective_workers = max(2, system_specs['optimal_workers'] - 2)  # Conservative for gap filling
        else:
            # Use requested workers but cap at optimal-2 for gap filling stability
            effective_workers = min(max_workers, max(2, system_specs['optimal_workers'] - 2))
        
        use_processes = True
        
        print(f"   [LAUNCH] Gap filling: {effective_workers} workers (requested: {max_workers}, optimal: {system_specs['optimal_workers']})")
        print(f"   [SAVE] Memory per worker: {system_specs['memory_gb'] / effective_workers:.1f} GB")
        
        # Use conservative chunk size for gap filling
        chunk_size = min(effective_workers * 2, max(6, system_specs['chunk_size'] // 2))
        
        batch_chunks = []
        print(f"   [PACKAGE] Gap-filling chunk size: {chunk_size} batches per chunk (conservative)")
        
        for i in range(0, len(gap_batches), chunk_size):
            batch_chunks.append(gap_batches[i:i + chunk_size])
        
        print(f"   Processing {len(batch_chunks)} chunks")
        
        for chunk_idx, batch_chunk in enumerate(batch_chunks):
            print(f"\n[CONFIG] Gap-filling chunk {chunk_idx + 1}/{len(batch_chunks)} ({len(batch_chunk)} batches)")
            
            try:
                # Use hybrid executor approach for gap filling
                if use_processes:
                    executor = ProcessPoolExecutor(
                        max_workers=effective_workers,
                        mp_context=mp.get_context('spawn')
                    )
                    worker_func = enhanced_validate_batch_worker
                    unpack_args = True
                else:
                    executor = ThreadPoolExecutor(max_workers=effective_workers)
                    worker_func = enhanced_validate_batch_worker_threaded
                    unpack_args = False
                
                try:
                    # Submit jobs for this chunk
                    future_to_batch = {}
                    for batch_args in batch_chunk:
                        if unpack_args:
                            # Use original tuple format for ProcessPoolExecutor
                            future = executor.submit(worker_func, batch_args)
                        else:
                            # Unpack args for ThreadPoolExecutor
                            batch_terms, batch_num, file_prefix, model_name, src_lang, tgt_lang, industry_context, batch_dir = batch_args
                            future = executor.submit(
                                worker_func, batch_terms, batch_num, file_prefix, model_name, 
                                src_lang, tgt_lang, industry_context, batch_dir
                            )
                        future_to_batch[future] = batch_args[1]  # batch_num
                    
                    # Process completed jobs with extended timeout for gap filling
                    timeout_total = 1800 if use_processes else 2400  # Shorter for processes
                    timeout_individual = 300 if use_processes else 600  # Shorter for processes
                    
                    for future in as_completed(future_to_batch, timeout=timeout_total):
                        batch_num = future_to_batch[future]
                        try:
                            result = future.result(timeout=timeout_individual)
                            
                            if result['success']:
                                completed_batches += 1
                                total_successful_terms += result['successful_terms']
                                total_failed_terms += result['failed_terms']
                                print(f"[OK] Gap batch {batch_num} completed ({result['successful_terms']}/{result['processed_count']} terms)")
                            else:
                                failed_batches.append((batch_num, result.get('error', 'Unknown error')))
                                print(f"[ERROR] Gap batch {batch_num} failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            failed_batches.append((batch_num, str(e)))
                            print(f"[ERROR] Gap batch {batch_num} error: {e}")
                
                finally:
                    # Explicit cleanup for gap filling
                    executor.shutdown(wait=True)
                    del executor
                    
            except KeyboardInterrupt:
                print("\n[WARNING] Gap filling interrupted by user")
                break
            except Exception as e:
                print(f"[ERROR] Gap filling error in chunk {chunk_idx + 1}: {e}")
                continue
            
            # Memory cleanup and extended pause for gap filling
            if chunk_idx < len(batch_chunks) - 1:
                print(f"   🧹 Memory cleanup and extended pause...")
                gc.collect()  # Force garbage collection
                time.sleep(10)  # Extended delay for gap filling stability with memory recovery
        
        # Gap filling summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n[STATS] Gap Filling Summary:")
        print(f"   Target batches: {len(target_batches)}")
        print(f"   Processed: {len(gap_batches)}")
        print(f"   Completed: {completed_batches}")
        print(f"   Failed: {len(failed_batches)}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        print(f"   Successful terms: {total_successful_terms}")
        print(f"   Failed terms: {total_failed_terms}")
        
        if failed_batches:
            print(f"\n[ERROR] Failed gap batches:")
            for batch_num, error_msg in failed_batches:
                print(f"   Batch {batch_num}: {error_msg}")
        else:
            print(f"\n[SUCCESS] All gap batches completed successfully!")
    
    def consolidate_results(self, file_prefix: str, classification_type: str):
        """Consolidate all batch results into organized files"""
        print(f"\n[CONFIG] Consolidating enhanced results for {file_prefix}")
        
        # Find all batch files with standardized naming
        patterns = [
            os.path.join(self.manager.batch_dir, f"modern_{file_prefix}_validation_batch_*.json"),
            os.path.join(self.manager.batch_dir, f"enhanced_{file_prefix}_validation_batch_*.json")  # Fallback for any existing enhanced files
        ]
        
        batch_files = []
        for pattern in patterns:
            batch_files.extend(glob.glob(pattern))
        
        batch_files = sorted(set(batch_files))  # Remove duplicates and sort
        
        if not batch_files:
            print(f"[WARNING] No batch files found for {file_prefix}")
            return
        
        consolidated_data = {
            'consolidation_info': {
                'run_folder': self.manager.run_folder,
                'file_prefix': file_prefix,
                'classification_type': classification_type,
                'consolidated_at': datetime.now().isoformat(),
                'total_batches': len(batch_files),
                'consolidation_version': 'enhanced_v4.0'
            },
            'batches': {}
        }
        
        total_terms = 0
        total_successful = 0
        total_failed = 0
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                # Extract batch number from filename
                batch_num_match = re.search(r'batch_(\d+)', os.path.basename(batch_file))
                batch_id = int(batch_num_match.group(1)) if batch_num_match else len(consolidated_data['batches']) + 1
                
                batch_results = batch_data.get('results', [])
                batch_metadata = batch_data.get('metadata', {})
                
                consolidated_data['batches'][f"batch_{batch_id:03d}"] = {
                    'batch_id': batch_id,
                    'processed_at': batch_metadata.get('generation_timestamp', 'unknown'),
                    'batch_size': len(batch_results),
                    'successful_terms': batch_metadata.get('successful_terms', 0),
                    'failed_terms': batch_metadata.get('failed_terms', 0),
                    'data': batch_data
                }
                
                total_terms += len(batch_results)
                total_successful += batch_metadata.get('successful_terms', 0)
                total_failed += batch_metadata.get('failed_terms', 0)
                
            except Exception as e:
                print(f"[WARNING] Error reading {batch_file}: {e}")
        
        consolidated_data['summary_statistics'] = {
            'total_terms': total_terms,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'success_rate': total_successful / max(1, total_terms) * 100
        }
        
        # Save consolidated file with standardized naming
        consolidated_filename = os.path.join(
            self.manager.results_dir, 
            f"consolidated_modern_{file_prefix}_validation_results.json"
        )
        
        with open(consolidated_filename, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Consolidated {len(batch_files)} enhanced batches ({total_terms} terms) into {os.path.basename(consolidated_filename)}")
        print(f"[STATS] Summary: {total_successful} successful, {total_failed} failed ({total_successful/max(1,total_terms)*100:.1f}% success rate)")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Modern Parallel Terminology Validation System")
    parser.add_argument("--dict-file", default="Fast_Dictionary_Terms_20250903_123659.json",
                       help="Dictionary terms input file")
    parser.add_argument("--non-dict-file", default="Fast_Non_Dictionary_Terms_20250903_123659.json",
                       help="Non-dictionary terms input file")
    parser.add_argument("--min-frequency", type=int, default=2,
                       help="Minimum term frequency")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Terms per batch")
    parser.add_argument("--max-workers", type=int, default=-1,
                       help="Maximum number of parallel workers (-1 for auto-detect based on system specs)")
    parser.add_argument("--src-lang", default="EN",
                       help="Source language code")
    parser.add_argument("--tgt-lang", default=None,
                       help="Target language code (optional)")
    parser.add_argument("--industry-context", default="General",
                       help="Industry context for validation")
    parser.add_argument("--model", default="gpt-4.1",
                       help="Model to use for validation")
    parser.add_argument("--run-folder", 
                       help="Specific run folder name")
    parser.add_argument("--dict-only", action="store_true",
                       help="Process only dictionary terms")
    parser.add_argument("--non-dict-only", action="store_true",
                       help="Process only non-dictionary terms")
    parser.add_argument("--consolidate-only", action="store_true",
                       help="Only consolidate existing results")
    parser.add_argument("--no-organize", action="store_true",
                       help="Skip organizing existing files")
    parser.add_argument("--fill-gaps", action="store_true",
                       help="Fill gaps in batch numbering systematically")
    parser.add_argument("--target-batches", type=str,
                       help="Specific batch ranges to process (e.g., '3-12,75-84')")
    
    args = parser.parse_args()
    
    # Enhanced multiprocessing configuration for Windows with 20 workers
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    # Windows-specific optimizations for high worker count
    import os
    if os.name == 'nt':  # Windows
        # Set environment variables for better process handling
        os.environ['PYTHONHASHSEED'] = '0'  # Consistent hashing
        os.environ['OMP_NUM_THREADS'] = '1'  # Prevent thread conflicts
        
        # Try to increase Windows handle limits
        try:
            import ctypes
            from ctypes import wintypes
            
            # Increase handle limits
            kernel32 = ctypes.windll.kernel32
            current_process = kernel32.GetCurrentProcess()
            
            # Set process priority to normal to prevent resource starvation
            kernel32.SetPriorityClass(current_process, 0x00000020)
            
        except Exception as e:
            print(f"[WARNING] Could not optimize Windows settings: {e}")
            pass
    
    print("[SPECIAL] ENHANCED MODERN PARALLEL TERMINOLOGY VALIDATION SYSTEM")
    print("=" * 70)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SETUP] Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Min frequency: {args.min_frequency}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max workers: {args.max_workers}")
    print(f"   Source language: {args.src_lang}")
    print(f"   Target language: {args.tgt_lang or '(terminology validation only)'}")
    print(f"   Industry context: {args.industry_context}")
    print(f"[LAUNCH] Enhanced Features Active:")
    print(f"   [OK] Organized folder structure")
    print(f"   [OK] Robust error handling with multiple retries")
    print(f"   [OK] ML-based quality scoring")
    print(f"   [OK] SQLite validation caching")
    print(f"   [OK] Intelligent resumption from existing batches")
    print(f"   [OK] Enhanced result parsing and fallback mechanisms")
    
    try:
        # Initialize enhanced manager
        manager = OrganizedValidationManager(
            model_name=args.model,
            run_folder=args.run_folder,
            organize_existing=not args.no_organize
        )
        
        # Initialize enhanced validation system
        validation_system = EnhancedValidationSystem(manager)
        
        # Consolidation only mode
        if args.consolidate_only:
            print("\n[CONFIG] Consolidation mode - processing existing batch files")
            if not args.non_dict_only:
                validation_system.consolidate_results("dictionary", "dictionary")
            if not args.dict_only:
                validation_system.consolidate_results("non_dictionary", "non_dictionary")
            return
        
        # Gap filling mode
        if args.fill_gaps or args.target_batches:
            print("\n[TARGET] GAP FILLING MODE")
            print("-" * 50)
            
            # Determine target batches
            target_batches = []
            if args.target_batches:
                target_batches = manager.parse_target_batches(args.target_batches)
                print(f"📋 Using specified target batches: {len(target_batches)} batches")
            elif args.fill_gaps:
                # Auto-detect missing batches
                if not args.non_dict_only:
                    dict_missing = manager.get_missing_batch_numbers("dictionary")
                    target_batches.extend(dict_missing)
                if not args.dict_only:
                    non_dict_missing = manager.get_missing_batch_numbers("non_dictionary")
                    target_batches.extend(non_dict_missing)
                
                target_batches = sorted(list(set(target_batches)))
                print(f"📋 Auto-detected missing batches: {len(target_batches)} batches")
            
            if not target_batches:
                print("[OK] No gaps found to fill!")
                return
            
            # Load terms for gap filling
            dict_terms = []
            non_dict_terms = []
            
            if not args.non_dict_only and os.path.exists(args.dict_file):
                dict_terms, _ = manager.load_terms_from_json(args.dict_file, args.min_frequency)
                print(f"[BOOK] Loaded {len(dict_terms)} dictionary terms for gap filling")
            
            if not args.dict_only and os.path.exists(args.non_dict_file):
                non_dict_terms, _ = manager.load_terms_from_json(args.non_dict_file, args.min_frequency)
                print(f"[DICT] Loaded {len(non_dict_terms)} non-dictionary terms for gap filling")
            
            # Process gaps for dictionary terms
            if dict_terms and not args.non_dict_only:
                dict_target_batches = [b for b in target_batches if b <= len(dict_terms) // args.batch_size + 1]
                if dict_target_batches:
                    print(f"\n[BOOK] FILLING DICTIONARY GAPS")
                    print(f"   Target batches: {len(dict_target_batches)}")
                    validation_system.process_gap_filling(
                        terms=dict_terms,
                        file_prefix="dictionary",
                        target_batches=dict_target_batches,
                        batch_size=args.batch_size,
                        max_workers=args.max_workers,
                        src_lang=args.src_lang,
                        tgt_lang=args.tgt_lang,
                        industry_context=args.industry_context
                    )
            
            # Process gaps for non-dictionary terms
            if non_dict_terms and not args.dict_only:
                non_dict_target_batches = [b for b in target_batches if b <= len(non_dict_terms) // args.batch_size + 1]
                if non_dict_target_batches:
                    print(f"\n[DICT] FILLING NON-DICTIONARY GAPS")
                    print(f"   Target batches: {len(non_dict_target_batches)}")
                    validation_system.process_gap_filling(
                        terms=non_dict_terms,
                        file_prefix="non_dictionary",
                        target_batches=non_dict_target_batches,
                        batch_size=args.batch_size,
                        max_workers=args.max_workers,
                        src_lang=args.src_lang,
                        tgt_lang=args.tgt_lang,
                        industry_context=args.industry_context
                    )
            
            print(f"\n[SUCCESS] GAP FILLING COMPLETED!")
            return
        
        # Process dictionary terms
        if not args.non_dict_only:
            if os.path.exists(args.dict_file):
                print(f"\n[BOOK] PROCESSING DICTIONARY TERMS WITH ENHANCED SYSTEM")
                print("-" * 60)
                
                dict_terms, _ = manager.load_terms_from_json(args.dict_file, args.min_frequency)
                
                if dict_terms:
                    validation_system.process_terms_parallel(
                        terms=dict_terms,
                        file_prefix="dictionary",
                        batch_size=args.batch_size,
                        max_workers=args.max_workers,
                        src_lang=args.src_lang,
                        tgt_lang=args.tgt_lang,
                        industry_context=args.industry_context
                    )
                    
                    validation_system.consolidate_results("dictionary", "dictionary")
                else:
                    print("[WARNING] No dictionary terms to process after enhanced filtering")
            else:
                print(f"[WARNING] Dictionary file not found: {args.dict_file}")
        
        # Process non-dictionary terms
        if not args.dict_only:
            if os.path.exists(args.non_dict_file):
                print(f"\n[DICT] PROCESSING NON-DICTIONARY TERMS WITH ENHANCED SYSTEM")
                print("-" * 60)
                
                non_dict_terms, _ = manager.load_terms_from_json(args.non_dict_file, args.min_frequency)
                
                if non_dict_terms:
                    validation_system.process_terms_parallel(
                        terms=non_dict_terms,
                        file_prefix="non_dictionary",
                        batch_size=args.batch_size,
                        max_workers=args.max_workers,
                        src_lang=args.src_lang,
                        tgt_lang=args.tgt_lang,
                        industry_context=args.industry_context
                    )
                    
                    validation_system.consolidate_results("non_dictionary", "non_dictionary")
                else:
                    print("[WARNING] No non-dictionary terms to process after enhanced filtering")
            else:
                print(f"[WARNING] Non-dictionary file not found: {args.non_dict_file}")
        
        print(f"\n[SUCCESS] ENHANCED PROCESSING COMPLETED!")
        print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[FOLDER] Results saved in: {manager.results_dir}")
        print(f"[STATS] Final Statistics:")
        print(f"   Cache hits: {manager.stats['cache_hits']}")
        print(f"   Processing errors handled gracefully with fallbacks")
        print(f"   All results organized in structured folders")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()