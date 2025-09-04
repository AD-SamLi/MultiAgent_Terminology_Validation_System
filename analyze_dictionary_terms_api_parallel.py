#!/usr/bin/env python3
"""
API-FOCUSED PARALLEL dictionary analysis with resume capability
Optimized specifically for parallel API processing:
1. Concurrent API calls with configurable thread pools
2. Smart API batching and rate limiting
3. Resume from checkpoint with API call tracking
4. Optimized for PyMultiDictionary parallel requests
5. Separate fast filtering from API processing
"""

import json
import time
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import queue
import threading

try:
    from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO, DICT_MW
    PYMULTI_AVAILABLE = True
except ImportError:
    print("❌ PyMultiDictionary not installed. Install with: pip install PyMultiDictionary")
    PYMULTI_AVAILABLE = False

# Global thread-safe counters
api_stats_lock = Lock()
api_stats = {
    "total_api_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "dict_words_found": 0,
    "non_dict_words": 0,
    "processing_time": 0
}

# Rate limiting semaphore
rate_limiter = None

# Common English words for fast pre-filtering
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 
    'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 
    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 
    'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 
    'most', 'us', 'is', 'water', 'long', 'find', 'here', 'thing', 'great', 'man', 'world', 'life', 'still', 'public', 'human',
    'read', 'left', 'put', 'end', 'why', 'called', 'should', 'never', 'did', 'different', 'number', 'part', 'turned', 'right',
    'three', 'small', 'large', 'next', 'early', 'during', 'press', 'close', 'night', 'real', 'almost', 'let', 'open', 'got',
    'together', 'already', 'lot', 'those', 'both', 'paper', 'important', 'children', 'side', 'feet', 'car', 'mile',
    'walk', 'white', 'sea', 'began', 'grow', 'took', 'river', 'four', 'carry', 'state', 'once', 'book', 'hear', 'stop',
    'without', 'second', 'later', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far', 'really', 'above', 'girl', 
    'sometimes', 'mountain', 'cut', 'young', 'talk', 'soon', 'list', 'song', 'leave', 'family', 'body', 'music', 'color', 
    'stand', 'sun', 'questions', 'fish', 'area', 'mark', 'dog', 'horse', 'birds', 'problem', 'complete', 'room', 'knew',
    'since', 'ever', 'piece', 'told', 'usually', 'friends', 'easy', 'heard', 'order', 'red', 'door', 'sure', 'become',
    'top', 'ship', 'across', 'today', 'short', 'better', 'best', 'however', 'low', 'hours', 'black', 'products', 'happened',
    'whole', 'measure', 'remember', 'waves', 'reached', 'listen', 'wind', 'rock', 'space', 'covered', 'fast', 'several',
    'hold', 'himself', 'toward', 'five', 'step', 'morning', 'passed', 'vowel', 'true', 'hundred', 'against', 'pattern'
}

def is_obviously_non_dictionary(word: str) -> Tuple[bool, str]:
    """Fast check for obviously non-dictionary terms"""
    word_lower = word.lower().strip()
    
    if not word_lower or len(word_lower) < 2:
        return True, "too_short"
    
    if len(word_lower) > 25:
        return True, "too_long"
    
    # Contains numbers
    if any(char.isdigit() for char in word_lower):
        return True, "contains_numbers"
    
    # Contains special characters
    if any(char in "()[]{}@#$%^&*+=|\\:;\"<>,.?/~`" for char in word_lower):
        return True, "special_chars"
    
    # Multiple uppercase letters (acronyms)
    if sum(1 for c in word if c.isupper()) >= 2:
        return True, "likely_acronym"
    
    # CamelCase
    if any(word[i].islower() and word[i+1].isupper() for i in range(len(word)-1)):
        return True, "camelcase"
    
    # Contains underscores or hyphens
    if '_' in word or '-' in word:
        return True, "technical_naming"
    
    # Common file extensions
    if word_lower in ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'jpg', 'png', 'exe', 'dll', 'zip', 'rar']:
        return True, "file_extension"
    
    # Technical suffixes
    tech_suffixes = ['cad', 'cam', 'bim', 'api', 'xml', 'html', 'css', 'exe', 'dll', 'cfg', 'sys', 'app', 'ware', 'tech']
    if any(word_lower.endswith(suffix) for suffix in tech_suffixes):
        return True, "technical_suffix"
    
    return False, "needs_api_check"

def is_obviously_dictionary(word: str) -> Tuple[bool, str]:
    """Fast check for obviously dictionary terms"""
    word_lower = word.lower().strip()
    
    # Common words
    if word_lower in COMMON_ENGLISH_WORDS:
        return True, "common_word"
    
    # Simple plurals
    if word_lower.endswith('s') and len(word_lower) > 3:
        singular = word_lower[:-1]
        if singular in COMMON_ENGLISH_WORDS:
            return True, "plural_form"
    
    # Simple past tense
    if word_lower.endswith('ed') and len(word_lower) > 4:
        base = word_lower[:-2]
        if base in COMMON_ENGLISH_WORDS:
            return True, "past_tense"
    
    # Present participle
    if word_lower.endswith('ing') and len(word_lower) > 5:
        base = word_lower[:-3]
        if base in COMMON_ENGLISH_WORDS:
            return True, "ing_form"
    
    # Adverbs
    if word_lower.endswith('ly') and len(word_lower) > 4:
        base = word_lower[:-2]
        if base in COMMON_ENGLISH_WORDS:
            return True, "adverb_form"
    
    return False, "needs_api_check"

class APIWorker:
    """Thread-safe API worker for dictionary lookups"""
    
    def __init__(self, worker_id: int, delay_between_calls: float = 0.1):
        self.worker_id = worker_id
        self.delay = delay_between_calls
        self.dictionary = MultiDictionary()
        self.local_stats = {
            "calls_made": 0,
            "successful": 0,
            "failed": 0,
            "dict_found": 0
        }
    
    def check_word(self, word: str) -> Dict:
        """Check a single word using API with rate limiting"""
        global rate_limiter, api_stats, api_stats_lock
        
        # Rate limiting
        if rate_limiter:
            rate_limiter.acquire()
        
        start_time = time.time()
        
        result = {
            "word": word,
            "in_dictionary": False,
            "method": "api_parallel",
            "worker_id": self.worker_id,
            "meaning_found": False,
            "synonyms_found": False,
            "api_source": None
        }
        
        try:
            word_lower = word.lower().strip()
            
            # Try Merriam-Webster first (fastest, most reliable)
            try:
                mw_meaning = self.dictionary.meaning('en', word_lower, dictionary=DICT_MW)
                if mw_meaning and str(mw_meaning).strip() and str(mw_meaning) != "None":
                    result["in_dictionary"] = True
                    result["meaning_found"] = True
                    result["api_source"] = "merriam_webster"
                    result["confidence"] = "high"
                    self.local_stats["dict_found"] += 1
                    self.local_stats["successful"] += 1
                    return result
            except Exception as e:
                result["mw_error"] = str(e)[:50]
            
            # Try synonyms (often faster than full meaning)
            try:
                synonyms = self.dictionary.synonym('en', word_lower)
                if synonyms and isinstance(synonyms, list) and len(synonyms) > 0:
                    result["in_dictionary"] = True
                    result["synonyms_found"] = True
                    result["api_source"] = "synonym_check"
                    result["confidence"] = "medium"
                    self.local_stats["dict_found"] += 1
                    self.local_stats["successful"] += 1
                    return result
            except Exception as e:
                result["synonym_error"] = str(e)[:50]
            
            # Final attempt with Educalingo
            try:
                edu_meaning = self.dictionary.meaning('en', word_lower, dictionary=DICT_EDUCALINGO)
                if edu_meaning and str(edu_meaning).strip() and str(edu_meaning) != "None":
                    result["in_dictionary"] = True
                    result["meaning_found"] = True
                    result["api_source"] = "educalingo"
                    result["confidence"] = "medium"
                    self.local_stats["dict_found"] += 1
                    self.local_stats["successful"] += 1
                    return result
            except Exception as e:
                result["educalingo_error"] = str(e)[:50]
            
            # Not found in any dictionary
            result["reason"] = "not_found_in_dictionaries"
            result["confidence"] = "high"
            self.local_stats["successful"] += 1
            
        except Exception as e:
            result["reason"] = f"api_error: {str(e)[:50]}"
            result["confidence"] = "error"
            self.local_stats["failed"] += 1
        
        finally:
            # Update global stats
            self.local_stats["calls_made"] += 1
            
            with api_stats_lock:
                api_stats["total_api_calls"] += 1
                api_stats["processing_time"] += time.time() - start_time
                if result.get("in_dictionary", False):
                    api_stats["dict_words_found"] += 1
                else:
                    api_stats["non_dict_words"] += 1
                
                if result.get("confidence") != "error":
                    api_stats["successful_calls"] += 1
                else:
                    api_stats["failed_calls"] += 1
            
            # Rate limiting delay
            if self.delay > 0:
                time.sleep(self.delay)
        
        return result

def process_api_batch_optimized(terms_needing_api: List[Dict], config: Dict) -> Tuple[List, List]:
    """Optimized parallel API processing with smart batching"""
    global rate_limiter
    
    if not PYMULTI_AVAILABLE:
        print("❌ PyMultiDictionary not available - defaulting all uncertain terms to non-dictionary")
        for term in terms_needing_api:
            term["dictionary_analysis"] = {
                "word": term.get('term', ''),
                "in_dictionary": False,
                "reason": "no_api_available",
                "method": "default",
                "confidence": "assumed"
            }
        return [], terms_needing_api
    
    max_workers = config.get('api_workers', 8)
    delay_between_calls = config.get('api_delay', 0.05)  # 50ms delay
    max_concurrent_requests = config.get('max_concurrent', 20)
    
    print(f"🌐 Starting parallel API processing:")
    print(f"   • Terms to check: {len(terms_needing_api):,}")
    print(f"   • API workers: {max_workers}")
    print(f"   • Delay between calls: {delay_between_calls}s")
    print(f"   • Max concurrent requests: {max_concurrent_requests}")
    
    # Rate limiter for concurrent requests
    rate_limiter = Semaphore(max_concurrent_requests)
    
    dictionary_words = []
    non_dictionary_words = []
    
    # Sort by frequency (check high-frequency terms first)
    terms_sorted = sorted(terms_needing_api, key=lambda x: x.get('frequency', 0), reverse=True)
    
    start_time = time.time()
    
    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create workers
        workers = [APIWorker(i, delay_between_calls) for i in range(max_workers)]
        
        # Submit all API calls
        future_to_term = {}
        worker_idx = 0
        
        for term_data in terms_sorted:
            term = term_data.get('term', '')
            worker = workers[worker_idx % len(workers)]
            
            future = executor.submit(worker.check_word, term)
            future_to_term[future] = (term_data, worker)
            worker_idx += 1
        
        print(f"   📤 Submitted {len(future_to_term)} API requests")
        
        # Collect results with progress tracking
        completed = 0
        progress_interval = max(1, len(future_to_term) // 20)  # Update every 5%
        
        for future in as_completed(future_to_term):
            term_data, worker = future_to_term[future]
            completed += 1
            
            try:
                api_result = future.result(timeout=30)  # 30 second timeout
                term_data["dictionary_analysis"] = api_result
                
                # Categorize result
                if api_result.get("in_dictionary", False):
                    dictionary_words.append(term_data)
                else:
                    non_dictionary_words.append(term_data)
                
            except Exception as e:
                # Handle timeout or other errors
                term_data["dictionary_analysis"] = {
                    "word": term_data.get('term', ''),
                    "in_dictionary": False,
                    "reason": f"api_timeout_or_error: {str(e)[:30]}",
                    "method": "api_parallel",
                    "confidence": "error",
                    "worker_id": worker.worker_id
                }
                non_dictionary_words.append(term_data)
            
            # Progress update
            if completed % progress_interval == 0 or completed == len(future_to_term):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(future_to_term) - completed
                eta = remaining / rate if rate > 0 else 0
                
                print(f"   📊 Progress: {completed:,}/{len(future_to_term):,} ({completed/len(future_to_term)*100:.1f}%) "
                      f"| Rate: {rate:.1f}/sec | ETA: {eta:.0f}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print worker statistics
    print(f"\n✅ Parallel API processing complete!")
    print(f"   ⏱️  Total time: {total_time:.1f}s")
    print(f"   📈 Average rate: {len(terms_needing_api)/total_time:.1f} terms/sec")
    print(f"   📖 Dictionary words found: {len(dictionary_words):,}")
    print(f"   🔧 Non-dictionary words: {len(non_dictionary_words):,}")
    
    # Worker performance stats
    print(f"\n👥 Worker Performance:")
    for i, worker in enumerate(workers):
        if worker.local_stats["calls_made"] > 0:
            success_rate = worker.local_stats["successful"] / worker.local_stats["calls_made"] * 100
            print(f"   Worker {i}: {worker.local_stats['calls_made']} calls, "
                  f"{success_rate:.1f}% success, {worker.local_stats['dict_found']} dict words")
    
    return dictionary_words, non_dictionary_words

def fast_filter_terms(terms: List[Dict]) -> Tuple[List, List, List]:
    """Fast pre-filtering before API calls"""
    print(f"🔍 Fast pre-filtering {len(terms):,} terms...")
    
    definite_dictionary = []
    definite_non_dictionary = []
    needs_api_check = []
    
    for term_data in terms:
        term = term_data.get('term', '')
        
        # Check if obviously dictionary word
        is_dict, dict_reason = is_obviously_dictionary(term)
        if is_dict:
            term_data["dictionary_analysis"] = {
                "word": term,
                "in_dictionary": True,
                "reason": dict_reason,
                "method": "fast_filter",
                "confidence": "high"
            }
            definite_dictionary.append(term_data)
            continue
        
        # Check if obviously non-dictionary
        is_non_dict, non_dict_reason = is_obviously_non_dictionary(term)
        if is_non_dict:
            term_data["dictionary_analysis"] = {
                "word": term,
                "in_dictionary": False,
                "reason": non_dict_reason,
                "method": "fast_filter",
                "confidence": "high"
            }
            definite_non_dictionary.append(term_data)
            continue
        
        # Needs API check
        needs_api_check.append(term_data)
    
    print(f"✅ Fast filtering complete:")
    print(f"   📖 Definite dictionary words: {len(definite_dictionary):,}")
    print(f"   🔧 Definite non-dictionary: {len(definite_non_dictionary):,}")
    print(f"   ❓ Need API check: {len(needs_api_check):,}")
    
    return definite_dictionary, definite_non_dictionary, needs_api_check

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint data"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
    return None

def save_checkpoint(checkpoint_file: str, data: Dict):
    """Save checkpoint data"""
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")

def analyze_terms_api_parallel(terms: List[Dict], checkpoint_file: str, config: Dict) -> Tuple[List, List]:
    """Main API-parallel analysis with resume capability"""
    
    print(f"🚀 API-PARALLEL ANALYSIS of {len(terms):,} terms")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    
    if checkpoint:
        print(f"📁 Found checkpoint - resuming analysis...")
        dictionary_words = checkpoint.get('dictionary_words', [])
        non_dictionary_words = checkpoint.get('non_dictionary_words', [])
        processed_terms = checkpoint.get('processed_terms', set())
        
        # Filter out already processed terms
        remaining_terms = [t for t in terms if t.get('term') not in processed_terms]
        
        print(f"   📊 Previously processed: {len(terms) - len(remaining_terms):,} terms")
        print(f"   📊 Dictionary words so far: {len(dictionary_words):,}")
        print(f"   📊 Non-dictionary so far: {len(non_dictionary_words):,}")
        print(f"   📊 Remaining to process: {len(remaining_terms):,}")
        
        if not remaining_terms:
            print("✅ All terms already processed!")
            return dictionary_words, non_dictionary_words
        
        terms_to_process = remaining_terms
    else:
        print(f"🆕 Starting fresh analysis...")
        dictionary_words = []
        non_dictionary_words = []
        processed_terms = set()
        terms_to_process = terms
    
    # Stage 1: Fast pre-filtering
    fast_dict, fast_non_dict, needs_api = fast_filter_terms(terms_to_process)
    
    # Add fast results to totals
    dictionary_words.extend(fast_dict)
    non_dictionary_words.extend(fast_non_dict)
    
    # Update processed terms
    for term_data in fast_dict + fast_non_dict:
        processed_terms.add(term_data.get('term'))
    
    # Save checkpoint after fast filtering
    checkpoint_data = {
        'dictionary_words': dictionary_words,
        'non_dictionary_words': non_dictionary_words,
        'processed_terms': processed_terms,
        'stage': 'fast_filtering_complete',
        'timestamp': datetime.now().isoformat()
    }
    save_checkpoint(checkpoint_file, checkpoint_data)
    
    # Stage 2: API processing for uncertain terms
    if needs_api:
        # Limit API calls if configured
        max_api_calls = config.get('max_api_calls', len(needs_api))
        
        if len(needs_api) > max_api_calls:
            print(f"⚠️  Too many terms need API check ({len(needs_api):,}), limiting to top {max_api_calls:,} by frequency")
            needs_api_sorted = sorted(needs_api, key=lambda x: x.get('frequency', 0), reverse=True)
            api_terms = needs_api_sorted[:max_api_calls]
            remaining_terms = needs_api_sorted[max_api_calls:]
            
            # Default remaining terms to non-dictionary
            for term_data in remaining_terms:
                term_data["dictionary_analysis"] = {
                    "word": term_data.get('term', ''),
                    "in_dictionary": False,
                    "reason": "exceeded_api_limit",
                    "method": "default",
                    "confidence": "assumed"
                }
                processed_terms.add(term_data.get('term'))
            
            non_dictionary_words.extend(remaining_terms)
        else:
            api_terms = needs_api
        
        if api_terms:
            print(f"\n🌐 Stage 2: Parallel API processing...")
            api_dict_words, api_non_dict_words = process_api_batch_optimized(api_terms, config)
            
            # Add API results to totals
            dictionary_words.extend(api_dict_words)
            non_dictionary_words.extend(api_non_dict_words)
            
            # Update processed terms
            for term_data in api_dict_words + api_non_dict_words:
                processed_terms.add(term_data.get('term'))
    
    # Final cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🗑️  Checkpoint cleaned up")
    
    total_terms = len(dictionary_words) + len(non_dictionary_words)
    
    print(f"\n📊 API-PARALLEL ANALYSIS COMPLETE!")
    print(f"   ✅ Total terms: {total_terms:,}")
    print(f"   📖 Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
    print(f"   🔧 Non-dictionary: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
    
    with api_stats_lock:
        print(f"   🌐 Total API calls: {api_stats['total_api_calls']:,}")
        print(f"   ⚡ API success rate: {api_stats['successful_calls']/max(1,api_stats['total_api_calls'])*100:.1f}%")
        if api_stats['total_api_calls'] > 0:
            avg_time = api_stats['processing_time'] / api_stats['total_api_calls']
            print(f"   ⏱️  Average API call time: {avg_time:.3f}s")
    
    return dictionary_words, non_dictionary_words

def save_results_api_parallel(dictionary_words: List, non_dictionary_words: List, original_metadata: Dict, config: Dict):
    """Save results with API parallelization details"""
    
    timestamp = datetime.now().isoformat()
    
    # Enhanced metadata
    base_metadata = {
        **original_metadata,
        "analysis_type": "API-Parallel English dictionary validation with resume capability",
        "analysis_date": timestamp,
        "total_terms_analyzed": len(dictionary_words) + len(non_dictionary_words),
        "api_parallel_config": config,
        "validation_methods": [
            "fast_pre_filtering", 
            "parallel_api_verification"
        ],
        "performance": {
            **api_stats,
            "api_workers": config.get('api_workers', 8),
            "api_delay": config.get('api_delay', 0.05),
            "max_concurrent": config.get('max_concurrent', 20)
        }
    }
    
    # Save files
    dict_file = "Terms_In_English_Dictionary_API_Parallel.json"
    non_dict_file = "Terms_Not_In_English_Dictionary_API_Parallel.json"
    
    dict_data = {
        "metadata": {**base_metadata, "file_type": "Dictionary words (API-parallel analysis)"},
        "dictionary_terms": dictionary_words
    }
    
    non_dict_data = {
        "metadata": {**base_metadata, "file_type": "Non-dictionary terms (API-parallel analysis)"},
        "non_dictionary_terms": non_dictionary_words
    }
    
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)
    
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        json.dump(non_dict_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(dictionary_words):,} dictionary words to {dict_file}")
    print(f"✅ Saved {len(non_dictionary_words):,} non-dictionary words to {non_dict_file}")
    
    return dict_file, non_dict_file

def load_terms_data(file_path: str, limit_top_n: Optional[int] = None) -> Tuple[Dict, List]:
    """Load terms data"""
    print(f"📖 Loading terms from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        terms = data.get('new_terms', [])
        
        if limit_top_n:
            terms = sorted(terms, key=lambda x: x.get('frequency', 0), reverse=True)[:limit_top_n]
            print(f"🔝 Limited to top {len(terms)} terms by frequency")
        
        print(f"✅ Loaded {len(terms)} terms successfully")
        return metadata, terms
        
    except Exception as e:
        print(f"❌ Failed to load terms data: {e}")
        return {}, []

def main():
    """Main API-parallel analysis function"""
    
    print("🚀 API-PARALLEL DICTIONARY ANALYSIS")
    print("=" * 60)
    
    # Configuration
    input_file = "New_Terms_Candidates_Clean.json"
    checkpoint_file = "api_parallel_checkpoint.pkl"
    
    # Get configuration
    print("📋 API-Parallel Configuration:")
    print("1. FULL analysis (~61K terms, max parallel API)")
    print("2. TOP 10,000 terms (high-frequency focus)")
    print("3. TOP 5,000 terms (balanced)")
    print("4. TOP 1,000 terms (quick test)")
    
    while True:
        try:
            choice = input("\nChoose option (1-4): ").strip()
            if choice == "1":
                limit_top_n = None
                max_api_calls = 5000  # Reasonable limit for full analysis
                api_workers = 12
                api_delay = 0.03  # 30ms delay
                max_concurrent = 25
                break
            elif choice == "2":
                limit_top_n = 10000
                max_api_calls = 3000
                api_workers = 10
                api_delay = 0.05
                max_concurrent = 20
                break
            elif choice == "3":
                limit_top_n = 5000
                max_api_calls = 2000
                api_workers = 8
                api_delay = 0.05
                max_concurrent = 15
                break
            elif choice == "4":
                limit_top_n = 1000
                max_api_calls = 500
                api_workers = 6
                api_delay = 0.1
                max_concurrent = 10
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled")
            return
    
    # Load data
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    metadata, terms = load_terms_data(input_file, limit_top_n)
    if not terms:
        print("❌ No terms loaded. Exiting.")
        return
    
    config = {
        "limit_top_n": limit_top_n,
        "max_api_calls": max_api_calls,
        "api_workers": api_workers,
        "api_delay": api_delay,
        "max_concurrent": max_concurrent,
        "total_terms_loaded": len(terms)
    }
    
    print(f"\n⚙️  API-Parallel Configuration:")
    print(f"   • Terms to analyze: {len(terms):,}")
    print(f"   • Max API calls: {max_api_calls:,}")
    print(f"   • API workers: {api_workers}")
    print(f"   • API delay: {api_delay}s")
    print(f"   • Max concurrent requests: {max_concurrent}")
    
    # Run analysis
    start_time = time.time()
    try:
        dictionary_words, non_dictionary_words = analyze_terms_api_parallel(terms, checkpoint_file, config)
        end_time = time.time()
        
        print(f"\n⏱️  Analysis completed in {end_time - start_time:.1f} seconds")
        
        # Save results
        if dictionary_words or non_dictionary_words:
            dict_file, non_dict_file = save_results_api_parallel(dictionary_words, non_dictionary_words, metadata, config)
            
            print(f"\n🎉 API-PARALLEL ANALYSIS COMPLETE!")
            print(f"📁 Output files:")
            print(f"   • Dictionary words: {dict_file}")
            print(f"   • Non-dictionary words: {non_dict_file}")
            print(f"⚡ Features: Parallel API calls, smart rate limiting, resume capability")
            
        else:
            print("❌ No results to save.")
            
    except KeyboardInterrupt:
        print(f"\n⏸️  Analysis interrupted. Progress saved to checkpoint.")
        print(f"💡 Run the script again to resume from where you left off.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if os.path.exists(checkpoint_file):
            print(f"💾 Checkpoint preserved: {checkpoint_file}")

if __name__ == "__main__":
    main()
