#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED parallel dictionary analysis with resume capability
Features:
1. Multi-stage filtering (heuristics → common words → API)
2. Parallel processing for API calls
3. Resume from checkpoint if interrupted
4. Progress tracking and real-time updates
5. Configurable batch sizes and worker counts
"""

import json
import time
import os
import re
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp

try:
    from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO, DICT_MW
    PYMULTI_AVAILABLE = True
except ImportError:
    print("❌ PyMultiDictionary not installed. Install with: pip install PyMultiDictionary")
    PYMULTI_AVAILABLE = False

# Global progress tracking
progress_lock = Lock()
global_progress = {
    "processed": 0,
    "dictionary_words": 0,
    "non_dictionary_words": 0,
    "api_calls": 0
}

# Common English words list (expanded)
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
    'without', 'second', 'later', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far', 'indian', 'really', 'almost', 'let',
    'above', 'girl', 'sometimes', 'mountain', 'cut', 'young', 'talk', 'soon', 'list', 'song', 'leave', 'family', 'body', 'music',
    'color', 'stand', 'sun', 'questions', 'fish', 'area', 'mark', 'dog', 'horse', 'birds', 'problem', 'complete', 'room', 'knew',
    'since', 'ever', 'piece', 'told', 'usually', 'friends', 'easy', 'heard', 'order', 'red', 'door', 'sure', 'become',
    'top', 'ship', 'across', 'today', 'short', 'better', 'best', 'however', 'low', 'hours', 'black', 'products', 'happened',
    'whole', 'measure', 'remember', 'waves', 'reached', 'listen', 'wind', 'rock', 'space', 'covered', 'fast', 'several',
    'hold', 'himself', 'toward', 'five', 'step', 'morning', 'passed', 'vowel', 'true', 'hundred', 'against', 'pattern', 'numeral',
    'table', 'north', 'slowly', 'money', 'map', 'farm', 'pulled', 'draw', 'voice', 'seen', 'cold', 'cried', 'plan', 'notice',
    'south', 'sing', 'war', 'ground', 'fall', 'king', 'town', 'unit', 'figure', 'certain', 'field', 'travel', 'wood', 'fire',
    'upon', 'done', 'english', 'road', 'half', 'ten', 'fly', 'gave', 'box', 'finally', 'wait', 'correct', 'oh', 'quickly', 'person',
    'became', 'shown', 'minutes', 'strong', 'verb', 'stars', 'front', 'feel', 'fact', 'inches', 'street', 'decided', 'contain',
    'course', 'surface', 'produce', 'building', 'ocean', 'class', 'note', 'nothing', 'rest', 'carefully', 'scientists', 'inside',
    'wheels', 'stay', 'green', 'known', 'island', 'week', 'less', 'machine', 'base', 'ago', 'stood', 'plane', 'system', 'behind',
    'ran', 'round', 'boat', 'game', 'force', 'brought', 'heat', 'quite', 'broken', 'case', 'middle', 'kill', 'son',
    'lake', 'moment', 'scale', 'loud', 'spring', 'observing', 'child', 'straight', 'consonant', 'nation', 'dictionary', 'milk',
    'speed', 'method', 'organ', 'pay', 'age', 'section', 'dress', 'cloud', 'surprise', 'quiet', 'stone', 'tiny', 'climb', 'bad',
    'oil', 'blood', 'touch', 'grew', 'cent', 'mix', 'team', 'wire', 'cost', 'lost', 'brown', 'wear', 'garden', 'equal', 'sent',
    'choose', 'fell', 'fit', 'flow', 'fair', 'bank', 'collect', 'save', 'control', 'decimal', 'ear', 'else', 'broke',
    'gentle', 'woman', 'captain', 'practice', 'separate', 'difficult', 'doctor', 'please', 'protect', 'noon', 'whose', 'locate',
    'ring', 'character', 'insect', 'caught', 'period', 'indicate', 'radio', 'spoke', 'atom', 'history', 'effect', 'electric',
    'expect', 'crop', 'modern', 'element', 'hit', 'student', 'corner', 'party', 'supply', 'bone', 'rail', 'imagine', 'provide',
    'agree', 'thus', 'capital', 'chair', 'danger', 'fruit', 'rich', 'thick', 'soldier', 'process', 'operate', 'guess',
    'necessary', 'sharp', 'wing', 'create', 'neighbor', 'wash', 'bat', 'rather', 'crowd', 'corn', 'compare', 'poem', 'string',
    'bell', 'depend', 'meat', 'rub', 'tube', 'famous', 'dollar', 'stream', 'fear', 'sight', 'thin', 'triangle', 'planet', 'hurry',
    'chief', 'colony', 'clock', 'mine', 'tie', 'enter', 'major', 'fresh', 'search', 'send', 'yellow', 'gun', 'allow', 'print',
    'dead', 'spot', 'desert', 'suit', 'current', 'lift', 'rose', 'continue', 'block', 'chart', 'hat', 'sell', 'success', 'company',
    'subtract', 'event', 'particular', 'deal', 'swim', 'term', 'opposite', 'wife', 'shoe', 'shoulder', 'spread', 'arrange', 'camp',
    'invent', 'cotton', 'born', 'determine', 'quart', 'nine', 'truck', 'noise', 'level', 'chance', 'gather', 'shop', 'stretch',
    'throw', 'shine', 'property', 'column', 'molecule', 'select', 'wrong', 'gray', 'repeat', 'require', 'broad', 'prepare', 'salt',
    'nose', 'plural', 'anger', 'claim', 'continent', 'oxygen', 'sugar', 'death', 'pretty', 'skill', 'women', 'season', 'solution',
    'magnet', 'silver', 'thank', 'branch', 'match', 'suffix', 'especially', 'fig', 'afraid', 'huge', 'sister', 'steel', 'discuss',
    'forward', 'similar', 'guide', 'experience', 'score', 'apple', 'bought', 'led', 'pitch', 'coat', 'mass', 'card', 'band', 'rope',
    'slip', 'win', 'dream', 'evening', 'condition', 'feed', 'tool', 'total', 'basic', 'smell', 'valley', 'nor', 'double', 'seat',
    'arrive', 'master', 'track', 'parent', 'shore', 'division', 'sheet', 'substance', 'favor', 'connect', 'post', 'spend', 'chord',
    'fat', 'glad', 'original', 'share', 'station', 'dad', 'bread', 'charge', 'proper', 'bar', 'offer', 'segment', 'slave', 'duck',
    'instant', 'market', 'degree', 'populate', 'chick', 'dear', 'enemy', 'reply', 'drink', 'occur', 'support', 'speech', 'nature',
    'range', 'steam', 'motion', 'path', 'liquid', 'log', 'meant', 'quotient', 'teeth', 'shell', 'neck'
}

# Extended technical indicators
TECHNICAL_PREFIXES = {
    'auto', 'multi', 'pre', 'post', 'sub', 'super', 'inter', 'intra', 'anti', 'pro', 'semi', 'non', 'un', 'over', 'under',
    'out', 'up', 'down', 'in', 'off', 'on', 're', 'de', 'ex', 'co', 'counter', 'cross', 'trans', 'ultra', 'mega', 'micro',
    'macro', 'mini', 'maxi', 'hyper', 'hypo', 'neo', 'pseudo', 'quasi', 'self', 'meta'
}

TECHNICAL_SUFFIXES = {
    'ware', 'tech', 'soft', 'app', 'sys', 'api', 'xml', 'html', 'css', 'exe', 'dll', 'cfg', 'ini', 'log', 'tmp', 'bak',
    'cad', 'cam', 'cae', 'bim', 'gis', 'erp', 'crm', 'scm', 'plm', 'pdm', 'matic', 'tron', 'sync', 'async', 'auto',
    'manual', 'digital', 'analog', 'virtual', 'cyber', 'online', 'offline', 'real', 'time', 'space', 'scape', 'scope',
    'gram', 'graph', 'logy', 'ology', 'ism', 'ist', 'ize', 'ise', 'able', 'ible', 'ful', 'less', 'ness', 'ment', 'tion',
    'sion', 'ance', 'ence', 'ity', 'ety', 'ary', 'ery', 'ory', 'ive', 'ous', 'eous', 'ious'
}

BRAND_INDICATORS = {
    'autodesk', 'microsoft', 'adobe', 'google', 'apple', 'intel', 'nvidia', 'amd', 'ibm', 'oracle', 'sap', 'cisco',
    'dell', 'hp', 'lenovo', 'asus', 'samsung', 'lg', 'sony', 'canon', 'nikon', 'siemens', 'bosch', 'ge', 'philips'
}

FILE_EXTENSIONS = {
    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'rtf', 'csv', 'json', 'xml', 'html', 'css', 'js', 'py',
    'java', 'cpp', 'c', 'h', 'cs', 'vb', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'scala', 'sql', 'md', 'tex', 'bib',
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'ico', 'webp', 'mp3', 'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv',
    'wav', 'flac', 'ogg', 'zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz', 'iso', 'dmg', 'exe', 'msi', 'deb', 'rpm'
}

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint data if exists"""
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

def is_likely_technical_term(word: str) -> Tuple[bool, str]:
    """Enhanced heuristic check for technical terms"""
    word_lower = word.lower().strip()
    
    if not word_lower or len(word_lower) < 2:
        return True, "too_short_or_empty"
    
    if len(word_lower) > 30:
        return True, "too_long"
    
    # Contains numbers
    if any(char.isdigit() for char in word_lower):
        return True, "contains_numbers"
    
    # Contains special characters
    special_chars = "()[]{}@#$%^&*+=|\\:;\"<>,.?/~`"
    if any(char in special_chars for char in word_lower):
        return True, "contains_special_chars"
    
    # Multiple consecutive uppercase (acronyms)
    if re.search(r'[A-Z]{2,}', word):
        return True, "likely_acronym"
    
    # Technical prefixes
    for prefix in TECHNICAL_PREFIXES:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            remainder = word_lower[len(prefix):]
            if any(remainder.endswith(suffix) for suffix in TECHNICAL_SUFFIXES):
                return True, f"technical_compound_{prefix}"
    
    # Technical suffixes
    for suffix in TECHNICAL_SUFFIXES:
        if word_lower.endswith(suffix):
            return True, f"technical_suffix_{suffix}"
    
    # Brand names
    if word[0].isupper() and word_lower not in COMMON_ENGLISH_WORDS:
        if any(brand in word_lower for brand in BRAND_INDICATORS):
            return True, "brand_name"
    
    # File extensions
    if word_lower in FILE_EXTENSIONS:
        return True, "file_extension"
    
    # CamelCase (likely technical)
    if re.search(r'[a-z][A-Z]', word):
        return True, "camelcase"
    
    # Contains underscores or hyphens (technical naming)
    if '_' in word or '-' in word:
        return True, "technical_naming"
    
    return False, "passed_heuristics"

def check_word_fast(word: str) -> Dict:
    """Fast multi-stage word checking"""
    word_lower = word.lower().strip()
    
    # Stage 1: Heuristic filtering
    is_technical, reason = is_likely_technical_term(word)
    if is_technical:
        return {
            "word": word,
            "in_dictionary": False,
            "reason": reason,
            "method": "heuristic_filter",
            "confidence": "high" if reason in ["contains_numbers", "contains_special_chars", "too_long", "camelcase"] else "medium"
        }
    
    # Stage 2: Common words check
    if word_lower in COMMON_ENGLISH_WORDS:
        return {
            "word": word,
            "in_dictionary": True,
            "reason": "common_english_word",
            "method": "common_words_list",
            "confidence": "high"
        }
    
    # Stage 3: Basic English patterns
    # Plurals
    if word_lower.endswith('s') and len(word_lower) > 3:
        singular = word_lower[:-1]
        if singular in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "plural_of_common_word",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": singular
            }
    
    # Past tense (-ed)
    if word_lower.endswith('ed') and len(word_lower) > 4:
        base = word_lower[:-2]
        if base in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "past_tense_of_common_word",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": base
            }
        # Handle doubled consonant (e.g., "stopped" -> "stop")
        if len(base) > 2 and base[-1] == base[-2]:
            base_single = base[:-1]
            if base_single in COMMON_ENGLISH_WORDS:
                return {
                    "word": word,
                    "in_dictionary": True,
                    "reason": "past_tense_doubled_consonant",
                    "method": "pattern_matching",
                    "confidence": "medium",
                    "base_word": base_single
                }
    
    # Present participle (-ing)
    if word_lower.endswith('ing') and len(word_lower) > 5:
        base = word_lower[:-3]
        if base in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "ing_form_of_common_word",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": base
            }
        # Handle doubled consonant (e.g., "running" -> "run")
        if len(base) > 2 and base[-1] == base[-2]:
            base_single = base[:-1]
            if base_single in COMMON_ENGLISH_WORDS:
                return {
                    "word": word,
                    "in_dictionary": True,
                    "reason": "ing_form_doubled_consonant",
                    "method": "pattern_matching",
                    "confidence": "medium",
                    "base_word": base_single
                }
    
    # Comparative (-er) and superlative (-est)
    if word_lower.endswith('er') and len(word_lower) > 4:
        base = word_lower[:-2]
        if base in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "comparative_form",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": base
            }
    
    if word_lower.endswith('est') and len(word_lower) > 5:
        base = word_lower[:-3]
        if base in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "superlative_form",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": base
            }
    
    # Adverbs (-ly)
    if word_lower.endswith('ly') and len(word_lower) > 4:
        base = word_lower[:-2]
        if base in COMMON_ENGLISH_WORDS:
            return {
                "word": word,
                "in_dictionary": True,
                "reason": "adverb_form",
                "method": "pattern_matching",
                "confidence": "medium",
                "base_word": base
            }
    
    # Need API check
    return {
        "word": word,
        "in_dictionary": None,
        "reason": "requires_api_check",
        "method": "needs_verification",
        "confidence": "unknown"
    }

def check_word_api_worker(word: str) -> Dict:
    """API-based word checking for parallel processing"""
    global global_progress
    
    result = {
        "word": word,
        "in_dictionary": False,
        "method": "api_check",
        "meaning_found": False,
        "synonyms_found": False
    }
    
    try:
        dictionary = MultiDictionary()
        word_lower = word.lower().strip()
        
        # Try Merriam-Webster first (fastest, most authoritative)
        try:
            mw_meaning = dictionary.meaning('en', word_lower, dictionary=DICT_MW)
            if mw_meaning and str(mw_meaning).strip() and str(mw_meaning) != "None":
                result["in_dictionary"] = True
                result["meaning_found"] = True
                result["dictionary_source"] = "merriam_webster"
                result["confidence"] = "high"
                
                with progress_lock:
                    global_progress["api_calls"] += 1
                return result
        except:
            pass
        
        # Try synonyms (often faster than meaning)
        try:
            synonyms = dictionary.synonym('en', word_lower)
            if synonyms and isinstance(synonyms, list) and len(synonyms) > 0:
                result["in_dictionary"] = True
                result["synonyms_found"] = True
                result["dictionary_source"] = "synonym_check"
                result["confidence"] = "medium"
                
                with progress_lock:
                    global_progress["api_calls"] += 1
                return result
        except:
            pass
        
        # Final check with Educalingo
        try:
            edu_meaning = dictionary.meaning('en', word_lower, dictionary=DICT_EDUCALINGO)
            if edu_meaning and str(edu_meaning).strip() and str(edu_meaning) != "None":
                result["in_dictionary"] = True
                result["meaning_found"] = True
                result["dictionary_source"] = "educalingo"
                result["confidence"] = "medium"
                
                with progress_lock:
                    global_progress["api_calls"] += 1
                return result
        except:
            pass
        
        # Not found
        result["reason"] = "not_found_in_dictionaries"
        result["confidence"] = "high"
        
    except Exception as e:
        result["reason"] = f"api_error: {str(e)[:50]}"
        result["confidence"] = "error"
    
    with progress_lock:
        global_progress["api_calls"] += 1
    
    return result

def process_terms_batch(terms_batch: List[Dict], batch_id: int) -> Tuple[List, List]:
    """Process a batch of terms with fast filtering"""
    batch_dict_words = []
    batch_non_dict_words = []
    batch_uncertain = []
    
    # Fast filtering stage
    for term_data in terms_batch:
        term = term_data.get('term', '')
        fast_result = check_word_fast(term)
        
        enhanced_term = {
            **term_data,
            "dictionary_analysis": fast_result
        }
        
        if fast_result.get("in_dictionary") is True:
            batch_dict_words.append(enhanced_term)
        elif fast_result.get("in_dictionary") is False:
            batch_non_dict_words.append(enhanced_term)
        else:
            batch_uncertain.append(enhanced_term)
    
    return batch_dict_words, batch_non_dict_words, batch_uncertain

def process_api_batch_parallel(uncertain_terms: List[Dict], max_workers: int = 8) -> Tuple[List, List]:
    """Process uncertain terms with parallel API calls"""
    if not PYMULTI_AVAILABLE:
        # Default all to non-dictionary
        for term in uncertain_terms:
            term["dictionary_analysis"]["in_dictionary"] = False
            term["dictionary_analysis"]["reason"] = "no_api_available"
            term["dictionary_analysis"]["confidence"] = "assumed"
        return [], uncertain_terms
    
    api_dict_words = []
    api_non_dict_words = []
    
    # Parallel API processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all API calls
        future_to_term = {}
        for term_data in uncertain_terms:
            term = term_data.get('term', '')
            future = executor.submit(check_word_api_worker, term)
            future_to_term[future] = term_data
        
        # Collect results
        for future in as_completed(future_to_term):
            term_data = future_to_term[future]
            try:
                api_result = future.get(timeout=30)  # 30 second timeout per word
                term_data["dictionary_analysis"] = api_result
                
                if api_result["in_dictionary"]:
                    api_dict_words.append(term_data)
                else:
                    api_non_dict_words.append(term_data)
                    
            except Exception as e:
                # Handle timeout or other errors
                term_data["dictionary_analysis"]["in_dictionary"] = False
                term_data["dictionary_analysis"]["reason"] = f"api_timeout_or_error: {str(e)[:30]}"
                term_data["dictionary_analysis"]["confidence"] = "error"
                api_non_dict_words.append(term_data)
    
    return api_dict_words, api_non_dict_words

def analyze_terms_parallel(terms: List[Dict], checkpoint_file: str, config: Dict) -> Tuple[List, List]:
    """Main parallel analysis function with checkpointing"""
    
    print(f"🚀 PARALLEL ANALYSIS of {len(terms)} terms")
    print(f"⚡ Config: {config['batch_size']} batch size, {config['max_workers']} workers, {config['max_api_calls']} API limit")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    start_idx = 0
    dictionary_words = []
    non_dictionary_words = []
    all_uncertain_terms = []
    
    if checkpoint:
        start_idx = checkpoint.get('last_processed_idx', 0)
        dictionary_words = checkpoint.get('dictionary_words', [])
        non_dictionary_words = checkpoint.get('non_dictionary_words', [])
        all_uncertain_terms = checkpoint.get('uncertain_terms', [])
        print(f"📁 Resuming from checkpoint: processed {start_idx}/{len(terms)} terms")
        print(f"   📊 Dict words so far: {len(dictionary_words)}")
        print(f"   📊 Non-dict words so far: {len(non_dictionary_words)}")
        print(f"   ❓ Uncertain terms so far: {len(all_uncertain_terms)}")
    
    # Stage 1: Fast batch processing
    print(f"\n🔍 Stage 1: Fast filtering (batches of {config['batch_size']})...")
    
    batch_size = config['batch_size']
    total_batches = (len(terms) - start_idx + batch_size - 1) // batch_size
    
    for i in range(start_idx, len(terms), batch_size):
        batch_end = min(i + batch_size, len(terms))
        batch_terms = terms[i:batch_end]
        batch_num = (i // batch_size) + 1
        
        print(f"   Batch {batch_num}/{total_batches}: processing terms {i+1}-{batch_end}")
        
        # Process batch
        batch_dict, batch_non_dict, batch_uncertain = process_terms_batch(batch_terms, batch_num)
        
        # Accumulate results
        dictionary_words.extend(batch_dict)
        non_dictionary_words.extend(batch_non_dict)
        all_uncertain_terms.extend(batch_uncertain)
        
        # Update global progress
        with progress_lock:
            global_progress["processed"] = batch_end
            global_progress["dictionary_words"] = len(dictionary_words)
            global_progress["non_dictionary_words"] = len(non_dictionary_words)
        
        # Save checkpoint every 10 batches
        if batch_num % 10 == 0:
            checkpoint_data = {
                'last_processed_idx': batch_end,
                'dictionary_words': dictionary_words,
                'non_dictionary_words': non_dictionary_words,
                'uncertain_terms': all_uncertain_terms,
                'timestamp': datetime.now().isoformat()
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            print(f"   💾 Checkpoint saved at batch {batch_num}")
        
        # Progress update
        if batch_num % 5 == 0:
            print(f"   📊 Progress: {len(dictionary_words)} dict, {len(non_dictionary_words)} non-dict, {len(all_uncertain_terms)} uncertain")
    
    print(f"✅ Stage 1 complete:")
    print(f"   📊 Dictionary words: {len(dictionary_words)}")
    print(f"   📊 Non-dictionary words: {len(non_dictionary_words)}")
    print(f"   ❓ Uncertain terms: {len(all_uncertain_terms)}")
    
    # Stage 2: Parallel API processing for uncertain terms
    if all_uncertain_terms and len(all_uncertain_terms) <= config['max_api_calls']:
        print(f"\n🌐 Stage 2: Parallel API verification of {len(all_uncertain_terms)} uncertain terms...")
        
        api_dict_words, api_non_dict_words = process_api_batch_parallel(
            all_uncertain_terms, 
            config['max_workers']
        )
        
        dictionary_words.extend(api_dict_words)
        non_dictionary_words.extend(api_non_dict_words)
        
        print(f"✅ API processing complete:")
        print(f"   📖 Additional dictionary words found: {len(api_dict_words)}")
        print(f"   🔧 Additional non-dictionary words: {len(api_non_dict_words)}")
        
    elif all_uncertain_terms:
        # Too many uncertain terms - process top frequency ones
        print(f"\n🌐 Stage 2: Limited API processing (top {config['max_api_calls']} by frequency)...")
        
        # Sort by frequency and take top N
        sorted_uncertain = sorted(all_uncertain_terms, key=lambda x: x.get('frequency', 0), reverse=True)
        top_uncertain = sorted_uncertain[:config['max_api_calls']]
        remaining_uncertain = sorted_uncertain[config['max_api_calls']:]
        
        api_dict_words, api_non_dict_words = process_api_batch_parallel(
            top_uncertain,
            config['max_workers']
        )
        
        dictionary_words.extend(api_dict_words)
        non_dictionary_words.extend(api_non_dict_words)
        
        # Default remaining to non-dictionary
        for term in remaining_uncertain:
            term["dictionary_analysis"]["in_dictionary"] = False
            term["dictionary_analysis"]["reason"] = "exceeded_api_limit"
            term["dictionary_analysis"]["confidence"] = "assumed"
        
        non_dictionary_words.extend(remaining_uncertain)
        
        print(f"✅ Limited API processing complete:")
        print(f"   📖 Dictionary words from top terms: {len(api_dict_words)}")
        print(f"   🔧 Non-dictionary (API checked): {len(api_non_dict_words)}")
        print(f"   🔧 Non-dictionary (assumed): {len(remaining_uncertain)}")
    
    # Final checkpoint cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🗑️  Checkpoint file cleaned up")
    
    total_terms = len(dictionary_words) + len(non_dictionary_words)
    
    print(f"\n📊 PARALLEL ANALYSIS COMPLETE!")
    print(f"   ✅ Total terms: {total_terms:,}")
    print(f"   📖 Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
    print(f"   🔧 Non-dictionary: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
    print(f"   🌐 API calls made: {global_progress['api_calls']:,}")
    
    return dictionary_words, non_dictionary_words

def save_results_parallel(dictionary_words: List, non_dictionary_words: List, original_metadata: Dict, config: Dict):
    """Save results with parallel processing details"""
    
    timestamp = datetime.now().isoformat()
    
    # Enhanced metadata
    base_metadata = {
        **original_metadata,
        "analysis_type": "Parallel English dictionary validation with resume capability",
        "analysis_date": timestamp,
        "total_terms_analyzed": len(dictionary_words) + len(non_dictionary_words),
        "parallel_config": config,
        "validation_methods": [
            "heuristic_filtering", 
            "common_words_list", 
            "pattern_matching", 
            "parallel_api_verification"
        ],
        "performance": {
            "api_calls_made": global_progress["api_calls"],
            "batch_processing": True,
            "parallel_workers": config["max_workers"],
            "checkpoint_enabled": True
        }
    }
    
    # Dictionary words file
    dict_metadata = {
        **base_metadata,
        "dictionary_words_count": len(dictionary_words),
        "dictionary_words_percentage": len(dictionary_words) / (len(dictionary_words) + len(non_dictionary_words)) * 100,
        "file_type": "Valid English dictionary words (parallel analysis)"
    }
    
    # Non-dictionary words file
    non_dict_metadata = {
        **base_metadata,
        "non_dictionary_words_count": len(non_dictionary_words),
        "non_dictionary_words_percentage": len(non_dictionary_words) / (len(dictionary_words) + len(non_dictionary_words)) * 100,
        "file_type": "Terms not in English dictionaries - technical terms, proper nouns, etc. (parallel analysis)"
    }
    
    # Save files
    dict_file = "Terms_In_English_Dictionary_Parallel.json"
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump({"metadata": dict_metadata, "dictionary_terms": dictionary_words}, f, indent=2, ensure_ascii=False)
    
    non_dict_file = "Terms_Not_In_English_Dictionary_Parallel.json"
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        json.dump({"metadata": non_dict_metadata, "non_dictionary_terms": non_dictionary_words}, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(dictionary_words):,} dictionary words to {dict_file}")
    print(f"✅ Saved {len(non_dictionary_words):,} non-dictionary words to {non_dict_file}")
    
    return dict_file, non_dict_file

def create_detailed_report(dictionary_words: List, non_dictionary_words: List, config: Dict):
    """Create comprehensive analysis report"""
    
    total_terms = len(dictionary_words) + len(non_dictionary_words)
    
    print(f"\n{'='*80}")
    print("📊 PARALLEL DICTIONARY ANALYSIS REPORT")
    print(f"{'='*80}")
    
    print(f"🚀 PERFORMANCE METRICS:")
    print(f"   • Parallel workers used: {config['max_workers']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • API calls limit: {config['max_api_calls']}")
    print(f"   • API calls made: {global_progress['api_calls']:,}")
    
    print(f"\n📈 ANALYSIS RESULTS:")
    print(f"   • Total terms analyzed: {total_terms:,}")
    print(f"   • Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
    print(f"   • Non-dictionary words: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
    
    # Method breakdown
    print(f"\n🔍 ANALYSIS METHODS BREAKDOWN:")
    
    all_terms = dictionary_words + non_dictionary_words
    method_counts = Counter()
    confidence_counts = Counter()
    
    for term in all_terms:
        analysis = term.get('dictionary_analysis', {})
        method = analysis.get('method', 'unknown')
        confidence = analysis.get('confidence', 'unknown')
        method_counts[method] += 1
        confidence_counts[confidence] += 1
    
    for method, count in method_counts.most_common():
        percentage = count / total_terms * 100
        print(f"   • {method}: {count:,} terms ({percentage:.1f}%)")
    
    print(f"\n🎯 CONFIDENCE LEVELS:")
    for confidence, count in confidence_counts.most_common():
        percentage = count / total_terms * 100
        print(f"   • {confidence}: {count:,} terms ({percentage:.1f}%)")
    
    # Top dictionary words
    if dictionary_words:
        dict_sorted = sorted(dictionary_words, key=lambda x: x.get('frequency', 0), reverse=True)
        print(f"\n📖 TOP DICTIONARY WORDS:")
        for i, term in enumerate(dict_sorted[:10], 1):
            analysis = term.get('dictionary_analysis', {})
            method = analysis.get('method', 'unknown')
            freq = term.get('frequency', 0)
            print(f"   {i:2}. '{term.get('term', 'N/A'):20}' (freq: {freq:4}, method: {method})")
    
    # Top non-dictionary words
    if non_dictionary_words:
        non_dict_sorted = sorted(non_dictionary_words, key=lambda x: x.get('frequency', 0), reverse=True)
        print(f"\n🔧 TOP NON-DICTIONARY WORDS:")
        for i, term in enumerate(non_dict_sorted[:10], 1):
            analysis = term.get('dictionary_analysis', {})
            reason = analysis.get('reason', 'unknown')
            freq = term.get('frequency', 0)
            print(f"   {i:2}. '{term.get('term', 'N/A'):20}' (freq: {freq:4}, reason: {reason})")
    
    print(f"\n{'='*80}")

def load_terms_data(file_path: str, limit_top_n: Optional[int] = None) -> Tuple[Dict, List]:
    """Load terms data with optional limiting"""
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
    """Main parallel analysis function"""
    
    print("🚀 PARALLEL DICTIONARY ANALYSIS WITH RESUME CAPABILITY")
    print("=" * 70)
    
    # Configuration
    input_file = "New_Terms_Candidates_Clean.json"
    checkpoint_file = "dictionary_analysis_checkpoint.pkl"
    
    # Get user configuration
    print("📋 Analysis Configuration:")
    print("1. FULL analysis (~61K terms, parallel, resume-capable)")
    print("2. TOP 10,000 terms (high-frequency focus)")
    print("3. TOP 5,000 terms (balanced)")
    print("4. TOP 1,000 terms (quick test)")
    
    while True:
        try:
            choice = input("\nChoose option (1-4): ").strip()
            if choice == "1":
                limit_top_n = None
                max_api_calls = 3000
                batch_size = 100
                max_workers = 8
                break
            elif choice == "2":
                limit_top_n = 10000
                max_api_calls = 2000
                batch_size = 50
                max_workers = 6
                break
            elif choice == "3":
                limit_top_n = 5000
                max_api_calls = 1000
                batch_size = 50
                max_workers = 4
                break
            elif choice == "4":
                limit_top_n = 1000
                max_api_calls = 500
                batch_size = 25
                max_workers = 4
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
        "batch_size": batch_size,
        "max_workers": max_workers,
        "total_terms_loaded": len(terms)
    }
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Terms to analyze: {len(terms):,}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Max workers: {max_workers}")
    print(f"   • Max API calls: {max_api_calls:,}")
    print(f"   • Checkpoint file: {checkpoint_file}")
    
    # Run parallel analysis
    start_time = time.time()
    try:
        dictionary_words, non_dictionary_words = analyze_terms_parallel(terms, checkpoint_file, config)
        end_time = time.time()
        
        print(f"\n⏱️  Analysis completed in {end_time - start_time:.1f} seconds")
        
        # Save results
        if dictionary_words or non_dictionary_words:
            dict_file, non_dict_file = save_results_parallel(dictionary_words, non_dictionary_words, metadata, config)
            create_detailed_report(dictionary_words, non_dictionary_words, config)
            
            print(f"\n🎉 PARALLEL ANALYSIS COMPLETE!")
            print(f"📁 Output files:")
            print(f"   • Dictionary words: {dict_file}")
            print(f"   • Non-dictionary words: {non_dict_file}")
            print(f"⚡ Features: Parallel processing, resume capability, optimized filtering")
            
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
