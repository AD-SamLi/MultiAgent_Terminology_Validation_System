#!/usr/bin/env python3
"""
Analyze terms from New_Terms_Candidates_Clean.json to determine which are valid English dictionary words
Split them into two files: dictionary words vs non-dictionary words
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO, DICT_MW
    PYMULTI_AVAILABLE = True
except ImportError:
    print("❌ PyMultiDictionary not installed. Install with: pip install PyMultiDictionary")
    PYMULTI_AVAILABLE = False

def load_terms_data(file_path: str) -> Tuple[Dict, List]:
    """Load the terms data from JSON file"""
    print(f"📖 Loading terms from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        terms = data.get('new_terms', [])
        
        print(f"✅ Loaded {len(terms)} terms successfully")
        print(f"📊 Original total: {metadata.get('total_new_terms', 'Unknown')}")
        
        return metadata, terms
        
    except Exception as e:
        print(f"❌ Failed to load terms data: {e}")
        return {}, []

def check_word_in_dictionary(word: str, dictionary: MultiDictionary) -> Dict:
    """
    Check if a word exists in the English dictionary using PyMultiDictionary
    Returns dictionary with results
    """
    word_lower = word.lower().strip()
    
    # Skip empty words or very short words that are likely abbreviations
    if not word_lower or len(word_lower) < 2:
        return {
            "word": word,
            "in_dictionary": False,
            "reason": "too_short_or_empty",
            "meaning_found": False,
            "synonyms_found": False
        }
    
    # Skip words with numbers or special characters (likely technical terms)
    if any(char.isdigit() or char in "()[]{}@#$%^&*+=|\\:;\"'<>,.?/" for char in word_lower):
        return {
            "word": word,
            "in_dictionary": False,
            "reason": "contains_special_chars_or_numbers",
            "meaning_found": False,
            "synonyms_found": False
        }
    
    result = {
        "word": word,
        "in_dictionary": False,
        "reason": None,
        "meaning_found": False,
        "synonyms_found": False,
        "meaning": None,
        "synonyms": None
    }
    
    try:
        # Try to get meaning from multiple sources
        meaning_found = False
        meaning_result = None
        
        # Try Merriam-Webster first (most authoritative for English)
        try:
            mw_meaning = dictionary.meaning('en', word_lower, dictionary=DICT_MW)
            if mw_meaning and mw_meaning != "None" and str(mw_meaning).strip():
                meaning_found = True
                meaning_result = mw_meaning
                result["dictionary_source"] = "merriam_webster"
        except:
            pass
        
        # If MW fails, try Educalingo
        if not meaning_found:
            try:
                edu_meaning = dictionary.meaning('en', word_lower, dictionary=DICT_EDUCALINGO)
                if edu_meaning and edu_meaning != "None" and str(edu_meaning).strip():
                    meaning_found = True
                    meaning_result = edu_meaning
                    result["dictionary_source"] = "educalingo"
            except:
                pass
        
        # Try to get synonyms
        synonyms_found = False
        synonyms_result = None
        try:
            synonyms = dictionary.synonym('en', word_lower)
            if synonyms and isinstance(synonyms, list) and len(synonyms) > 0:
                synonyms_found = True
                synonyms_result = synonyms[:5]  # Keep first 5 synonyms
        except:
            pass
        
        # Determine if word is in dictionary
        if meaning_found or synonyms_found:
            result["in_dictionary"] = True
            result["meaning_found"] = meaning_found
            result["synonyms_found"] = synonyms_found
            result["meaning"] = str(meaning_result)[:200] + "..." if meaning_result and len(str(meaning_result)) > 200 else meaning_result
            result["synonyms"] = synonyms_result
            result["reason"] = "found_in_dictionary"
        else:
            result["reason"] = "not_found_in_dictionary"
    
    except Exception as e:
        result["reason"] = f"error_checking: {str(e)[:100]}"
    
    return result

def analyze_terms_batch(terms: List[Dict], batch_size: int = 50, delay: float = 0.1) -> Tuple[List, List]:
    """
    Analyze terms in batches to avoid overwhelming the dictionary service
    """
    if not PYMULTI_AVAILABLE:
        print("❌ PyMultiDictionary not available. Cannot analyze terms.")
        return [], []
    
    print(f"🔍 Analyzing {len(terms)} terms in batches of {batch_size}...")
    
    dictionary = MultiDictionary()
    
    dictionary_words = []
    non_dictionary_words = []
    
    total_terms = len(terms)
    processed = 0
    
    for i in range(0, total_terms, batch_size):
        batch = terms[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_terms + batch_size - 1) // batch_size
        
        print(f"📋 Processing batch {batch_num}/{total_batches} ({len(batch)} terms)...")
        
        for term_data in batch:
            term = term_data.get('term', '')
            
            # Check if word is in dictionary
            dict_result = check_word_in_dictionary(term, dictionary)
            
            # Create enhanced term data
            enhanced_term = {
                **term_data,  # Original term data
                "dictionary_analysis": dict_result
            }
            
            # Categorize
            if dict_result["in_dictionary"]:
                dictionary_words.append(enhanced_term)
            else:
                non_dictionary_words.append(enhanced_term)
            
            processed += 1
            
            # Small delay to avoid overwhelming the service
            if delay > 0:
                time.sleep(delay)
        
        # Progress update
        print(f"   ✅ Processed {min(i + batch_size, total_terms)}/{total_terms} terms")
        print(f"   📊 Dictionary words so far: {len(dictionary_words)}")
        print(f"   📊 Non-dictionary words so far: {len(non_dictionary_words)}")
        
        # Longer delay between batches
        if i + batch_size < total_terms:
            time.sleep(1.0)
    
    print(f"\n📊 ANALYSIS COMPLETE!")
    print(f"   ✅ Total terms analyzed: {processed}")
    print(f"   📖 Dictionary words: {len(dictionary_words)} ({len(dictionary_words)/processed*100:.1f}%)")
    print(f"   🔧 Non-dictionary words: {len(non_dictionary_words)} ({len(non_dictionary_words)/processed*100:.1f}%)")
    
    return dictionary_words, non_dictionary_words

def save_results(dictionary_words: List, non_dictionary_words: List, original_metadata: Dict):
    """Save the categorized results to separate JSON files"""
    
    timestamp = datetime.now().isoformat()
    
    # Create metadata for dictionary words
    dict_metadata = {
        **original_metadata,
        "analysis_type": "English dictionary validation",
        "analysis_date": timestamp,
        "total_terms_analyzed": len(dictionary_words) + len(non_dictionary_words),
        "dictionary_words_count": len(dictionary_words),
        "dictionary_words_percentage": len(dictionary_words) / (len(dictionary_words) + len(non_dictionary_words)) * 100 if (len(dictionary_words) + len(non_dictionary_words)) > 0 else 0,
        "validation_method": "PyMultiDictionary with Merriam-Webster and Educalingo sources",
        "file_type": "Valid English dictionary words"
    }
    
    # Create metadata for non-dictionary words  
    non_dict_metadata = {
        **original_metadata,
        "analysis_type": "English dictionary validation", 
        "analysis_date": timestamp,
        "total_terms_analyzed": len(dictionary_words) + len(non_dictionary_words),
        "non_dictionary_words_count": len(non_dictionary_words),
        "non_dictionary_words_percentage": len(non_dictionary_words) / (len(dictionary_words) + len(non_dictionary_words)) * 100 if (len(dictionary_words) + len(non_dictionary_words)) > 0 else 0,
        "validation_method": "PyMultiDictionary with Merriam-Webster and Educalingo sources",
        "file_type": "Terms not found in English dictionaries (technical terms, proper nouns, etc.)"
    }
    
    # Save dictionary words
    dict_file = "Terms_In_English_Dictionary.json"
    dict_data = {
        "metadata": dict_metadata,
        "dictionary_terms": dictionary_words
    }
    
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(dictionary_words)} dictionary words to {dict_file}")
    
    # Save non-dictionary words
    non_dict_file = "Terms_Not_In_English_Dictionary.json"
    non_dict_data = {
        "metadata": non_dict_metadata,
        "non_dictionary_terms": non_dictionary_words
    }
    
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        json.dump(non_dict_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(non_dictionary_words)} non-dictionary words to {non_dict_file}")
    
    return dict_file, non_dict_file

def create_summary_report(dictionary_words: List, non_dictionary_words: List):
    """Create a summary report of the analysis"""
    
    total_terms = len(dictionary_words) + len(non_dictionary_words)
    
    print(f"\n{'='*60}")
    print("📊 DICTIONARY ANALYSIS SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"📈 Total terms analyzed: {total_terms:,}")
    print(f"📖 Terms found in English dictionaries: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
    print(f"🔧 Terms NOT found in dictionaries: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
    
    # Analyze dictionary words by frequency
    if dictionary_words:
        dict_frequencies = [term.get('frequency', 0) for term in dictionary_words]
        total_dict_freq = sum(dict_frequencies)
        avg_dict_freq = total_dict_freq / len(dictionary_words) if dictionary_words else 0
        
        print(f"\n📖 DICTIONARY WORDS ANALYSIS:")
        print(f"   • Total frequency: {total_dict_freq:,}")
        print(f"   • Average frequency: {avg_dict_freq:.1f}")
        
        # Top dictionary words
        dict_sorted = sorted(dictionary_words, key=lambda x: x.get('frequency', 0), reverse=True)
        print(f"   • Top 5 dictionary words:")
        for i, term in enumerate(dict_sorted[:5], 1):
            dict_info = term.get('dictionary_analysis', {})
            source = dict_info.get('dictionary_source', 'unknown')
            print(f"     {i}. '{term.get('term', 'N/A')}' (freq: {term.get('frequency', 0)}, source: {source})")
    
    # Analyze non-dictionary words by frequency
    if non_dictionary_words:
        non_dict_frequencies = [term.get('frequency', 0) for term in non_dictionary_words]
        total_non_dict_freq = sum(non_dict_frequencies)
        avg_non_dict_freq = total_non_dict_freq / len(non_dictionary_words) if non_dictionary_words else 0
        
        print(f"\n🔧 NON-DICTIONARY WORDS ANALYSIS:")
        print(f"   • Total frequency: {total_non_dict_freq:,}")
        print(f"   • Average frequency: {avg_non_dict_freq:.1f}")
        
        # Top non-dictionary words
        non_dict_sorted = sorted(non_dictionary_words, key=lambda x: x.get('frequency', 0), reverse=True)
        print(f"   • Top 5 non-dictionary words:")
        for i, term in enumerate(non_dict_sorted[:5], 1):
            dict_info = term.get('dictionary_analysis', {})
            reason = dict_info.get('reason', 'unknown')
            print(f"     {i}. '{term.get('term', 'N/A')}' (freq: {term.get('frequency', 0)}, reason: {reason})")
        
        # Analyze reasons for non-dictionary classification
        reasons = {}
        for term in non_dictionary_words:
            reason = term.get('dictionary_analysis', {}).get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        
        print(f"\n🔍 REASONS FOR NON-DICTIONARY CLASSIFICATION:")
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(non_dictionary_words) * 100
            print(f"   • {reason}: {count:,} terms ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")

def main():
    """Main function to analyze terms"""
    
    print("🔍 DICTIONARY TERMS ANALYSIS")
    print("=" * 50)
    
    if not PYMULTI_AVAILABLE:
        print("❌ PyMultiDictionary is required for this analysis.")
        print("💡 Install with: pip install PyMultiDictionary")
        return
    
    # Load terms data
    input_file = "New_Terms_Candidates_Clean.json"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    metadata, terms = load_terms_data(input_file)
    if not terms:
        print("❌ No terms loaded. Exiting.")
        return
    
    print(f"\n🎯 Starting dictionary analysis of {len(terms)} terms...")
    print("⚠️  This may take several minutes due to API rate limiting...")
    
    # Analyze terms
    start_time = time.time()
    dictionary_words, non_dictionary_words = analyze_terms_batch(terms, batch_size=30, delay=0.2)
    end_time = time.time()
    
    print(f"\n⏱️  Analysis completed in {end_time - start_time:.1f} seconds")
    
    # Save results
    if dictionary_words or non_dictionary_words:
        dict_file, non_dict_file = save_results(dictionary_words, non_dictionary_words, metadata)
        
        # Create summary report
        create_summary_report(dictionary_words, non_dictionary_words)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Output files:")
        print(f"   • Dictionary words: {dict_file}")
        print(f"   • Non-dictionary words: {non_dict_file}")
        
    else:
        print("❌ No results to save.")

if __name__ == "__main__":
    main()
