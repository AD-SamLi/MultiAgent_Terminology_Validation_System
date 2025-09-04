#!/usr/bin/env python3
"""
Split new terms candidates into dictionary terms and non-dictionary terms
Uses DictionarySmolTool to check if terms exist in English dictionaries
"""

import json
import os
import time
from typing import Dict, List, Tuple
from dictionary_agent import DictionarySmolTool

def load_new_terms_candidates(json_file: str) -> Dict:
    """Load the new terms candidates data"""
    print(f"📖 Loading new terms candidates from: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {data['metadata']['total_new_terms']:,} new terms candidates")
        return data
        
    except Exception as e:
        print(f"❌ Error loading new terms candidates: {e}")
        return {}

def check_term_in_dictionary(tool: DictionarySmolTool, term: str) -> Tuple[bool, str, Dict]:
    """
    Check if a term exists in the English dictionary
    
    Returns:
        (exists, status_message, full_response_data)
    """
    try:
        # Try to get meaning using educalingo (most comprehensive)
        result = tool.forward("meaning", word=term, language="en", dictionary_source="educalingo")
        data = json.loads(result)
        
        if 'error' in data:
            # Try with Merriam-Webster as backup
            result_mw = tool.forward("meaning", word=term, language="en", dictionary_source="merriam_webster")
            data_mw = json.loads(result_mw)
            
            if 'error' in data_mw:
                return False, f"Not found in dictionaries", {"educalingo_error": data.get('error'), "merriam_webster_error": data_mw.get('error')}
            else:
                return True, "Found in Merriam-Webster", data_mw
        else:
            return True, "Found in Educalingo", data
            
    except Exception as e:
        return False, f"Error checking term: {str(e)}", {"exception": str(e)}

def analyze_terms_with_dictionary(new_terms_data: Dict, batch_size: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze all terms to see which exist in dictionaries
    
    Returns:
        (dictionary_terms, non_dictionary_terms)
    """
    print("🔍 Analyzing terms with dictionary tool...")
    
    # Initialize dictionary tool
    tool = DictionarySmolTool()
    
    terms_list = new_terms_data.get('new_terms', [])
    total_terms = len(terms_list)
    
    dictionary_terms = []
    non_dictionary_terms = []
    
    print(f"📊 Processing {total_terms:,} terms...")
    print("This may take a while as we check each term against dictionary sources...")
    
    start_time = time.time()
    
    for idx, term_entry in enumerate(terms_list):
        if idx % 100 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total_terms - idx) / rate if rate > 0 else 0
            print(f"   Processed {idx:,}/{total_terms:,} terms ({idx/total_terms*100:.1f}%) - ETA: {remaining/60:.1f} min")
        
        term = term_entry.get('term', '').strip()
        
        if not term:
            # Skip empty terms
            non_dictionary_terms.append({
                **term_entry,
                "dictionary_status": "empty_term",
                "dictionary_check": {"error": "Empty term"}
            })
            continue
        
        # Check if term exists in dictionary
        exists, status, response_data = check_term_in_dictionary(tool, term)
        
        # Create enhanced term entry with dictionary information
        enhanced_term = {
            **term_entry,
            "dictionary_status": "found" if exists else "not_found",
            "dictionary_check": {
                "status": status,
                "checked_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "response_summary": {
                    "has_meaning": exists,
                    "sources_checked": ["educalingo", "merriam_webster"] if not exists else (["educalingo"] if "Educalingo" in status else ["merriam_webster"])
                }
            }
        }
        
        # Add meaning preview if found
        if exists and 'meaning' in response_data:
            meaning = response_data['meaning']
            if isinstance(meaning, (list, tuple)) and len(meaning) > 1:
                enhanced_term["dictionary_check"]["meaning_preview"] = str(meaning[1])[:200] + "..." if len(str(meaning[1])) > 200 else str(meaning[1])
            elif isinstance(meaning, dict):
                # Handle Merriam-Webster format
                first_key = next(iter(meaning.keys())) if meaning else ""
                if first_key and meaning[first_key]:
                    enhanced_term["dictionary_check"]["meaning_preview"] = str(meaning[first_key][0])[:200] + "..." if len(str(meaning[first_key][0])) > 200 else str(meaning[first_key][0])
        
        # Categorize the term
        if exists:
            dictionary_terms.append(enhanced_term)
        else:
            non_dictionary_terms.append(enhanced_term)
        
        # Small delay to avoid overwhelming the dictionary service
        time.sleep(0.1)
    
    elapsed_total = time.time() - start_time
    print(f"\n✅ Analysis completed in {elapsed_total/60:.1f} minutes")
    print(f"📊 Results:")
    print(f"   Dictionary terms: {len(dictionary_terms):,} ({len(dictionary_terms)/total_terms*100:.1f}%)")
    print(f"   Non-dictionary terms: {len(non_dictionary_terms):,} ({len(non_dictionary_terms)/total_terms*100:.1f}%)")
    
    return dictionary_terms, non_dictionary_terms

def create_split_files(dictionary_terms: List[Dict], non_dictionary_terms: List[Dict], 
                      original_metadata: Dict) -> bool:
    """Create separate files for dictionary and non-dictionary terms"""
    print("\n💾 Creating split files...")
    
    try:
        # Dictionary terms file
        dict_file = "Dictionary_Terms_Found.json"
        dict_data = {
            "metadata": {
                **original_metadata,
                "file_type": "dictionary_terms",
                "description": "Terms found in English dictionaries (Educalingo, Merriam-Webster)",
                "total_terms": len(dictionary_terms),
                "dictionary_sources_used": ["educalingo", "merriam_webster"],
                "split_date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "dictionary_terms": dictionary_terms
        }
        
        with open(dict_file, 'w', encoding='utf-8') as f:
            json.dump(dict_data, f, indent=2, ensure_ascii=False)
        
        dict_size_mb = os.path.getsize(dict_file) / (1024 * 1024)
        print(f"✅ Dictionary terms file created: {dict_file} ({dict_size_mb:.1f} MB)")
        
        # Non-dictionary terms file
        non_dict_file = "Non_Dictionary_Terms.json"
        non_dict_data = {
            "metadata": {
                **original_metadata,
                "file_type": "non_dictionary_terms",
                "description": "Terms NOT found in English dictionaries - potential technical terms, brand names, or domain-specific vocabulary",
                "total_terms": len(non_dictionary_terms),
                "dictionary_sources_checked": ["educalingo", "merriam_webster"],
                "split_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "analysis_note": "These terms may be: technical terminology, brand names, product-specific terms, abbreviations, or emerging vocabulary"
            },
            "non_dictionary_terms": non_dictionary_terms
        }
        
        with open(non_dict_file, 'w', encoding='utf-8') as f:
            json.dump(non_dict_data, f, indent=2, ensure_ascii=False)
        
        non_dict_size_mb = os.path.getsize(non_dict_file) / (1024 * 1024)
        print(f"✅ Non-dictionary terms file created: {non_dict_file} ({non_dict_size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating split files: {e}")
        return False

def show_analysis_summary(dictionary_terms: List[Dict], non_dictionary_terms: List[Dict]):
    """Show detailed analysis summary"""
    print(f"\n{'='*70}")
    print("📊 DETAILED ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    total_terms = len(dictionary_terms) + len(non_dictionary_terms)
    
    print(f"📈 Overall Statistics:")
    print(f"   Total terms analyzed: {total_terms:,}")
    print(f"   Dictionary terms: {len(dictionary_terms):,} ({len(dictionary_terms)/total_terms*100:.1f}%)")
    print(f"   Non-dictionary terms: {len(non_dictionary_terms):,} ({len(non_dictionary_terms)/total_terms*100:.1f}%)")
    
    # Show top dictionary terms by frequency
    if dictionary_terms:
        print(f"\n🔤 Top Dictionary Terms (by frequency):")
        top_dict_terms = sorted(dictionary_terms, key=lambda x: x['frequency'], reverse=True)[:10]
        for i, term in enumerate(top_dict_terms, 1):
            print(f"   {i:2}. '{term['term']}' - {term['frequency']:,}x")
    
    # Show top non-dictionary terms by frequency
    if non_dictionary_terms:
        print(f"\n🔧 Top Non-Dictionary Terms (by frequency):")
        top_non_dict_terms = sorted(non_dictionary_terms, key=lambda x: x['frequency'], reverse=True)[:10]
        for i, term in enumerate(top_non_dict_terms, 1):
            print(f"   {i:2}. '{term['term']}' - {term['frequency']:,}x")
    
    # Frequency distribution analysis
    dict_high_freq = sum(1 for t in dictionary_terms if t['frequency'] >= 20)
    non_dict_high_freq = sum(1 for t in non_dictionary_terms if t['frequency'] >= 20)
    
    print(f"\n📊 High-Frequency Terms (≥20 occurrences):")
    print(f"   Dictionary terms: {dict_high_freq:,}")
    print(f"   Non-dictionary terms: {non_dict_high_freq:,}")
    
    print(f"\n💡 Insights:")
    if len(non_dictionary_terms) > len(dictionary_terms):
        print(f"   • Most terms ({len(non_dictionary_terms):,}) are NOT in standard dictionaries")
        print(f"   • This suggests many technical/domain-specific terms in your data")
        print(f"   • Non-dictionary terms may need specialized glossaries")
    else:
        print(f"   • Most terms ({len(dictionary_terms):,}) are in standard dictionaries")
        print(f"   • Good coverage of standard vocabulary")
    
    if non_dict_high_freq > 0:
        print(f"   • {non_dict_high_freq} high-frequency non-dictionary terms need attention")
        print(f"   • These are likely important technical terms for your domain")

def main():
    """Main function"""
    print("📚 DICTIONARY TERMS SPLITTER")
    print("=" * 60)
    
    input_file = "New_Terms_Candidates_Clean.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Load new terms candidates
    new_terms_data = load_new_terms_candidates(input_file)
    if not new_terms_data:
        print("❌ Failed to load new terms data")
        return
    
    # Analyze terms with dictionary
    print(f"\n⚠️  Warning: This process will check {new_terms_data['metadata']['total_new_terms']:,} terms")
    print("   against dictionary sources. This may take 1-2 hours to complete.")
    print("   You can interrupt with Ctrl+C and results will be saved for completed terms.")
    
    response = input("\nProceed with analysis? (y/n): ").strip().lower()
    if response != 'y':
        print("Analysis cancelled by user")
        return
    
    try:
        dictionary_terms, non_dictionary_terms = analyze_terms_with_dictionary(new_terms_data)
        
        # Create split files
        success = create_split_files(dictionary_terms, non_dictionary_terms, new_terms_data['metadata'])
        
        if success:
            # Show analysis summary
            show_analysis_summary(dictionary_terms, non_dictionary_terms)
            
            print(f"\n🎉 Process completed successfully!")
            print(f"✅ Files created:")
            print(f"   • Dictionary_Terms_Found.json - {len(dictionary_terms):,} terms found in dictionaries")
            print(f"   • Non_Dictionary_Terms.json - {len(non_dictionary_terms):,} terms NOT in dictionaries")
            print(f"\n💡 Next steps:")
            print(f"   1. Review high-frequency non-dictionary terms for glossary addition")
            print(f"   2. Consider domain-specific dictionaries for technical terms")
            print(f"   3. Validate important terms manually for accuracy")
        else:
            print(f"\n❌ Failed to create split files")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        print("Partial results may be available")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


