#!/usr/bin/env python3
"""
Validate term candidates from dictionary classification results.
Processes both dictionary and non-dictionary terms separately.
"""

import os
import json
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any
from terminology_review_agent import TerminologyReviewAgent

def load_classified_terms(file_path: str, term_type: str) -> List[Dict[str, Any]]:
    """Load terms from classified dictionary results"""
    print(f"📂 Loading {term_type} terms from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the appropriate term list based on type
        if term_type == "dictionary":
            terms_data = data.get('dictionary_terms', [])
        else:  # non_dictionary
            terms_data = data.get('non_dictionary_terms', [])
        
        total_terms = len(terms_data)
        print(f"✅ Found {total_terms} {term_type} terms in the file")
        
        return terms_data
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return []
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []

def filter_and_prioritize_terms(terms_data: List[Dict], min_frequency: int = 5, 
                               limit: int = None) -> List[str]:
    """Filter and prioritize terms based on frequency and technical relevance"""
    
    print(f"🔍 Filtering terms with criteria:")
    print(f"   • Minimum frequency: {min_frequency}")
    print(f"   • Exclude single characters: True")
    print(f"   • Exclude common words: True")
    
    # Common words to exclude
    common_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
        'after', 'above', 'below', 'between', 'among', 'a', 'an', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
        'their', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall', 'very', 'too', 'so',
        'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how',
        'what', 'which', 'who', 'whom', 'whose', 'all', 'any', 'some', 'no',
        'not', 'only', 'own', 'same', 'such', 'than', 'as', 'if', 'because'
    }
    
    filtered_terms = []
    
    for term_data in terms_data:
        term = term_data.get('term', '').strip().lower()
        frequency = term_data.get('frequency', 0)
        
        # Skip if below frequency threshold
        if frequency < min_frequency:
            continue
            
        # Skip single characters
        if len(term) <= 1:
            continue
            
        # Skip common words
        if term in common_words:
            continue
            
        # Skip purely numeric terms
        if term.replace('.', '').replace('-', '').isdigit():
            continue
            
        filtered_terms.append({
            'term': term_data.get('term', ''),  # Keep original case
            'frequency': frequency,
            'original_texts': term_data.get('original_texts', [])
        })
    
    print(f"✅ Filtered to {len(filtered_terms)} terms from {len(terms_data)} original terms")
    
    # Sort by frequency (descending) and then by technical relevance
    print(f"🎯 Prioritizing terms based on frequency and technical relevance...")
    
    # Technical keywords that indicate CAD/AEC/Manufacturing relevance
    technical_keywords = {
        'autodesk', 'cad', 'bim', '3d', 'design', 'model', 'drawing', 'dwg', 
        'civil', 'architecture', 'engineering', 'manufacturing', 'construction',
        'inventor', 'fusion', 'revit', 'autocad', 'maya', 'plant', 'mechanical',
        'structural', 'electrical', 'plumbing', 'hvac', 'analysis', 'simulation',
        'rendering', 'visualization', 'collaboration', 'cloud', 'file', 'layer',
        'dimension', 'annotation', 'viewport', 'layout', 'block', 'component',
        'assembly', 'part', 'feature', 'sketch', 'extrude', 'revolve', 'sweep',
        'loft', 'fillet', 'chamfer', 'pattern', 'mirror', 'array', 'constraint',
        'parameter', 'variable', 'expression', 'formula', 'material', 'property',
        'attribute', 'metadata', 'standard', 'specification', 'tolerance',
        'precision', 'accuracy', 'quality', 'validation', 'verification'
    }
    
    def calculate_technical_score(term_info):
        term = term_info['term'].lower()
        score = term_info['frequency']  # Base score from frequency
        
        # Boost for technical keywords
        for keyword in technical_keywords:
            if keyword in term:
                score += 1000  # Significant boost for technical terms
                break
        
        return score
    
    # Sort by technical score
    filtered_terms.sort(key=calculate_technical_score, reverse=True)
    
    # Apply limit if specified
    if limit and limit < len(filtered_terms):
        filtered_terms = filtered_terms[:limit]
        print(f"🔢 Limited to first {limit} terms")
    
    print(f"✅ Terms prioritized by technical relevance")
    
    # Return just the term strings and original texts
    result = []
    for term_info in filtered_terms:
        result.append(term_info['term'])
    
    return result, filtered_terms  # Return both for original_texts access

def get_already_processed_terms(file_prefix: str) -> set:
    """Get terms that have already been processed from existing batch files"""
    processed_terms = set()
    
    batch_files = glob.glob(f"{file_prefix}_batch_*.json")
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            for result in results:
                term = result.get('term', '').strip()
                if term:
                    processed_terms.add(term)
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing batch file {batch_file}: {e}")
    
    return processed_terms

def validate_terms_batch(agent: TerminologyReviewAgent, terms: List[str], 
                        term_details: List[Dict], batch_size: int = 20, 
                        industry_context: str = "General", src_lang: str = "EN", 
                        tgt_lang: str = None, file_prefix: str = "validation") -> List[Dict]:
    """Validate terms in batches with original texts"""
    
    # Check for already processed terms
    processed_terms = get_already_processed_terms(file_prefix)
    
    if processed_terms:
        print(f"📋 Found {len(processed_terms)} already processed terms")
        print(f"🔄 Resuming from where we left off...")
        
        # Filter out already processed terms
        remaining_terms = []
        remaining_details = []
        
        for i, term in enumerate(terms):
            if term not in processed_terms:
                remaining_terms.append(term)
                if i < len(term_details):
                    remaining_details.append(term_details[i])
        
        terms = remaining_terms
        term_details = remaining_details
        
        print(f"✅ Filtered to {len(terms)} remaining terms to process")
    
    if not terms:
        print(f"🎉 All terms have already been processed!")
        return []
    
    print(f"🔄 Validating {len(terms)} terms in batches of {batch_size}")
    
    all_results = []
    
    # Determine starting batch number
    existing_batches = glob.glob(f"{file_prefix}_batch_*.json")
    last_batch_num = 0
    for batch_file in existing_batches:
        try:
            # Extract batch number from filename
            import re
            match = re.search(rf"{file_prefix}_batch_(\d+)_", batch_file)
            if match:
                batch_num = int(match.group(1))
                last_batch_num = max(last_batch_num, batch_num)
        except:
            continue
    
    for i in range(0, len(terms), batch_size):
        batch = terms[i:i + batch_size]
        batch_details = term_details[i:i + batch_size]
        batch_num = last_batch_num + (i // batch_size) + 1  # Continue from last batch
        total_batches = (len(terms) + batch_size - 1) // batch_size
        
        print(f"\n📊 Processing batch {batch_num}/{last_batch_num + total_batches}: {batch}")
        
        try:
            # Create output filename for this batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{file_prefix}_batch_{batch_num}_{timestamp}.json"
            
            # Extract original texts for each term in batch
            batch_original_texts = []
            for term_detail in batch_details:
                original_texts = term_detail.get('original_texts', [])
                batch_original_texts.append(original_texts)
            
            # Validate the batch with original texts
            result = agent.batch_validate_terms(
                terms=batch,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                industry_context=industry_context,
                save_to_file=output_file,
                original_texts=batch_original_texts  # Pass original texts
            )
            
            # Parse the JSON result to extract individual term results
            try:
                result_data = json.loads(result) if isinstance(result, str) else result
                if isinstance(result_data, dict) and 'results' in result_data:
                    batch_results = result_data['results']
                    all_results.extend(batch_results)
                    print(f"✅ Batch {batch_num} completed: {len(batch_results)} terms validated")
                else:
                    print(f"⚠️ Unexpected result format for batch {batch_num}")
                    
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse result for batch {batch_num}")
                
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            continue
    
    return all_results

def generate_summary_report(results: List[Dict], output_file: str, term_type: str):
    """Generate a summary report of validation results"""
    
    if not results:
        print("⚠️ No results to summarize")
        return
    
    # Calculate statistics
    total_terms = len(results)
    status_counts = {}
    score_distribution = {"high": 0, "medium": 0, "low": 0}
    
    for result in results:
        status = result.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
        
        score = result.get('validation_score', 0)
        if score >= 0.7:
            score_distribution["high"] += 1
        elif score >= 0.4:
            score_distribution["medium"] += 1
        else:
            score_distribution["low"] += 1
    
    # Create summary
    summary = {
        "metadata": {
            "term_type": term_type,
            "total_terms_processed": total_terms,
            "timestamp": datetime.now().isoformat(),
            "summary_file": output_file
        },
        "statistics": {
            "status_breakdown": status_counts,
            "score_distribution": score_distribution,
            "recommendations": {
                "high_priority": sum(1 for r in results if r.get('validation_score', 0) >= 0.7),
                "needs_review": sum(1 for r in results if 0.4 <= r.get('validation_score', 0) < 0.7),
                "low_priority": sum(1 for r in results if r.get('validation_score', 0) < 0.4)
            }
        },
        "sample_results": results[:10]  # First 10 results as samples
    }
    
    # Save summary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 VALIDATION SUMMARY ({term_type.upper()} TERMS)")
    print("=" * 60)
    print(f"✅ Total terms processed: {total_terms}")
    print(f"📈 High priority (≥0.7): {score_distribution['high']}")
    print(f"📋 Needs review (0.4-0.7): {score_distribution['medium']}")
    print(f"📉 Low priority (<0.4): {score_distribution['low']}")
    print(f"💾 Summary saved to: {output_file}")

def main():
    """Main function to process both dictionary and non-dictionary terms"""
    
    parser = argparse.ArgumentParser(description='Validate dictionary-classified term candidates')
    parser.add_argument('--dictionary-file', 
                       default='Fast_Dictionary_Terms_20250903_123659.json',
                       help='Dictionary terms JSON file')
    parser.add_argument('--non-dictionary-file', 
                       default='Fast_Non_Dictionary_Terms_20250903_123659.json',
                       help='Non-dictionary terms JSON file')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of terms to process per file (default: None - process all)')
    parser.add_argument('--min-frequency', type=int, default=5, 
                       help='Minimum frequency threshold (default: 5)')
    parser.add_argument('--batch-size', type=int, default=20, 
                       help='Batch size for processing (default: 20)')
    parser.add_argument('--industry', default='General', 
                       choices=['CAD', 'AEC', 'Manufacturing', 'General'],
                       help='Industry context (default: General)')
    parser.add_argument('--src-lang', default='EN', help='Source language (default: EN)')
    parser.add_argument('--model', default='gpt-4.1', choices=['gpt-5', 'gpt-4.1'],
                       help='Model to use (default: gpt-4.1)')
    parser.add_argument('--process-dictionary', action='store_true', default=True,
                       help='Process dictionary terms (default: True)')
    parser.add_argument('--process-non-dictionary', action='store_true', default=True,
                       help='Process non-dictionary terms (default: True)')
    
    args = parser.parse_args()
    
    print("🌟 DICTIONARY CLASSIFICATION VALIDATION")
    print("=" * 60)
    print(f"📁 Dictionary file: {args.dictionary_file}")
    print(f"📁 Non-dictionary file: {args.non_dictionary_file}")
    print(f"🔢 Processing limit per file: {'ALL' if args.limit is None else args.limit} terms")
    print(f"📊 Minimum frequency: {args.min_frequency}")
    print(f"🏭 Industry context: {args.industry}")
    print(f"🌐 Language: {args.src_lang} (terminology validation only)")
    print(f"🤖 Model: {args.model}")
    print()
    
    # Initialize the terminology review agent
    glossary_folder = "glossary"
    if not os.path.exists(glossary_folder):
        print(f"❌ Glossary folder not found: {glossary_folder}")
        return
    
    print(f"🤖 Initializing Terminology Review Agent...")
    try:
        agent = TerminologyReviewAgent(glossary_folder, model_name=args.model)
        print(f"✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Process dictionary terms
    if args.process_dictionary and os.path.exists(args.dictionary_file):
        print(f"\n🔍 PROCESSING DICTIONARY TERMS")
        print("=" * 60)
        
        dictionary_terms_data = load_classified_terms(args.dictionary_file, "dictionary")
        if dictionary_terms_data:
            dict_terms, dict_details = filter_and_prioritize_terms(
                dictionary_terms_data, args.min_frequency, args.limit
            )
            
            if dict_terms:
                print(f"🎯 Top dictionary terms selected for validation:")
                for i, term in enumerate(dict_terms[:10], 1):
                    print(f"   {i}. {term}")
                if len(dict_terms) > 10:
                    print(f"   ... and {len(dict_terms) - 10} more")
                
                print(f"\n🚀 Starting dictionary terms validation...")
                dict_results = validate_terms_batch(
                    agent=agent,
                    terms=dict_terms,
                    term_details=dict_details,
                    batch_size=args.batch_size,
                    industry_context=args.industry,
                    src_lang=args.src_lang,
                    tgt_lang=None,
                    file_prefix="dictionary_validation"
                )
                
                # Generate summary report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dict_summary_file = f"dictionary_terms_summary_{timestamp}.json"
                generate_summary_report(dict_results, dict_summary_file, "dictionary")
    
    # Process non-dictionary terms
    if args.process_non_dictionary and os.path.exists(args.non_dictionary_file):
        print(f"\n🔍 PROCESSING NON-DICTIONARY TERMS")
        print("=" * 60)
        
        non_dict_terms_data = load_classified_terms(args.non_dictionary_file, "non_dictionary")
        if non_dict_terms_data:
            non_dict_terms, non_dict_details = filter_and_prioritize_terms(
                non_dict_terms_data, args.min_frequency, args.limit
            )
            
            if non_dict_terms:
                print(f"🎯 Top non-dictionary terms selected for validation:")
                for i, term in enumerate(non_dict_terms[:10], 1):
                    print(f"   {i}. {term}")
                if len(non_dict_terms) > 10:
                    print(f"   ... and {len(non_dict_terms) - 10} more")
                
                print(f"\n🚀 Starting non-dictionary terms validation...")
                non_dict_results = validate_terms_batch(
                    agent=agent,
                    terms=non_dict_terms,
                    term_details=non_dict_details,
                    batch_size=args.batch_size,
                    industry_context=args.industry,
                    src_lang=args.src_lang,
                    tgt_lang=None,
                    file_prefix="non_dictionary_validation"
                )
                
                # Generate summary report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                non_dict_summary_file = f"non_dictionary_terms_summary_{timestamp}.json"
                generate_summary_report(non_dict_results, non_dict_summary_file, "non_dictionary")
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print("Check the generated batch files and summary reports for detailed results.")

if __name__ == "__main__":
    main()
