#!/usr/bin/env python3
"""
Script to validate new term candidates from New_Terms_Candidates_Clean.json
using the Terminology Review Agent with web search capabilities.

This script processes the large JSON file of term candidates and validates them
for Autodesk terminology inclusion using comprehensive analysis.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse

from terminology_review_agent import TerminologyReviewAgent


def load_term_candidates(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load term candidates from the JSON file"""
    print(f"📂 Loading term candidates from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_terms = data.get('new_terms', [])
        total_terms = len(new_terms)
        
        print(f"✅ Found {total_terms} term candidates in the file")
        
        if limit and limit < total_terms:
            new_terms = new_terms[:limit]
            print(f"🔢 Limited processing to first {limit} terms")
        
        return new_terms
        
    except Exception as e:
        print(f"❌ Error loading term candidates: {e}")
        return []


def filter_terms_by_criteria(terms: List[Dict], min_frequency: int = 5, 
                           exclude_single_chars: bool = True,
                           exclude_common_words: bool = True) -> List[Dict]:
    """Filter terms based on various criteria to focus on meaningful candidates"""
    
    print(f"🔍 Filtering terms with criteria:")
    print(f"   • Minimum frequency: {min_frequency}")
    print(f"   • Exclude single characters: {exclude_single_chars}")
    print(f"   • Exclude common words: {exclude_common_words}")
    
    # Common words to exclude (not technical terms)
    common_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
        'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'a', 'an', 'all', 'any',
        'some', 'many', 'much', 'more', 'most', 'other', 'another', 'such',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
        'here', 'there', 'where', 'when', 'why', 'how', 'what', 'which',
        'who', 'whom', 'whose', 'if', 'unless', 'until', 'while', 'although',
        'because', 'since', 'as', 'create', 'access', 'folder', 'tools',
        'information', 'objects', 'tab', 'type', 'error', 'available',
        'display', 'cloud', 'account', 'add', 'selected', 'number', 'change'
    }
    
    filtered_terms = []
    
    for term_data in terms:
        term = term_data.get('term', '').lower().strip()
        frequency = term_data.get('frequency', 0)
        
        # Skip if below minimum frequency
        if frequency < min_frequency:
            continue
            
        # Skip single characters
        if exclude_single_chars and len(term) <= 1:
            continue
            
        # Skip common non-technical words
        if exclude_common_words and term in common_words:
            continue
            
        # Skip terms that are clearly not technical (basic English words)
        if exclude_common_words and term in ['you', 'your', 'we', 'our', 'they', 'their']:
            continue
            
        filtered_terms.append(term_data)
    
    print(f"✅ Filtered to {len(filtered_terms)} terms from {len(terms)} original terms")
    return filtered_terms


def prioritize_terms(terms: List[Dict]) -> List[Dict]:
    """Prioritize terms based on technical relevance indicators"""
    
    print("🎯 Prioritizing terms based on technical relevance...")
    
    # Technical indicators that suggest a term might be valuable
    technical_indicators = [
        'cad', 'bim', '3d', '2d', 'dwg', 'dxf', 'revit', 'autocad', 'inventor',
        'fusion', 'maya', 'max', 'civil', 'plant', 'mechanical', 'electrical',
        'structural', 'architecture', 'engineering', 'design', 'model', 'modeling',
        'render', 'animation', 'simulation', 'analysis', 'drawing', 'sketch',
        'mesh', 'surface', 'solid', 'parametric', 'constraint', 'assembly',
        'component', 'part', 'feature', 'geometry', 'coordinate', 'dimension',
        'annotation', 'layer', 'block', 'symbol', 'library', 'template',
        'workflow', 'collaboration', 'cloud', 'api', 'plugin', 'addon',
        'extension', 'tool', 'command', 'function', 'property', 'attribute',
        'material', 'texture', 'lighting', 'camera', 'viewport', 'workspace'
    ]
    
    def calculate_priority_score(term_data: Dict) -> float:
        term = term_data.get('term', '').lower()
        frequency = term_data.get('frequency', 0)
        pos_tags = term_data.get('pos_tag_variations', {}).get('tags', [])
        
        score = 0.0
        
        # Base score from frequency (normalized)
        score += min(frequency / 100, 5.0)
        
        # Bonus for technical indicators
        for indicator in technical_indicators:
            if indicator in term:
                score += 2.0
                break
        
        # Bonus for technical POS tags
        technical_pos_tags = ['NOUN', 'PROPN', 'ADJ']
        for tag in pos_tags:
            if any(tech_tag in tag for tech_tag in technical_pos_tags):
                score += 1.0
                break
        
        # Bonus for compound terms (likely technical)
        if '_' in term or '-' in term or ' ' in term:
            score += 1.5
        
        # Penalty for very common terms
        if frequency > 1000:
            score -= 1.0
            
        return score
    
    # Calculate scores and sort
    for term_data in terms:
        term_data['priority_score'] = calculate_priority_score(term_data)
    
    # Sort by priority score (descending)
    prioritized_terms = sorted(terms, key=lambda x: x['priority_score'], reverse=True)
    
    print(f"✅ Terms prioritized by technical relevance")
    return prioritized_terms


def validate_terms_batch(agent: TerminologyReviewAgent, terms: List[str], 
                        batch_size: int = 10, industry_context: str = "General",
                        src_lang: str = "EN", tgt_lang: str = None) -> List[Dict]:
    """Validate terms in batches to manage API rate limits"""
    
    print(f"🔄 Validating {len(terms)} terms in batches of {batch_size}")
    
    all_results = []
    
    for i in range(0, len(terms), batch_size):
        batch = terms[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(terms) + batch_size - 1) // batch_size
        
        print(f"\n📊 Processing batch {batch_num}/{total_batches}: {batch}")
        
        try:
            # Create output filename for this batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"term_validation_batch_{batch_num}_{timestamp}.json"
            
            # Validate the batch
            result = agent.batch_validate_terms(
                terms=batch,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                industry_context=industry_context,
                save_to_file=output_file
            )
            
            # Parse the JSON result to extract individual term results
            try:
                result_data = json.loads(result) if isinstance(result, str) else result
                if 'results' in result_data:
                    all_results.extend(result_data['results'])
                else:
                    all_results.append(result_data)
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse result for batch {batch_num}")
                all_results.append({"batch": batch_num, "error": "Could not parse result"})
            
            print(f"✅ Batch {batch_num} completed")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            all_results.append({"batch": batch_num, "terms": batch, "error": str(e)})
    
    return all_results


def generate_summary_report(results: List[Dict], output_file: str):
    """Generate a summary report of validation results"""
    
    print(f"📋 Generating summary report...")
    
    # Analyze results
    total_terms = len(results)
    recommended_terms = []
    needs_review_terms = []
    not_recommended_terms = []
    error_terms = []
    
    for result in results:
        if 'error' in result:
            error_terms.append(result)
        elif 'status' in result:
            status = result.get('status', 'unknown')
            if status == 'recommended':
                recommended_terms.append(result)
            elif status == 'needs_review':
                needs_review_terms.append(result)
            elif status == 'not_recommended':
                not_recommended_terms.append(result)
    
    # Create summary
    summary = {
        "validation_summary": {
            "timestamp": datetime.now().isoformat(),
            "total_terms_processed": total_terms,
            "recommended_count": len(recommended_terms),
            "needs_review_count": len(needs_review_terms),
            "not_recommended_count": len(not_recommended_terms),
            "error_count": len(error_terms)
        },
        "recommended_terms": [
            {
                "term": r.get('term', 'unknown'),
                "score": r.get('validation_score', 0),
                "key_reasons": r.get('recommendations', [])[:3]  # Top 3 reasons
            }
            for r in recommended_terms[:20]  # Top 20 recommended
        ],
        "needs_review_terms": [
            {
                "term": r.get('term', 'unknown'),
                "score": r.get('validation_score', 0),
                "key_reasons": r.get('recommendations', [])[:2]
            }
            for r in needs_review_terms[:10]  # Top 10 needing review
        ],
        "detailed_results": results
    }
    
    # Save summary report
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Summary report saved to: {output_file}")
        print(f"📊 Results summary:")
        print(f"   • ✅ Recommended: {len(recommended_terms)} terms")
        print(f"   • ⚠️ Needs Review: {len(needs_review_terms)} terms")
        print(f"   • ❌ Not Recommended: {len(not_recommended_terms)} terms")
        print(f"   • ⚡ Errors: {len(error_terms)} terms")
        
    except Exception as e:
        print(f"❌ Error saving summary report: {e}")


def main():
    """Main function to process term candidates"""
    
    parser = argparse.ArgumentParser(description='Validate new term candidates using Terminology Review Agent')
    parser.add_argument('--input-file', default='New_Terms_Candidates_Clean.json', 
                       help='Input JSON file with term candidates')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of terms to process (default: None - process all terms)')
    parser.add_argument('--min-frequency', type=int, default=10, 
                       help='Minimum frequency threshold (default: 10)')
    parser.add_argument('--batch-size', type=int, default=20, 
                       help='Batch size for processing (default: 20)')
    parser.add_argument('--industry', default='General', 
                       choices=['CAD', 'AEC', 'Manufacturing', 'General'],
                       help='Industry context (default: General)')
    parser.add_argument('--src-lang', default='EN', help='Source language (default: EN)')
    parser.add_argument('--tgt-lang', default=None, help='Target language (optional, for translation context only)')
    parser.add_argument('--model', default='gpt-4.1', choices=['gpt-5', 'gpt-4.1'],
                       help='Model to use (default: gpt-4.1 for speed)')
    
    args = parser.parse_args()
    
    print("🌟 NEW TERM CANDIDATES VALIDATION")
    print("=" * 60)
    print(f"📁 Input file: {args.input_file}")
    print(f"🔢 Processing limit: {'ALL' if args.limit is None else args.limit} terms")
    print(f"📊 Minimum frequency: {args.min_frequency}")
    print(f"🏭 Industry context: {args.industry}")
    print(f"🌐 Language: {args.src_lang}" + (f" → {args.tgt_lang}" if args.tgt_lang else " (terminology validation only)"))
    print(f"🤖 Model: {args.model}")
    print()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    # Load term candidates
    term_candidates = load_term_candidates(args.input_file, args.limit)
    if not term_candidates:
        print("❌ No term candidates loaded. Exiting.")
        return
    
    # Filter terms
    filtered_terms = filter_terms_by_criteria(
        term_candidates, 
        min_frequency=args.min_frequency
    )
    
    if not filtered_terms:
        print("❌ No terms remain after filtering. Try lowering --min-frequency.")
        return
    
    # Prioritize terms
    prioritized_terms = prioritize_terms(filtered_terms)
    
    # Extract just the term strings for validation
    terms_to_validate = [term_data['term'] for term_data in prioritized_terms]
    
    print(f"\n🎯 Top terms selected for validation:")
    for i, term in enumerate(terms_to_validate[:10]):
        print(f"   {i+1}. {term}")
    if len(terms_to_validate) > 10:
        print(f"   ... and {len(terms_to_validate) - 10} more")
    
    # Initialize the terminology review agent
    print(f"\n🤖 Initializing Terminology Review Agent...")
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    
    try:
        agent = TerminologyReviewAgent(glossary_folder, model_name=args.model)
        
        # Validate terms in batches
        print(f"\n🚀 Starting validation process...")
        validation_results = validate_terms_batch(
            agent=agent,
            terms=terms_to_validate,
            batch_size=args.batch_size,
            industry_context=args.industry,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        # Generate summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"term_validation_summary_{timestamp}.json"
        generate_summary_report(validation_results, summary_file)
        
        print(f"\n🎉 Validation process completed!")
        print(f"📄 Summary report: {summary_file}")
        print(f"📁 Individual batch files also created")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
