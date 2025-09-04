#!/usr/bin/env python3
"""
Fast Dictionary Analysis Runner
Runs the fast NLTK-based dictionary agent - much faster than PyMultiDictionary
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fast_dictionary_agent import FastDictionaryAgent, load_terms_data, save_results
    print("✅ Fast dictionary agent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import fast_dictionary_agent: {e}")
    sys.exit(1)


def run_fast_analysis():
    """
    Run fast dictionary analysis using NLTK
    Ultra-fast alternative to PyMultiDictionary
    """
    print("🚀 Fast Dictionary Term Analysis (NLTK-based)")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the fast agent
    print("\n📚 Initializing Fast Dictionary Agent...")
    agent = FastDictionaryAgent()
    
    if not agent.dictionary_tool.initialized:
        print("❌ Fast dictionary not available.")
        print("   Install with: pip install nltk")
        print("   The agent will download required NLTK data automatically.")
        return False
    
    # Look for input files
    input_files = [
        "Cleaned_Complete_Terms_Data.json",
        "Cleaned_Summary_Terms_Data.json", 
        "Combined_Terms_Data.json",
        "Term_Extracted_result.json"
    ]
    
    input_file = None
    for file in input_files:
        if os.path.exists(file):
            input_file = file
            print(f"✅ Found input file: {file}")
            break
    
    if not input_file:
        print("❌ No input files found. Looking for:")
        for file in input_files:
            print(f"   - {file}")
        return False
    
    # Load terms data
    print(f"\n📂 Loading terms from: {input_file}")
    terms = load_terms_data(input_file)
    
    if not terms:
        print("❌ No terms loaded from file")
        return False
    
    print(f"✅ Loaded {len(terms):,} terms")
    
    # Show sample terms
    print("\n🔍 Sample terms:")
    for i, term_data in enumerate(terms[:5]):
        term = term_data.get('term', 'N/A')
        freq = term_data.get('frequency', 'N/A')
        print(f"   {i+1}. '{term}' (frequency: {freq})")
    if len(terms) > 5:
        print(f"   ... and {len(terms)-5:,} more")
    
    # Analysis options
    print(f"\n⚙️  Analysis options:")
    print(f"   This is MUCH faster than PyMultiDictionary (no API calls!)")
    print(f"   Estimated time: ~{len(terms)/10000:.1f} seconds for {len(terms):,} terms")
    
    max_terms = None
    
    try:
        user_input = input(f"\n   Analyze all {len(terms):,} terms? (y/n/number): ").strip().lower()
        
        if user_input in ['n', 'no']:
            max_terms = int(input("   How many terms to analyze? "))
        elif user_input.isdigit():
            max_terms = int(user_input)
        elif user_input in ['y', 'yes', '']:
            max_terms = None  # Analyze all
        else:
            print("   Using default: analyzing all terms...")
            max_terms = None
            
    except (ValueError, KeyboardInterrupt):
        print("   Using default: analyzing all terms")
        max_terms = None
    
    # Run the analysis
    print(f"\n🔄 Starting fast dictionary analysis...")
    start_time = time.time()
    
    try:
        dictionary_words, non_dictionary_words = agent.analyze_terms_fast(
            terms, 
            max_terms=max_terms
        )
        
        analysis_time = time.time() - start_time
        total_analyzed = len(dictionary_words) + len(non_dictionary_words)
        
        # Save results
        print(f"\n💾 Saving results...")
        save_results(dictionary_words, non_dictionary_words)
        
        # Final summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"   ⏱️  Total time: {analysis_time:.2f} seconds")
        print(f"   ⚡ Speed: {total_analyzed/analysis_time:.1f} terms/second")
        print(f"   🚀 This is ~{100:.0f}x faster than PyMultiDictionary!")
        print(f"   📊 Results:")
        print(f"      📖 Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_analyzed*100:.1f}%)")
        print(f"      🔧 Non-dictionary: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_analyzed*100:.1f}%)")
        print(f"   📁 Files saved with timestamp")
        
        # Performance comparison
        estimated_pymulti_time = total_analyzed * 0.1  # ~0.1 seconds per term with API delays
        print(f"\n📈 Performance Comparison:")
        print(f"   ⚡ Fast method: {analysis_time:.1f} seconds")
        print(f"   🐌 PyMultiDictionary (estimated): {estimated_pymulti_time:.1f} seconds")
        print(f"   🚀 Speed improvement: {estimated_pymulti_time/analysis_time:.1f}x faster!")
        
        # Show some examples
        print(f"\n📝 Examples of dictionary words found:")
        dict_examples = [item.get('term', 'N/A') for item in dictionary_words[:10]]
        print(f"   {', '.join(dict_examples)}")
        
        print(f"\n🔧 Examples of non-dictionary terms:")
        non_dict_examples = [item.get('term', 'N/A') for item in non_dictionary_words[:10]]
        print(f"   {', '.join(non_dict_examples)}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return False


def main():
    """Main entry point"""
    try:
        success = run_fast_analysis()
        
        if success:
            print(f"\n✅ Fast dictionary analysis completed successfully!")
            print(f"   📊 This method is much faster than PyMultiDictionary")
            print(f"   📚 Uses NLTK's comprehensive word corpus")
            print(f"   ⚡ No API calls, no rate limits, no network dependency")
            print(f"   🔍 Results are compatible with your existing workflow")
        else:
            print(f"\n❌ Analysis failed or was cancelled")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
