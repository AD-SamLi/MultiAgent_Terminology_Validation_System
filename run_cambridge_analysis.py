#!/usr/bin/env python3
"""
Cambridge Dictionary Analysis Runner
Runs the Cambridge dictionary agent on your terms data using the same workflow as analyze_dictionary_terms
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cambridge_dictionary_agent import CambridgeDictionaryAgent, load_terms_data, save_results
    print("✅ Cambridge dictionary agent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import cambridge_dictionary_agent: {e}")
    print("   Make sure cambridge_dictionary_agent.py is in the same directory")
    sys.exit(1)


def run_cambridge_analysis():
    """
    Run Cambridge dictionary analysis on your terms data
    Same interface as your existing analyze_dictionary_terms functions
    """
    print("🚀 Cambridge Dictionary Term Analysis")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the Cambridge agent
    print("\n📚 Initializing Cambridge Dictionary Agent...")
    agent = CambridgeDictionaryAgent()
    
    if not agent.dictionary_tool:
        print("❌ Cambridge dictionary not available.")
        print("   Install with: pip install cambridge")
        return False
    
    # Look for input files (same priority as your existing scripts)
    input_files = [
        "Cleaned_Complete_Terms_Data.json",
        "Cleaned_Summary_Terms_Data.json", 
        "Combined_Terms_Data.json",
        "Term_Extracted_result.json"  # if you have this format
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
    
    # Ask user for limits
    print(f"\n⚙️  Analysis options:")
    max_terms = None
    
    try:
        user_input = input(f"   Analyze all {len(terms):,} terms? (y/n/number): ").strip().lower()
        
        if user_input in ['n', 'no']:
            max_terms = int(input("   How many terms to analyze? "))
        elif user_input.isdigit():
            max_terms = int(user_input)
        elif user_input in ['y', 'yes', '']:
            max_terms = None  # Analyze all
        else:
            print("   Invalid input, analyzing all terms...")
            max_terms = None
            
    except (ValueError, KeyboardInterrupt):
        print("   Using default: analyzing all terms")
        max_terms = None
    
    # Run the analysis
    print(f"\n🔄 Starting Cambridge dictionary analysis...")
    start_time = time.time()
    
    try:
        dictionary_words, non_dictionary_words = agent.analyze_terms_cambridge(
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
        print(f"   📊 Results:")
        print(f"      📖 Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_analyzed*100:.1f}%)")
        print(f"      🔧 Non-dictionary: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_analyzed*100:.1f}%)")
        print(f"   📁 Files saved with timestamp")
        
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


def compare_with_original():
    """
    Compare Cambridge results with original PyMultiDictionary results if available
    """
    print("\n🔍 Looking for comparison data...")
    
    # Look for existing results
    existing_files = [
        "Dictionary_Terms_Found.json",
        "Non_Dictionary_Terms.json"
    ]
    
    found_files = [f for f in existing_files if os.path.exists(f)]
    
    if found_files:
        print(f"✅ Found existing analysis files: {found_files}")
        print("   You can manually compare the results")
    else:
        print("   No existing analysis files found for comparison")


def main():
    """
    Main entry point
    """
    try:
        success = run_cambridge_analysis()
        
        if success:
            compare_with_original()
            print(f"\n✅ Cambridge dictionary analysis completed successfully!")
            print(f"   This method is much faster than PyMultiDictionary API calls")
            print(f"   Results are saved with timestamp for comparison")
        else:
            print(f"\n❌ Analysis failed or was cancelled")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
