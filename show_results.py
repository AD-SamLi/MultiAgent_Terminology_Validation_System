#!/usr/bin/env python3
"""
Script to display key results from the translation analysis
"""

import json
import os

def show_sample_results():
    """Show sample results from the translation analysis"""
    
    print("🎯 NLLB TRANSLATION ANALYSIS RESULTS")
    print("=" * 60)
    
    # Check if results exist
    dict_file = "translation_results/dictionary_terms_translation_results_20250903_163809.json"
    non_dict_file = "translation_results/non_dictionary_terms_translation_results_20250903_163809.json"
    
    if not os.path.exists(dict_file) or not os.path.exists(non_dict_file):
        print("❌ Results files not found. Run the analysis first.")
        return
    
    # Load dictionary terms results
    with open(dict_file, 'r', encoding='utf-8') as f:
        dict_data = json.load(f)
    
    # Load non-dictionary terms results
    with open(non_dict_file, 'r', encoding='utf-8') as f:
        non_dict_data = json.load(f)
    
    print(f"\n📚 DICTIONARY TERMS SAMPLE (5 examples)")
    print("-" * 50)
    for i, result in enumerate(dict_data['results'][:5]):
        term = result['term']
        score = result['translatability_score']
        same_count = result['same_languages']
        translated_count = result['translated_languages']
        total = result['total_languages']
        
        print(f"{i+1}. Term: \"{term}\"")
        print(f"   📊 Translatability Score: {score:.3f}")
        print(f"   🌍 Languages: {translated_count}/{total} translated, {same_count} kept same")
        
        # Show sample translations
        samples = result.get('sample_translations', {})
        if samples:
            print("   🔄 Sample translations:")
            for lang_code, translation in list(samples.items())[:3]:
                print(f"      {lang_code}: \"{translation}\"")
        print()
    
    print(f"\n📖 NON-DICTIONARY TERMS SAMPLE (5 examples)")
    print("-" * 50)
    for i, result in enumerate(non_dict_data['results'][:5]):
        term = result['term']
        score = result['translatability_score']
        same_count = result['same_languages']
        translated_count = result['translated_languages']
        total = result['total_languages']
        
        print(f"{i+1}. Term: \"{term}\"")
        print(f"   📊 Translatability Score: {score:.3f}")
        print(f"   🌍 Languages: {translated_count}/{total} translated, {same_count} kept same")
        
        # Show sample translations
        samples = result.get('sample_translations', {})
        if samples:
            print("   🔄 Sample translations:")
            for lang_code, translation in list(samples.items())[:3]:
                print(f"      {lang_code}: \"{translation}\"")
        print()
    
    # Summary statistics
    print(f"\n📈 OVERALL STATISTICS")
    print("-" * 50)
    
    dict_processing = dict_data.get('processing_info', {})
    non_dict_processing = non_dict_data.get('processing_info', {})
    
    print(f"Dictionary Terms:")
    print(f"  • Total processed: {dict_processing.get('processed_terms', 0)}")
    print(f"  • Processing time: {dict_processing.get('processing_time_seconds', 0)/60:.1f} minutes")
    print(f"  • Rate: {dict_processing.get('terms_per_second', 0):.3f} terms/second")
    
    print(f"\nNon-Dictionary Terms:")
    print(f"  • Total processed: {non_dict_processing.get('processed_terms', 0)}")
    print(f"  • Processing time: {non_dict_processing.get('processing_time_seconds', 0)/60:.1f} minutes")
    print(f"  • Rate: {non_dict_processing.get('terms_per_second', 0):.3f} terms/second")
    
    # Analysis summary
    dict_summary = dict_data.get('analysis_summary', {})
    non_dict_summary = non_dict_data.get('analysis_summary', {})
    
    if dict_summary:
        dict_overview = dict_summary.get('overview', {})
        print(f"\nDictionary Terms Analysis:")
        print(f"  • Average translatability: {dict_overview.get('average_translatability_score', 0):.3f}")
        print(f"  • Highly translatable: {dict_overview.get('highly_translatable_terms', 0)}")
        print(f"  • Poorly translatable: {dict_overview.get('poorly_translatable_terms', 0)}")
    
    if non_dict_summary:
        non_dict_overview = non_dict_summary.get('overview', {})
        print(f"\nNon-Dictionary Terms Analysis:")
        print(f"  • Average translatability: {non_dict_overview.get('average_translatability_score', 0):.3f}")
        print(f"  • Highly translatable: {non_dict_overview.get('highly_translatable_terms', 0)}")
        print(f"  • Poorly translatable: {non_dict_overview.get('poorly_translatable_terms', 0)}")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Full results available in:")
    print(f"   • translation_results/ - Raw translation data")
    print(f"   • analysis_reports/ - Analysis reports and visualizations")


if __name__ == "__main__":
    show_sample_results()

