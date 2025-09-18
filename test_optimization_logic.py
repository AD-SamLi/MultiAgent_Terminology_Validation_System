#!/usr/bin/env python3
"""
🧪 TEST OPTIMIZATION LOGIC
=========================

Quick test script to validate the smart language selection logic
and demonstrate the optimization strategies in action.
"""

import sys
sys.path.append('/home/samli/Documents/Python/Term_Verify')

from optimized_smart_runner import OptimizedSmartRunner, OptimizedSmartConfig

def test_term_categorization():
    """Test term categorization logic"""
    print("🧪 TESTING TERM CATEGORIZATION")
    print("=" * 40)
    
    config = OptimizedSmartConfig()
    runner = OptimizedSmartRunner(config=config)
    
    test_terms = [
        # Technical terms
        "software development",
        "API endpoint",
        "database system",
        "GPU acceleration",
        
        # Brand terms
        "Microsoft Office",
        "AMD Ryzen",
        "NVIDIA RTX",
        "Intel Core i7",
        
        # Business terms
        "market analysis",
        "customer service",
        "revenue growth",
        "business model",
        
        # Common terms
        "family dinner",
        "hospital visit",
        "school education",
        "car maintenance",
        
        # General terms
        "environmental impact",
        "cultural diversity",
        "social media",
        "climate change"
    ]
    
    for term in test_terms:
        category = runner._categorize_term(term)
        languages, tier = runner._select_languages_for_term(term)
        
        print(f"Term: '{term}'")
        print(f"  Category: {category}")
        print(f"  Processing Tier: {tier}")
        print(f"  Languages Selected: {len(languages)} (vs 202 full)")
        print(f"  Languages Saved: {202 - len(languages)}")
        print(f"  Efficiency Gain: {((202 - len(languages)) / 202 * 100):.1f}%")
        print()

def test_language_sets():
    """Test language set sizes and composition"""
    print("🌍 TESTING LANGUAGE SETS")
    print("=" * 40)
    
    config = OptimizedSmartConfig()
    runner = OptimizedSmartRunner(config=config)
    
    print(f"Core Languages ({len(runner.core_languages)}):")
    print(f"  Sample: {runner.core_languages[:10]}")
    print()
    
    print(f"Extended Languages ({len(runner.extended_languages)}):")
    extended_only = [lang for lang in runner.extended_languages if lang not in runner.core_languages]
    print(f"  Additional in Extended: {len(extended_only)}")
    print(f"  Sample Additional: {extended_only[:10]}")
    print()
    
    print(f"Full Languages ({len(runner.full_languages)}):")
    full_only = [lang for lang in runner.full_languages if lang not in runner.extended_languages]
    print(f"  Additional in Full: {len(full_only)}")
    print(f"  Sample Additional: {full_only[:10]}")
    print()
    
    print(f"High Borrowing Languages ({len(runner.high_borrowing_languages)}):")
    print(f"  Languages: {runner.high_borrowing_languages}")

def test_expansion_logic():
    """Test adaptive expansion logic"""
    print("🚀 TESTING EXPANSION LOGIC")
    print("=" * 40)
    
    config = OptimizedSmartConfig()
    runner = OptimizedSmartRunner(config=config)
    
    # Mock results with different translatability scores
    test_results = [
        {
            'term': 'high_translatability_term',
            'translatability_score': 0.98,
            'same_language_codes': ['eng_Latn', 'lus_Latn'],
            'translated_language_codes': runner.core_languages[2:],  # Most languages translate
        },
        {
            'term': 'medium_translatability_term',
            'translatability_score': 0.92,
            'same_language_codes': ['eng_Latn', 'lus_Latn', 'knc_Latn', 'lmo_Latn'],
            'translated_language_codes': runner.core_languages[4:],  # Some borrowing
        },
        {
            'term': 'low_translatability_term',
            'translatability_score': 0.25,
            'same_language_codes': runner.core_languages[:45],  # High borrowing
            'translated_language_codes': runner.core_languages[45:],
        }
    ]
    
    for result in test_results:
        term = result['term']
        score = result['translatability_score']
        
        should_expand, additional_langs = runner._should_expand_languages(result)
        
        print(f"Term: {term}")
        print(f"  Translatability Score: {score:.3f}")
        print(f"  Should Expand: {should_expand}")
        print(f"  Additional Languages: {len(additional_langs) if additional_langs else 0}")
        
        if should_expand:
            total_final = len(result['same_language_codes']) + len(result['translated_language_codes']) + len(additional_langs)
            print(f"  Final Language Count: {total_final}")
            print(f"  Efficiency vs Full: {((202 - total_final) / 202 * 100):.1f}% saved")
        else:
            original_count = len(result['same_language_codes']) + len(result['translated_language_codes'])
            print(f"  Final Language Count: {original_count} (no expansion)")
            print(f"  Efficiency vs Full: {((202 - original_count) / 202 * 100):.1f}% saved")
        print()

def calculate_expected_performance():
    """Calculate expected performance improvements"""
    print("📈 EXPECTED PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Based on analysis insights
    total_terms = 55103
    full_languages = 202
    
    # Conservative estimates based on term categories
    technical_terms_pct = 25  # 25% technical terms
    brand_terms_pct = 15      # 15% brand terms  
    business_terms_pct = 10   # 10% business terms
    common_terms_pct = 30     # 30% common terms
    general_terms_pct = 20    # 20% general terms
    
    # Average languages per category (conservative estimates)
    technical_avg_langs = 60   # Core set
    brand_avg_langs = 30       # Minimal set (high borrowing)
    business_avg_langs = 60    # Core set
    common_avg_langs = 80      # Core + some expansion
    general_avg_langs = 70     # Adaptive average
    
    # Calculate weighted average
    weighted_avg_langs = (
        (technical_terms_pct * technical_avg_langs) +
        (brand_terms_pct * brand_avg_langs) +
        (business_terms_pct * business_avg_langs) +
        (common_terms_pct * common_avg_langs) +
        (general_terms_pct * general_avg_langs)
    ) / 100
    
    # Performance calculations
    full_processing_translations = total_terms * full_languages
    smart_processing_translations = total_terms * weighted_avg_langs
    
    speedup_factor = full_processing_translations / smart_processing_translations
    efficiency_gain = ((full_processing_translations - smart_processing_translations) / full_processing_translations) * 100
    
    print(f"Dataset Analysis:")
    print(f"  Total Terms: {total_terms:,}")
    print(f"  Full Languages: {full_languages}")
    print()
    
    print(f"Term Category Distribution (estimated):")
    print(f"  Technical: {technical_terms_pct}% → avg {technical_avg_langs} languages")
    print(f"  Brand: {brand_terms_pct}% → avg {brand_avg_langs} languages")
    print(f"  Business: {business_terms_pct}% → avg {business_avg_langs} languages")
    print(f"  Common: {common_terms_pct}% → avg {common_avg_langs} languages")
    print(f"  General: {general_terms_pct}% → avg {general_avg_langs} languages")
    print()
    
    print(f"Performance Projections:")
    print(f"  Full Processing: {full_processing_translations:,} translations")
    print(f"  Smart Processing: {smart_processing_translations:,.0f} translations")
    print(f"  Weighted Avg Languages: {weighted_avg_langs:.1f}")
    print(f"  Speedup Factor: {speedup_factor:.1f}x faster")
    print(f"  Efficiency Gain: {efficiency_gain:.1f}%")
    print()
    
    # Time estimates (based on current processing rate)
    current_rate = 0.198  # terms/sec from analysis
    full_processing_hours = total_terms / current_rate / 3600
    smart_processing_hours = full_processing_hours / speedup_factor
    time_saved_hours = full_processing_hours - smart_processing_hours
    
    print(f"Time Estimates:")
    print(f"  Full Processing ETA: {full_processing_hours:.1f} hours")
    print(f"  Smart Processing ETA: {smart_processing_hours:.1f} hours")
    print(f"  Time Saved: {time_saved_hours:.1f} hours ({(time_saved_hours/full_processing_hours*100):.1f}%)")

def main():
    """Run all tests"""
    print("🧪 OPTIMIZATION LOGIC TESTING SUITE")
    print("=" * 50)
    print()
    
    try:
        test_term_categorization()
        print("\n" + "="*50 + "\n")
        
        test_language_sets()
        print("\n" + "="*50 + "\n")
        
        test_expansion_logic()
        print("\n" + "="*50 + "\n")
        
        calculate_expected_performance()
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for optimized smart processing!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
