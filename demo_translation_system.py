#!/usr/bin/env python3
"""
Demo script for NLLB Translation System
Shows basic functionality without processing the full dataset
"""

import os
import json
import time
from datetime import datetime

def demo_nllb_tool():
    """Demo the basic NLLB translation tool"""
    print("🧪 DEMO: NLLB Translation Tool")
    print("=" * 40)
    
    try:
        from nllb_translation_tool import NLLBTranslationTool
        
        # Initialize with small batch size and smaller model for demo
        print("🔧 Initializing NLLB tool...")
        tool = NLLBTranslationTool(model_name="small", batch_size=2)
        
        # Demo 1: Single translation
        print("\n1️⃣ Single Translation Demo:")
        result = tool.translate_text("computer", "eng_Latn", "spa_Latn")
        print(f"   '{result.original_text}' -> '{result.translated_text}' (Spanish)")
        print(f"   Same as original: {result.is_same}")
        
        # Demo 2: Batch translation
        print("\n2️⃣ Batch Translation Demo:")
        terms = ["software", "algorithm", "database"]
        results = tool.translate_batch(terms, "eng_Latn", "fra_Latn")
        for term, result in zip(terms, results):
            print(f"   '{term}' -> '{result.translated_text}' (French)")
        
        # Demo 3: Multiple languages for one term
        print("\n3️⃣ Multi-Language Demo (first 5 languages):")
        sample_languages = ["spa_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "por_Latn"]
        
        for lang in sample_languages:
            result = tool.translate_text("internet", "eng_Latn", lang)
            lang_name = tool.get_language_info(lang)
            print(f"   {lang_name}: '{result.translated_text}' (same: {result.is_same})")
        
        print("\n✅ NLLB Tool Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ NLLB Tool Demo failed: {e}")
        return False


def demo_translation_agent():
    """Demo the smolagents translation agent"""
    print("\n🤖 DEMO: Translation Agent")
    print("=" * 40)
    
    try:
        from nllb_translation_agent import NLLBTranslationAgent
        
        print("🔧 Initializing translation agent...")
        agent = NLLBTranslationAgent(device="auto", batch_size=2)
        
        # Demo agent capabilities
        print("\n1️⃣ Language Overview:")
        overview = agent.get_language_overview()
        print("   Agent response:", overview[:200] + "..." if len(overview) > 200 else overview)
        
        print("\n2️⃣ Term Analysis:")
        analysis = agent.analyze_term_translatability("software")
        print("   Agent response:", analysis[:200] + "..." if len(analysis) > 200 else analysis)
        
        print("\n✅ Translation Agent Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Translation Agent Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_small_batch_processing():
    """Demo processing a small batch of terms"""
    print("\n📊 DEMO: Small Batch Processing")
    print("=" * 40)
    
    try:
        from term_translation_processor import TermTranslationProcessor
        
        # Create sample term data
        sample_terms = [
            {"term": "computer", "frequency": 150},
            {"term": "software", "frequency": 120},
            {"term": "internet", "frequency": 200},
            {"term": "algorithm", "frequency": 80},
            {"term": "database", "frequency": 90}
        ]
        
        print(f"🔧 Processing {len(sample_terms)} sample terms...")
        
        # Initialize processor with small settings and smaller model
        processor = TermTranslationProcessor(device="auto", batch_size=2, max_workers=1)
        
        # Process the sample terms
        results = []
        for i, term_data in enumerate(sample_terms, 1):
            print(f"   Processing term {i}/{len(sample_terms)}: {term_data['term']}")
            result = processor.analyze_term_translatability(term_data)
            results.append(result)
        
        # Show results summary
        print(f"\n📊 Results Summary:")
        for result in results:
            if result.total_languages > 0:
                print(f"   '{result.term}': {result.translated_languages}/{result.total_languages} languages translated")
                print(f"     Translatability score: {result.translatability_score:.3f}")
                if result.sample_translations:
                    sample_lang, sample_trans = next(iter(result.sample_translations.items()))
                    lang_name = processor.translator.get_language_info(sample_lang)
                    print(f"     Sample ({lang_name}): '{sample_trans}'")
            else:
                print(f"   '{result.term}': Processing failed")
        
        print(f"\n✅ Small Batch Processing Demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Small Batch Processing Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_analysis_features():
    """Demo the analysis features with mock data"""
    print("\n📈 DEMO: Analysis Features")
    print("=" * 40)
    
    # Create mock translation results
    mock_results = {
        "processing_info": {
            "status": "completed",
            "total_terms": 5,
            "source_language": "eng_Latn"
        },
        "results": [
            {
                "term": "computer",
                "frequency": 150,
                "total_languages": 199,
                "same_languages": 120,
                "translated_languages": 75,
                "error_languages": 4,
                "translatability_score": 0.385,
                "same_language_codes": ["jpn_Jpan", "kor_Hang", "hin_Deva"],
                "translated_language_codes": ["spa_Latn", "fra_Latn", "deu_Latn"],
                "sample_translations": {"spa_Latn": "computadora", "fra_Latn": "ordinateur"}
            },
            {
                "term": "software",
                "frequency": 120,
                "total_languages": 199,
                "same_languages": 140,
                "translated_languages": 55,
                "error_languages": 4,
                "translatability_score": 0.282,
                "same_language_codes": ["jpn_Jpan", "kor_Hang", "hin_Deva"],
                "translated_language_codes": ["spa_Latn", "fra_Latn"],
                "sample_translations": {"spa_Latn": "software", "fra_Latn": "logiciel"}
            },
            {
                "term": "internet",
                "frequency": 200,
                "total_languages": 199,
                "same_languages": 180,
                "translated_languages": 15,
                "error_languages": 4,
                "translatability_score": 0.077,
                "same_language_codes": ["jpn_Jpan", "kor_Hang", "hin_Deva", "spa_Latn"],
                "translated_language_codes": ["zho_Hans"],
                "sample_translations": {"zho_Hans": "互联网"}
            }
        ]
    }
    
    # Save mock data temporarily
    mock_file = "demo_translation_results.json"
    
    try:
        with open(mock_file, 'w', encoding='utf-8') as f:
            json.dump(mock_results, f, indent=2)
        
        print(f"📁 Created mock results file: {mock_file}")
        
        # Try to run analysis
        try:
            from translation_analyzer import TranslationAnalyzer
            
            print("🔧 Running analysis on mock data...")
            analyzer = TranslationAnalyzer(mock_file)
            
            # Generate basic statistics
            valid_results = [r for r in analyzer.results if r.get('total_languages', 0) > 0]
            
            if valid_results:
                scores = [r.get('translatability_score', 0) for r in valid_results]
                avg_score = sum(scores) / len(scores)
                
                print(f"📊 Analysis Results:")
                print(f"   Terms analyzed: {len(valid_results)}")
                print(f"   Average translatability: {avg_score:.3f}")
                
                # Categorize terms
                highly_translatable = [r for r in valid_results if r.get('translatability_score', 0) >= 0.8]
                moderately_translatable = [r for r in valid_results if 0.3 <= r.get('translatability_score', 0) < 0.8]
                poorly_translatable = [r for r in valid_results if r.get('translatability_score', 0) < 0.3]
                
                print(f"   Highly translatable: {len(highly_translatable)}")
                print(f"   Moderately translatable: {len(moderately_translatable)}")
                print(f"   Poorly translatable: {len(poorly_translatable)}")
                
                # Show examples
                print(f"\n🔍 Examples:")
                for result in valid_results:
                    term = result.get('term', '')
                    score = result.get('translatability_score', 0)
                    same_count = result.get('same_languages', 0)
                    trans_count = result.get('translated_languages', 0)
                    
                    print(f"   '{term}': {trans_count}/{same_count + trans_count} translated (score: {score:.3f})")
        
        except ImportError:
            print("⚠️  Analysis modules not available for full demo")
        
        print(f"\n✅ Analysis Features Demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Analysis Features Demo failed: {e}")
        return False
    
    finally:
        # Clean up mock file
        if os.path.exists(mock_file):
            os.remove(mock_file)


def main():
    """Run all demos"""
    print("🌟 NLLB TRANSLATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the basic functionality without processing the full dataset.")
    print("For full analysis, use: python run_translation_analysis.py --test-mode")
    print()
    
    demos = [
        ("NLLB Tool", demo_nllb_tool),
        ("Translation Agent", demo_translation_agent), 
        ("Small Batch Processing", demo_small_batch_processing),
        ("Analysis Features", demo_analysis_features)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed")
                
        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            results.append((demo_name, False))
        
        # Small delay between demos
        time.sleep(1)
    
    # Final summary
    print(f"\n🏁 DEMO SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {demo_name}")
    
    print(f"\nOverall: {successful}/{total} demos successful")
    
    if successful == total:
        print("🎉 All demos completed successfully! System is ready for use.")
        print("\nNext steps:")
        print("1. Run test mode: python run_translation_analysis.py --test-mode")
        print("2. Run full analysis: python run_translation_analysis.py")
    else:
        print("⚠️  Some demos failed. Check error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
