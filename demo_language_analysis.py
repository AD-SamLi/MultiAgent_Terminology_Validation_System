#!/usr/bin/env python3
"""
Demo Language Analysis
Creates sample translation results and demonstrates the enhanced language analysis
"""

import json
import os
from datetime import datetime
from language_analysis_generator import LanguageAnalysisGenerator


def create_sample_translation_results():
    """Create sample translation results to demonstrate language analysis"""
    
    sample_results = [
        {
            "term": "algorithm",
            "frequency": 1250,
            "total_languages": 202,
            "same_languages": 85,
            "translated_languages": 115,
            "error_languages": 2,
            "translatability_score": 0.575,
            "same_language_codes": [
                "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "deu_Latn", "nld_Latn", 
                "dan_Latn", "swe_Latn", "nob_Latn", "pol_Latn", "ces_Latn", "rus_Cyrl",
                "arb_Arab", "tur_Latn", "jpn_Jpan", "kor_Hang", "hin_Deva"
            ],
            "translated_language_codes": [
                "zho_Hans", "zho_Hant", "ben_Beng", "tam_Taml", "tel_Telu", "mar_Deva",
                "guj_Gujr", "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru", "asm_Beng",
                "fin_Latn", "hun_Latn", "est_Latn", "lav_Latn", "lit_Latn", "ell_Grek"
            ],
            "error_language_codes": ["sat_Olck", "mni_Beng"],
            "sample_translations": {
                "zho_Hans": "算法",
                "zho_Hant": "演算法", 
                "ben_Beng": "অ্যালগরিদম",
                "fin_Latn": "algoritmi",
                "hun_Latn": "algoritmus"
            },
            "analysis_timestamp": "2025-09-04T13:45:00",
            "processing_time_seconds": 45.2
        },
        {
            "term": "database",
            "frequency": 2100,
            "total_languages": 202,
            "same_languages": 120,
            "translated_languages": 80,
            "error_languages": 2,
            "translatability_score": 0.4,
            "same_language_codes": [
                "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "deu_Latn", "nld_Latn",
                "dan_Latn", "swe_Latn", "nob_Latn", "nno_Latn", "pol_Latn", "ces_Latn",
                "slk_Latn", "hun_Latn", "ron_Latn", "bul_Cyrl", "rus_Cyrl", "ukr_Cyrl",
                "arb_Arab", "heb_Hebr", "tur_Latn", "jpn_Jpan", "kor_Hang", "hin_Deva",
                "ben_Beng", "guj_Gujr", "mar_Deva", "tam_Taml", "tel_Telu"
            ],
            "translated_language_codes": [
                "zho_Hans", "zho_Hant", "fin_Latn", "est_Latn", "lav_Latn", "lit_Latn",
                "ell_Grek", "mlt_Latn", "eus_Latn", "cat_Latn", "glg_Latn", "srd_Latn",
                "vie_Latn", "tha_Thai", "khm_Khmr", "lao_Laoo", "mya_Mymr", "swa_Latn"
            ],
            "error_language_codes": ["dzo_Tibt", "sat_Olck"],
            "sample_translations": {
                "zho_Hans": "数据库",
                "zho_Hant": "資料庫",
                "fin_Latn": "tietokanta",
                "eus_Latn": "datu-base",
                "swa_Latn": "hifadhidata"
            },
            "analysis_timestamp": "2025-09-04T13:45:45",
            "processing_time_seconds": 42.8
        },
        {
            "term": "software",
            "frequency": 3200,
            "total_languages": 202,
            "same_languages": 95,
            "translated_languages": 105,
            "error_languages": 2,
            "translatability_score": 0.525,
            "same_language_codes": [
                "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "cat_Latn", "glg_Latn",
                "deu_Latn", "nld_Latn", "dan_Latn", "swe_Latn", "nob_Latn", "nno_Latn",
                "pol_Latn", "ces_Latn", "slk_Latn", "hun_Latn", "ron_Latn", "rus_Cyrl",
                "arb_Arab", "heb_Hebr", "tur_Latn", "aze_Latn", "kaz_Cyrl", "jpn_Jpan"
            ],
            "translated_language_codes": [
                "zho_Hans", "zho_Hant", "kor_Hang", "hin_Deva", "ben_Beng", "guj_Gujr",
                "mar_Deva", "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym", "ory_Orya",
                "fin_Latn", "est_Latn", "lav_Latn", "lit_Latn", "ell_Grek", "mlt_Latn",
                "eus_Latn", "gle_Latn", "gla_Latn", "cym_Latn", "swa_Latn", "hau_Latn"
            ],
            "error_language_codes": ["bod_Tibt", "mni_Beng"],
            "sample_translations": {
                "zho_Hans": "软件",
                "zho_Hant": "軟體",
                "kor_Hang": "소프트웨어",
                "hin_Deva": "सॉफ्टवेयर",
                "fin_Latn": "ohjelmisto"
            },
            "analysis_timestamp": "2025-09-04T13:46:30",
            "processing_time_seconds": 44.1
        },
        {
            "term": "interface",
            "frequency": 1800,
            "total_languages": 202,
            "same_languages": 110,
            "translated_languages": 90,
            "error_languages": 2,
            "translatability_score": 0.45,
            "same_language_codes": [
                "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "cat_Latn", "ron_Latn",
                "deu_Latn", "nld_Latn", "dan_Latn", "swe_Latn", "nob_Latn", "pol_Latn",
                "ces_Latn", "slk_Latn", "hun_Latn", "bul_Cyrl", "rus_Cyrl", "ukr_Cyrl",
                "arb_Arab", "tur_Latn", "jpn_Jpan", "kor_Hang", "hin_Deva", "ben_Beng"
            ],
            "translated_language_codes": [
                "zho_Hans", "zho_Hant", "fin_Latn", "est_Latn", "ell_Grek", "mlt_Latn",
                "eus_Latn", "gle_Latn", "gla_Latn", "cym_Latn", "vie_Latn", "tha_Thai",
                "swa_Latn", "hau_Latn", "yor_Latn", "ibo_Latn", "amh_Ethi", "som_Latn"
            ],
            "error_language_codes": ["dzo_Tibt", "sat_Olck"],
            "sample_translations": {
                "zho_Hans": "接口",
                "zho_Hant": "介面",
                "fin_Latn": "käyttöliittymä",
                "eus_Latn": "interfaze",
                "swa_Latn": "kiolesura"
            },
            "analysis_timestamp": "2025-09-04T13:47:15",
            "processing_time_seconds": 43.5
        },
        {
            "term": "protocol",
            "frequency": 950,
            "total_languages": 202,
            "same_languages": 130,
            "translated_languages": 70,
            "error_languages": 2,
            "translatability_score": 0.35,
            "same_language_codes": [
                "fra_Latn", "spa_Latn", "ita_Latn", "por_Latn", "cat_Latn", "glg_Latn",
                "ron_Latn", "deu_Latn", "nld_Latn", "dan_Latn", "swe_Latn", "nob_Latn",
                "pol_Latn", "ces_Latn", "slk_Latn", "hun_Latn", "bul_Cyrl", "rus_Cyrl",
                "arb_Arab", "heb_Hebr", "tur_Latn", "aze_Latn", "jpn_Jpan", "kor_Hang",
                "hin_Deva", "ben_Beng", "guj_Gujr", "mar_Deva", "tam_Taml", "tel_Telu"
            ],
            "translated_language_codes": [
                "zho_Hans", "zho_Hant", "fin_Latn", "est_Latn", "lav_Latn", "lit_Latn",
                "ell_Grek", "mlt_Latn", "eus_Latn", "gle_Latn", "gla_Latn", "cym_Latn",
                "swa_Latn", "hau_Latn", "yor_Latn", "ibo_Latn", "amh_Ethi", "som_Latn"
            ],
            "error_language_codes": ["bod_Tibt", "mni_Beng"],
            "sample_translations": {
                "zho_Hans": "协议",
                "zho_Hant": "協定",
                "fin_Latn": "protokolla",
                "eus_Latn": "protokolo",
                "swa_Latn": "itifaki"
            },
            "analysis_timestamp": "2025-09-04T13:48:00",
            "processing_time_seconds": 41.9
        }
    ]
    
    # Create sample results file
    sample_data = {
        "processing_info": {
            "status": "completed",
            "session_id": "demo_20250904_134800",
            "file_type": "dictionary",
            "total_terms": 5,
            "processing_time_seconds": 217.5,
            "terms_per_second": 0.023,
            "start_time": "2025-09-04T13:45:00",
            "end_time": "2025-09-04T13:48:37",
            "source_language": "eng_Latn",
            "target_languages_count": 202,
            "device_used": "cuda",
            "model_size": "small",
            "architecture": "hybrid_gpu_cpu"
        },
        "results": sample_results
    }
    
    # Save sample results
    os.makedirs("demo_results", exist_ok=True)
    sample_file = "demo_results/sample_translation_results.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sample translation results created: {sample_file}")
    return sample_file


def main():
    """Main demo function"""
    print("🌍 LANGUAGE ANALYSIS DEMO")
    print("=" * 50)
    
    # Create sample data
    sample_file = create_sample_translation_results()
    
    # Generate language analysis
    analyzer = LanguageAnalysisGenerator()
    analysis = analyzer.analyze_translation_results(sample_file)
    
    # Save analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"demo_results/language_analysis_demo_{timestamp}.json"
    analyzer.generate_report(analysis, output_file)
    
    # Show key insights
    print(f"\n🔍 KEY LANGUAGE INSIGHTS:")
    print("-" * 30)
    
    summary = analysis['summary_statistics']
    print(f"📊 Languages with High Borrowing (>70%): {summary['languages_with_high_borrowing']}")
    print(f"📊 Languages with High Translation (>70%): {summary['languages_with_high_translation']}")
    print(f"📊 Average Borrowing Rate: {summary['average_borrowing_percentage']}%")
    print(f"📊 Average Translation Rate: {summary['average_translation_percentage']}%")
    
    print(f"\n🔝 TOP BORROWING LANGUAGES:")
    for i, lang_data in enumerate(analysis['top_rankings']['highest_borrowing_percentage'][:5], 1):
        print(f"{i}. {lang_data['language_name']} ({lang_data['language_code']}): {lang_data['borrowing_percentage']}%")
    
    print(f"\n🔝 TOP TRANSLATING LANGUAGES:")
    for i, lang_data in enumerate(analysis['top_rankings']['highest_translation_percentage'][:5], 1):
        print(f"{i}. {lang_data['language_name']} ({lang_data['language_code']}): {lang_data['translation_percentage']}%")
    
    print(f"\n🏛️ LANGUAGE FAMILY PATTERNS:")
    for family, data in list(analysis['language_family_analysis'].items())[:5]:
        print(f"• {data['family_name']}: {data['average_same_percentage']}% borrowing, {data['borrowing_tendency'].replace('_', ' ')}")
    
    print(f"\n📋 DETAILED TERM ANALYSIS (First 2 terms):")
    for i, term_data in enumerate(analysis['detailed_term_analysis'][:2], 1):
        print(f"\n{i}. Term: '{term_data['term']}' (Score: {term_data['translatability_score']})")
        print(f"   Languages keeping same ({term_data['languages_keeping_same']['count']}): {', '.join(term_data['languages_keeping_same']['names'][:5])}...")
        print(f"   Languages translating ({term_data['languages_translating']['count']}): {', '.join(term_data['languages_translating']['names'][:5])}...")
        if term_data['sample_translations']:
            print(f"   Sample translations: {list(term_data['sample_translations'].items())[:3]}")
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Full reports saved in: demo_results/")
    print(f"📊 This shows the enhanced language-specific data you requested!")


if __name__ == "__main__":
    main()

