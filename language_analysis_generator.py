#!/usr/bin/env python3
"""
Language Analysis Generator
Generates detailed language-specific reports from translation results
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime


class LanguageAnalysisGenerator:
    """
    Generates comprehensive language analysis from translation results
    """
    
    def __init__(self):
        self.language_families = {
            'Romance': ['fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'ron_Latn', 'cat_Latn', 'srd_Latn', 'scn_Latn'],
            'Germanic': ['deu_Latn', 'nld_Latn', 'dan_Latn', 'swe_Latn', 'nob_Latn', 'nno_Latn', 'isl_Latn', 'fao_Latn'],
            'Slavic': ['rus_Cyrl', 'pol_Latn', 'ces_Latn', 'slk_Latn', 'bul_Cyrl', 'hrv_Latn', 'srp_Cyrl', 'slv_Latn'],
            'Arabic': ['arb_Arab', 'ary_Arab', 'arz_Arab', 'acm_Arab', 'apc_Arab', 'ajp_Arab', 'aeb_Arab', 'ars_Arab'],
            'Indic': ['hin_Deva', 'ben_Beng', 'guj_Gujr', 'pan_Guru', 'mar_Deva', 'npi_Deva', 'asm_Beng', 'ory_Orya'],
            'East_Asian': ['zho_Hans', 'zho_Hant', 'jpn_Jpan', 'kor_Hang'],
            'African_Niger_Congo': ['swa_Latn', 'hau_Latn', 'yor_Latn', 'ibo_Latn', 'kin_Latn', 'run_Latn', 'nya_Latn'],
            'Afroasiatic': ['amh_Ethi', 'som_Latn', 'heb_Hebr', 'mlt_Latn'],
            'Turkic': ['tur_Latn', 'aze_Latn', 'kaz_Cyrl', 'kir_Cyrl', 'uzb_Latn', 'tat_Cyrl'],
            'Austronesian': ['ind_Latn', 'msa_Latn', 'tgl_Latn', 'jav_Latn', 'sun_Latn', 'min_Latn', 'bug_Latn'],
            'Celtic': ['gle_Latn', 'gla_Latn', 'cym_Latn', 'bre_Latn'],
            'Finno_Ugric': ['fin_Latn', 'hun_Latn', 'est_Latn'],
            'Other_European': ['ell_Grek', 'lav_Latn', 'lit_Latn', 'bel_Cyrl', 'ukr_Cyrl', 'mkd_Cyrl']
        }
        
        # Language names for better readability
        self.language_names = {
            'fra_Latn': 'French', 'spa_Latn': 'Spanish', 'ita_Latn': 'Italian', 'por_Latn': 'Portuguese',
            'deu_Latn': 'German', 'nld_Latn': 'Dutch', 'dan_Latn': 'Danish', 'swe_Latn': 'Swedish',
            'rus_Cyrl': 'Russian', 'pol_Latn': 'Polish', 'ces_Latn': 'Czech', 'slk_Latn': 'Slovak',
            'arb_Arab': 'Arabic', 'hin_Deva': 'Hindi', 'ben_Beng': 'Bengali', 'zho_Hans': 'Chinese (Simplified)',
            'jpn_Jpan': 'Japanese', 'kor_Hang': 'Korean', 'swa_Latn': 'Swahili', 'hau_Latn': 'Hausa'
        }
    
    def analyze_translation_results(self, results_file: str) -> Dict:
        """
        Analyze translation results file and generate language statistics
        
        Args:
            results_file: Path to translation results JSON file
            
        Returns:
            Dictionary with comprehensive language analysis
        """
        
        print(f"📊 Analyzing language patterns in: {results_file}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            if not results:
                raise ValueError("No results found in file")
            
            print(f"✅ Loaded {len(results)} translation results")
            
            # Generate comprehensive analysis
            analysis = self._generate_comprehensive_analysis(results)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Failed to analyze results: {e}")
            raise
    
    def _generate_comprehensive_analysis(self, results: List[Dict]) -> Dict:
        """Generate comprehensive language analysis"""
        
        # Initialize counters
        lang_same_count = defaultdict(int)
        lang_translated_count = defaultdict(int)
        lang_error_count = defaultdict(int)
        
        # Term-specific analysis
        term_details = []
        
        # Language family counters
        family_same_count = defaultdict(int)
        family_translated_count = defaultdict(int)
        
        # Process each result
        for result in results:
            if result.get('error'):
                continue
            
            term = result.get('term', '')
            frequency = result.get('frequency', 0)
            same_langs = result.get('same_language_codes', [])
            translated_langs = result.get('translated_language_codes', [])
            error_langs = result.get('error_language_codes', [])
            
            # Count per language
            for lang in same_langs:
                lang_same_count[lang] += 1
            for lang in translated_langs:
                lang_translated_count[lang] += 1
            for lang in error_langs:
                lang_error_count[lang] += 1
            
            # Term details with language-specific data
            term_detail = {
                'term': term,
                'frequency': frequency,
                'translatability_score': result.get('translatability_score', 0.0),
                'total_languages': len(same_langs) + len(translated_langs) + len(error_langs),
                'languages_keeping_same': {
                    'codes': same_langs,
                    'names': [self.language_names.get(lang, lang) for lang in same_langs[:10]],
                    'count': len(same_langs)
                },
                'languages_translating': {
                    'codes': translated_langs,
                    'names': [self.language_names.get(lang, lang) for lang in translated_langs[:10]],
                    'count': len(translated_langs)
                },
                'languages_with_errors': {
                    'codes': error_langs,
                    'count': len(error_langs)
                },
                'sample_translations': result.get('sample_translations', {})
            }
            term_details.append(term_detail)
            
            # Language family analysis
            for family, family_langs in self.language_families.items():
                family_same = sum(1 for lang in same_langs if lang in family_langs)
                family_translated = sum(1 for lang in translated_langs if lang in family_langs)
                
                family_same_count[family] += family_same
                family_translated_count[family] += family_translated
        
        # Calculate language statistics
        all_languages = set(lang_same_count.keys()) | set(lang_translated_count.keys())
        language_statistics = {}
        
        for lang in all_languages:
            same_count = lang_same_count[lang]
            translated_count = lang_translated_count[lang]
            error_count = lang_error_count[lang]
            total_processed = same_count + translated_count + error_count
            
            if total_processed > 0:
                same_pct = (same_count / total_processed) * 100
                translated_pct = (translated_count / total_processed) * 100
                error_pct = (error_count / total_processed) * 100
                
                # Determine borrowing tendency
                if same_pct > 70:
                    tendency = 'high_borrowing'
                elif same_pct > 40:
                    tendency = 'medium_borrowing'
                elif translated_pct > 70:
                    tendency = 'high_translation'
                else:
                    tendency = 'mixed'
                
                language_statistics[lang] = {
                    'language_name': self.language_names.get(lang, lang),
                    'language_code': lang,
                    'terms_keeping_same': same_count,
                    'terms_translated': translated_count,
                    'terms_with_errors': error_count,
                    'total_terms_processed': total_processed,
                    'same_percentage': round(same_pct, 2),
                    'translated_percentage': round(translated_pct, 2),
                    'error_percentage': round(error_pct, 2),
                    'borrowing_tendency': tendency
                }
        
        # Language family analysis
        family_analysis = {}
        total_terms = len(results)
        
        for family, family_langs in self.language_families.items():
            same_total = family_same_count[family]
            translated_total = family_translated_count[family]
            family_size = len(family_langs)
            
            if total_terms > 0 and family_size > 0:
                avg_same_pct = (same_total / (total_terms * family_size)) * 100
                avg_translated_pct = (translated_total / (total_terms * family_size)) * 100
                
                total_instances = same_total + translated_total
                borrowing_ratio = same_total / total_instances if total_instances > 0 else 0
                
                if borrowing_ratio > 0.6:
                    family_tendency = 'high_borrowing'
                elif borrowing_ratio > 0.3:
                    family_tendency = 'medium_borrowing'
                else:
                    family_tendency = 'high_translation'
                
                family_analysis[family] = {
                    'family_name': family.replace('_', ' '),
                    'languages_in_family': family_size,
                    'total_same_instances': same_total,
                    'total_translated_instances': translated_total,
                    'average_same_percentage': round(avg_same_pct, 2),
                    'average_translated_percentage': round(avg_translated_pct, 2),
                    'borrowing_tendency': family_tendency,
                    'sample_languages': [self.language_names.get(lang, lang) for lang in family_langs[:5]]
                }
        
        # Top rankings
        top_borrowing_languages = sorted(
            [(lang, stats['same_percentage'], stats['language_name']) 
             for lang, stats in language_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:25]
        
        top_translating_languages = sorted(
            [(lang, stats['translated_percentage'], stats['language_name']) 
             for lang, stats in language_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:25]
        
        most_same_absolute = sorted(
            [(lang, stats['terms_keeping_same'], stats['language_name']) 
             for lang, stats in language_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:25]
        
        most_translated_absolute = sorted(
            [(lang, stats['terms_translated'], stats['language_name']) 
             for lang, stats in language_statistics.items()],
            key=lambda x: x[1], reverse=True
        )[:25]
        
        return {
            'analysis_info': {
                'generated_at': datetime.now().isoformat(),
                'total_terms_analyzed': len(results),
                'total_languages_analyzed': len(all_languages),
                'total_language_families': len(self.language_families)
            },
            'summary_statistics': {
                'languages_with_high_borrowing': len([l for l, s in language_statistics.items() if s['same_percentage'] > 70]),
                'languages_with_high_translation': len([l for l, s in language_statistics.items() if s['translated_percentage'] > 70]),
                'languages_with_mixed_behavior': len([l for l, s in language_statistics.items() if 30 <= s['same_percentage'] <= 70]),
                'average_borrowing_percentage': round(sum(s['same_percentage'] for s in language_statistics.values()) / len(language_statistics), 2),
                'average_translation_percentage': round(sum(s['translated_percentage'] for s in language_statistics.values()) / len(language_statistics), 2)
            },
            'language_statistics': language_statistics,
            'language_family_analysis': family_analysis,
            'top_rankings': {
                'highest_borrowing_percentage': [
                    {'language_code': lang, 'language_name': name, 'borrowing_percentage': pct}
                    for lang, pct, name in top_borrowing_languages
                ],
                'highest_translation_percentage': [
                    {'language_code': lang, 'language_name': name, 'translation_percentage': pct}
                    for lang, pct, name in top_translating_languages
                ],
                'most_terms_keeping_same': [
                    {'language_code': lang, 'language_name': name, 'terms_same': count}
                    for lang, count, name in most_same_absolute
                ],
                'most_terms_translating': [
                    {'language_code': lang, 'language_name': name, 'terms_translated': count}
                    for lang, count, name in most_translated_absolute
                ]
            },
            'detailed_term_analysis': term_details[:200]  # First 200 terms with detailed language data
        }
    
    def generate_report(self, analysis: Dict, output_file: str):
        """Generate and save comprehensive language analysis report"""
        
        print(f"📝 Generating comprehensive language analysis report...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Language analysis report saved to: {output_file}")
            
            # Also generate a human-readable summary
            summary_file = output_file.replace('.json', '_summary.txt')
            self._generate_human_readable_summary(analysis, summary_file)
            
        except Exception as e:
            print(f"❌ Failed to save language analysis report: {e}")
    
    def _generate_human_readable_summary(self, analysis: Dict, summary_file: str):
        """Generate human-readable summary"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🌍 COMPREHENSIVE LANGUAGE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary stats
            summary = analysis['summary_statistics']
            f.write("📊 SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Terms Analyzed: {analysis['analysis_info']['total_terms_analyzed']:,}\n")
            f.write(f"Total Languages Analyzed: {analysis['analysis_info']['total_languages_analyzed']}\n")
            f.write(f"Average Borrowing Rate: {summary['average_borrowing_percentage']}%\n")
            f.write(f"Average Translation Rate: {summary['average_translation_percentage']}%\n")
            f.write(f"Languages with High Borrowing (>70%): {summary['languages_with_high_borrowing']}\n")
            f.write(f"Languages with High Translation (>70%): {summary['languages_with_high_translation']}\n\n")
            
            # Top borrowing languages
            f.write("🔝 TOP 10 LANGUAGES BY BORROWING PERCENTAGE\n")
            f.write("-" * 50 + "\n")
            for i, lang_data in enumerate(analysis['top_rankings']['highest_borrowing_percentage'][:10], 1):
                f.write(f"{i:2d}. {lang_data['language_name']} ({lang_data['language_code']}): {lang_data['borrowing_percentage']}%\n")
            
            f.write("\n🔝 TOP 10 LANGUAGES BY TRANSLATION PERCENTAGE\n")
            f.write("-" * 50 + "\n")
            for i, lang_data in enumerate(analysis['top_rankings']['highest_translation_percentage'][:10], 1):
                f.write(f"{i:2d}. {lang_data['language_name']} ({lang_data['language_code']}): {lang_data['translation_percentage']}%\n")
            
            # Language family analysis
            f.write("\n🏛️ LANGUAGE FAMILY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for family, data in analysis['language_family_analysis'].items():
                f.write(f"{data['family_name']}:\n")
                f.write(f"  • Languages: {data['languages_in_family']}\n")
                f.write(f"  • Avg Borrowing: {data['average_same_percentage']}%\n")
                f.write(f"  • Avg Translation: {data['average_translated_percentage']}%\n")
                f.write(f"  • Tendency: {data['borrowing_tendency'].replace('_', ' ').title()}\n")
                f.write(f"  • Examples: {', '.join(data['sample_languages'])}\n\n")
        
        print(f"📋 Human-readable summary saved to: {summary_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate Language Analysis Reports")
    parser.add_argument("results_file", help="Path to translation results JSON file")
    parser.add_argument("--output-dir", default="language_analysis", 
                       help="Output directory for analysis reports")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate analysis
    analyzer = LanguageAnalysisGenerator()
    analysis = analyzer.analyze_translation_results(args.results_file)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(args.results_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{base_name}_language_analysis_{timestamp}.json")
    
    # Save report
    analyzer.generate_report(analysis, output_file)
    
    print(f"\n🎉 Language analysis completed!")
    print(f"📁 Reports saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

