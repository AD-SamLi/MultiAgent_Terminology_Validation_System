#!/usr/bin/env python3
"""
Translation Analysis and Reporting System
Analyzes translation results and generates comprehensive reports
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from pathlib import Path

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class AnalysisReport:
    """Container for analysis report data"""
    report_type: str
    timestamp: str
    total_terms: int
    summary_stats: Dict
    detailed_analysis: Dict
    visualizations: List[str]  # Paths to generated plots
    recommendations: List[str]

class TranslationAnalyzer:
    """
    Comprehensive analyzer for translation results
    """
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file
        
        Args:
            results_file: Path to translation results JSON file
        """
        self.results_file = results_file
        self.data = None
        self.results = None
        self.processing_info = None
        
        self._load_data()
    
    def _load_data(self):
        """Load translation results data"""
        print(f"📁 Loading translation results from: {self.results_file}")
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            self.processing_info = self.data.get('processing_info', {})
            self.results = self.data.get('results', [])
            
            print(f"✅ Loaded {len(self.results)} translation results")
            print(f"📊 Processing info: {self.processing_info.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            raise
    
    def generate_translatability_report(self, output_dir: str = "analysis_reports") -> AnalysisReport:
        """Generate comprehensive translatability analysis report"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("📊 Generating translatability analysis report...")
        
        # Basic statistics
        valid_results = [r for r in self.results if r.get('total_languages', 0) > 0]
        
        if not valid_results:
            raise ValueError("No valid results found for analysis")
        
        # Translatability scores
        scores = [r.get('translatability_score', 0) for r in valid_results]
        same_counts = [r.get('same_languages', 0) for r in valid_results]
        translated_counts = [r.get('translated_languages', 0) for r in valid_results]
        error_counts = [r.get('error_languages', 0) for r in valid_results]
        frequencies = [r.get('frequency', 0) for r in valid_results]
        
        # Summary statistics
        summary_stats = {
            "total_terms_analyzed": len(valid_results),
            "average_translatability_score": np.mean(scores),
            "median_translatability_score": np.median(scores),
            "std_translatability_score": np.std(scores),
            "min_translatability_score": np.min(scores),
            "max_translatability_score": np.max(scores),
            "average_same_languages": np.mean(same_counts),
            "average_translated_languages": np.mean(translated_counts),
            "average_error_languages": np.mean(error_counts),
            "total_target_languages": valid_results[0].get('total_languages', 0) if valid_results else 0
        }
        
        # Categorize terms by translatability
        highly_translatable = [r for r in valid_results if r.get('translatability_score', 0) >= 0.8]
        moderately_translatable = [r for r in valid_results if 0.3 <= r.get('translatability_score', 0) < 0.8]
        poorly_translatable = [r for r in valid_results if r.get('translatability_score', 0) < 0.3]
        
        # Language analysis
        language_analysis = self._analyze_language_patterns(valid_results)
        
        # Frequency analysis
        frequency_analysis = self._analyze_frequency_patterns(valid_results)
        
        # Generate visualizations
        visualizations = self._create_translatability_plots(
            valid_results, output_dir, timestamp
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(valid_results, summary_stats)
        
        # Detailed analysis
        detailed_analysis = {
            "translatability_categories": {
                "highly_translatable": {
                    "count": len(highly_translatable),
                    "percentage": (len(highly_translatable) / len(valid_results)) * 100,
                    "examples": [
                        {"term": r["term"], "score": r["translatability_score"], "frequency": r["frequency"]}
                        for r in sorted(highly_translatable, key=lambda x: x["translatability_score"], reverse=True)[:10]
                    ]
                },
                "moderately_translatable": {
                    "count": len(moderately_translatable),
                    "percentage": (len(moderately_translatable) / len(valid_results)) * 100,
                    "examples": [
                        {"term": r["term"], "score": r["translatability_score"], "frequency": r["frequency"]}
                        for r in sorted(moderately_translatable, key=lambda x: x["translatability_score"], reverse=True)[:10]
                    ]
                },
                "poorly_translatable": {
                    "count": len(poorly_translatable),
                    "percentage": (len(poorly_translatable) / len(valid_results)) * 100,
                    "examples": [
                        {"term": r["term"], "score": r["translatability_score"], "frequency": r["frequency"]}
                        for r in sorted(poorly_translatable, key=lambda x: x["frequency"], reverse=True)[:10]
                    ]
                }
            },
            "language_patterns": language_analysis,
            "frequency_patterns": frequency_analysis
        }
        
        # Create report
        report = AnalysisReport(
            report_type="translatability_analysis",
            timestamp=timestamp,
            total_terms=len(valid_results),
            summary_stats=summary_stats,
            detailed_analysis=detailed_analysis,
            visualizations=visualizations,
            recommendations=recommendations
        )
        
        # Save report
        report_file = os.path.join(output_dir, f"translatability_report_{timestamp}.json")
        self._save_report(report, report_file)
        
        # Generate human-readable summary
        self._generate_text_summary(report, output_dir, timestamp)
        
        print(f"✅ Translatability report generated: {report_file}")
        return report
    
    def _analyze_language_patterns(self, results: List[Dict]) -> Dict:
        """Analyze patterns in language behavior"""
        
        # Count languages that keep terms same vs translate
        same_language_counts = Counter()
        translated_language_counts = Counter()
        error_language_counts = Counter()
        
        for result in results:
            for lang in result.get('same_language_codes', []):
                same_language_counts[lang] += 1
            for lang in result.get('translated_language_codes', []):
                translated_language_counts[lang] += 1
            for lang in result.get('error_language_codes', []):
                error_language_counts[lang] += 1
        
        # Calculate language preferences
        total_terms = len(results)
        language_preferences = {}
        
        all_languages = set(same_language_counts.keys()) | set(translated_language_counts.keys())
        
        for lang in all_languages:
            same_count = same_language_counts.get(lang, 0)
            translated_count = translated_language_counts.get(lang, 0)
            total_count = same_count + translated_count
            
            if total_count > 0:
                language_preferences[lang] = {
                    "same_percentage": (same_count / total_count) * 100,
                    "translated_percentage": (translated_count / total_count) * 100,
                    "total_processed": total_count,
                    "coverage_percentage": (total_count / total_terms) * 100
                }
        
        # Find language families/scripts that tend to borrow vs translate
        script_analysis = defaultdict(lambda: {"same": 0, "translated": 0})
        
        for lang in all_languages:
            script = lang.split('_')[-1] if '_' in lang else 'Unknown'
            script_analysis[script]["same"] += same_language_counts.get(lang, 0)
            script_analysis[script]["translated"] += translated_language_counts.get(lang, 0)
        
        script_preferences = {}
        for script, counts in script_analysis.items():
            total = counts["same"] + counts["translated"]
            if total > 0:
                script_preferences[script] = {
                    "same_percentage": (counts["same"] / total) * 100,
                    "translated_percentage": (counts["translated"] / total) * 100,
                    "total_terms": total
                }
        
        return {
            "top_borrowing_languages": dict(same_language_counts.most_common(15)),
            "top_translating_languages": dict(translated_language_counts.most_common(15)),
            "top_error_languages": dict(error_language_counts.most_common(10)),
            "language_preferences": dict(sorted(
                language_preferences.items(), 
                key=lambda x: x[1]["same_percentage"], 
                reverse=True
            )[:20]),
            "script_preferences": dict(sorted(
                script_preferences.items(),
                key=lambda x: x[1]["same_percentage"],
                reverse=True
            ))
        }
    
    def _analyze_frequency_patterns(self, results: List[Dict]) -> Dict:
        """Analyze relationship between term frequency and translatability"""
        
        # Group by frequency ranges
        frequency_ranges = {
            "very_high": (500, float('inf')),
            "high": (100, 499),
            "medium": (20, 99),
            "low": (5, 19),
            "very_low": (1, 4)
        }
        
        frequency_analysis = {}
        
        for range_name, (min_freq, max_freq) in frequency_ranges.items():
            range_results = [
                r for r in results 
                if min_freq <= r.get('frequency', 0) <= max_freq
            ]
            
            if range_results:
                scores = [r.get('translatability_score', 0) for r in range_results]
                frequency_analysis[range_name] = {
                    "count": len(range_results),
                    "frequency_range": f"{min_freq}-{max_freq}",
                    "avg_translatability": np.mean(scores),
                    "median_translatability": np.median(scores),
                    "examples": [
                        {
                            "term": r["term"],
                            "frequency": r["frequency"],
                            "translatability_score": r["translatability_score"]
                        }
                        for r in sorted(range_results, key=lambda x: x["translatability_score"])[:5]
                    ]
                }
        
        return frequency_analysis
    
    def _create_translatability_plots(self, results: List[Dict], output_dir: str, timestamp: str) -> List[str]:
        """Create visualization plots for translatability analysis"""
        
        plot_files = []
        
        # 1. Translatability Score Distribution
        plt.figure(figsize=(12, 8))
        scores = [r.get('translatability_score', 0) for r in results]
        
        plt.subplot(2, 2, 1)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Translatability Scores')
        plt.xlabel('Translatability Score')
        plt.ylabel('Number of Terms')
        plt.grid(True, alpha=0.3)
        
        # 2. Same vs Translated Languages
        plt.subplot(2, 2, 2)
        same_counts = [r.get('same_languages', 0) for r in results]
        translated_counts = [r.get('translated_languages', 0) for r in results]
        
        plt.scatter(same_counts, translated_counts, alpha=0.6)
        plt.title('Same vs Translated Languages')
        plt.xlabel('Languages Keeping Term Same')
        plt.ylabel('Languages Translating Term')
        plt.grid(True, alpha=0.3)
        
        # 3. Frequency vs Translatability
        plt.subplot(2, 2, 3)
        frequencies = [r.get('frequency', 0) for r in results]
        plt.scatter(frequencies, scores, alpha=0.6)
        plt.title('Frequency vs Translatability')
        plt.xlabel('Term Frequency')
        plt.ylabel('Translatability Score')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # 4. Translatability Categories
        plt.subplot(2, 2, 4)
        highly_translatable = len([s for s in scores if s >= 0.8])
        moderately_translatable = len([s for s in scores if 0.3 <= s < 0.8])
        poorly_translatable = len([s for s in scores if s < 0.3])
        
        categories = ['Highly\nTranslatable', 'Moderately\nTranslatable', 'Poorly\nTranslatable']
        counts = [highly_translatable, moderately_translatable, poorly_translatable]
        colors = ['green', 'orange', 'red']
        
        plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.title('Translatability Categories')
        plt.ylabel('Number of Terms')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"translatability_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # 5. Language Script Analysis
        script_data = defaultdict(lambda: {"same": 0, "translated": 0})
        
        for result in results:
            for lang in result.get('same_language_codes', []):
                script = lang.split('_')[-1] if '_' in lang else 'Unknown'
                script_data[script]["same"] += 1
            for lang in result.get('translated_language_codes', []):
                script = lang.split('_')[-1] if '_' in lang else 'Unknown'
                script_data[script]["translated"] += 1
        
        # Filter scripts with significant data
        significant_scripts = {
            script: data for script, data in script_data.items()
            if data["same"] + data["translated"] >= 10
        }
        
        if significant_scripts:
            plt.figure(figsize=(14, 8))
            scripts = list(significant_scripts.keys())
            same_counts = [significant_scripts[script]["same"] for script in scripts]
            translated_counts = [significant_scripts[script]["translated"] for script in scripts]
            
            x = np.arange(len(scripts))
            width = 0.35
            
            plt.bar(x - width/2, same_counts, width, label='Keep Same', alpha=0.8)
            plt.bar(x + width/2, translated_counts, width, label='Translate', alpha=0.8)
            
            plt.title('Language Script Preferences: Borrowing vs Translation')
            plt.xlabel('Script/Writing System')
            plt.ylabel('Number of Terms')
            plt.xticks(x, scripts, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            script_plot_file = os.path.join(output_dir, f"script_analysis_{timestamp}.png")
            plt.savefig(script_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(script_plot_file)
        
        return plot_files
    
    def _generate_recommendations(self, results: List[Dict], summary_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Translatability insights
        avg_score = summary_stats.get('average_translatability_score', 0)
        
        if avg_score < 0.3:
            recommendations.append(
                "Most terms have low translatability. Consider focusing on terms that are "
                "more universally translatable for international products."
            )
        elif avg_score > 0.7:
            recommendations.append(
                "Terms show high translatability across languages. This suggests good "
                "potential for multilingual content adaptation."
            )
        else:
            recommendations.append(
                "Terms show moderate translatability. Review poorly translatable terms "
                "to determine if they should be kept as international standards."
            )
        
        # High-frequency, low-translatability terms
        high_freq_low_trans = [
            r for r in results 
            if r.get('frequency', 0) > 50 and r.get('translatability_score', 0) < 0.3
        ]
        
        if high_freq_low_trans:
            recommendations.append(
                f"Found {len(high_freq_low_trans)} high-frequency terms with low translatability. "
                "These may be technical terms that should remain standardized across languages."
            )
        
        # Error analysis
        avg_errors = summary_stats.get('average_error_languages', 0)
        if avg_errors > 5:
            recommendations.append(
                f"High error rate ({avg_errors:.1f} languages per term on average). "
                "Consider reviewing model performance or input text quality."
            )
        
        # Language coverage
        total_languages = summary_stats.get('total_target_languages', 0)
        if total_languages < 190:
            recommendations.append(
                f"Translation attempted for {total_languages} languages. "
                "Some languages may not be supported or have issues."
            )
        
        return recommendations
    
    def _save_report(self, report: AnalysisReport, report_file: str):
        """Save analysis report to JSON file"""
        
        report_data = {
            "report_info": {
                "type": report.report_type,
                "timestamp": report.timestamp,
                "total_terms": report.total_terms,
                "source_file": self.results_file
            },
            "summary_statistics": report.summary_stats,
            "detailed_analysis": report.detailed_analysis,
            "visualizations": report.visualizations,
            "recommendations": report.recommendations
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_text_summary(self, report: AnalysisReport, output_dir: str, timestamp: str):
        """Generate human-readable text summary"""
        
        summary_file = os.path.join(output_dir, f"translatability_summary_{timestamp}.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("TRANSLATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {report.timestamp}\n")
            f.write(f"Total Terms Analyzed: {report.total_terms:,}\n")
            f.write(f"Source File: {self.results_file}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            stats = report.summary_stats
            f.write(f"Average Translatability Score: {stats['average_translatability_score']:.3f}\n")
            f.write(f"Median Translatability Score: {stats['median_translatability_score']:.3f}\n")
            f.write(f"Standard Deviation: {stats['std_translatability_score']:.3f}\n")
            f.write(f"Range: {stats['min_translatability_score']:.3f} - {stats['max_translatability_score']:.3f}\n\n")
            
            f.write("TRANSLATABILITY CATEGORIES\n")
            f.write("-" * 30 + "\n")
            categories = report.detailed_analysis['translatability_categories']
            for category, data in categories.items():
                f.write(f"{category.replace('_', ' ').title()}: {data['count']:,} terms ({data['percentage']:.1f}%)\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
            
            f.write("VISUALIZATION FILES\n")
            f.write("-" * 30 + "\n")
            for viz_file in report.visualizations:
                f.write(f"• {os.path.basename(viz_file)}\n")
        
        print(f"📄 Text summary saved: {summary_file}")


def analyze_translation_results(results_file: str, output_dir: str = "analysis_reports"):
    """
    Main function to analyze translation results
    
    Args:
        results_file: Path to translation results JSON file
        output_dir: Directory for analysis outputs
    """
    
    print("📊 TRANSLATION RESULTS ANALYSIS")
    print("=" * 50)
    print(f"📁 Results file: {results_file}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Initialize analyzer
        analyzer = TranslationAnalyzer(results_file)
        
        # Generate translatability report
        report = analyzer.generate_translatability_report(output_dir)
        
        print("\n✅ ANALYSIS COMPLETED!")
        print(f"📊 Analyzed {report.total_terms:,} terms")
        print(f"📈 Average translatability: {report.summary_stats['average_translatability_score']:.3f}")
        print(f"📁 Reports saved in: {output_dir}")
        
        return report
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Default to looking for recent results
        results_dir = "translation_results"
        if os.path.exists(results_dir):
            result_files = [
                f for f in os.listdir(results_dir) 
                if f.endswith('.json') and 'translation_results' in f and 'intermediate' not in f
            ]
            if result_files:
                results_file = os.path.join(results_dir, sorted(result_files)[-1])
                print(f"Using most recent results file: {results_file}")
            else:
                print("No translation results files found in translation_results/")
                sys.exit(1)
        else:
            print("No translation_results directory found")
            sys.exit(1)
    
    analyze_translation_results(results_file)

