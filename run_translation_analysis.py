#!/usr/bin/env python3
"""
Main execution script for term translation analysis
Processes term candidates, translates to 200 languages, and generates comprehensive reports
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

from term_translation_processor import process_term_files
from translation_analyzer import analyze_translation_results


def setup_directories():
    """Create necessary directories"""
    directories = [
        "translation_results",
        "analysis_reports", 
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}")


def check_requirements():
    """Check if required files exist"""
    required_files = [
        "Fast_Dictionary_Terms_20250903_123659.json",
        "Fast_Non_Dictionary_Terms_20250903_123659.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    
    print("✅ All required files found")
    return True


def check_gpu_availability():
    """Check GPU availability for acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        else:
            print("💻 GPU not available, using CPU")
            return "cpu"
    except ImportError:
        print("⚠️  PyTorch not installed, using CPU")
        return "cpu"


def run_full_analysis(max_terms_per_file=None, device="auto", skip_processing=False):
    """
    Run complete translation analysis pipeline
    
    Args:
        max_terms_per_file: Maximum terms to process per file (None for all)
        device: Device to use for translation ("auto", "cuda", "cpu")
        skip_processing: Skip translation processing (analyze existing results)
    """
    
    print("🌟 NLLB TRANSLATION ANALYSIS PIPELINE")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration
    dictionary_file = "Fast_Dictionary_Terms_20250903_123659.json"
    non_dictionary_file = "Fast_Non_Dictionary_Terms_20250903_123659.json"
    output_dir = "translation_results"
    analysis_dir = "analysis_reports"
    
    print(f"📅 Run timestamp: {timestamp}")
    print(f"📚 Dictionary terms file: {dictionary_file}")
    print(f"📖 Non-dictionary terms file: {non_dictionary_file}")
    print(f"🔢 Max terms per file: {max_terms_per_file}")
    print(f"🎮 Device: {device}")
    print(f"⏭️  Skip processing: {skip_processing}")
    
    try:
        # Step 1: Setup
        print(f"\n1️⃣ SETUP")
        print("-" * 30)
        setup_directories()
        
        if not check_requirements():
            return False
        
        actual_device = check_gpu_availability() if device == "auto" else device
        
        # Step 2: Translation Processing (if not skipping)
        if not skip_processing:
            print(f"\n2️⃣ TRANSLATION PROCESSING")
            print("-" * 30)
            
            process_term_files(
                dictionary_file=dictionary_file,
                non_dictionary_file=non_dictionary_file,
                output_dir=output_dir,
                max_terms_per_file=max_terms_per_file,
                device=actual_device
            )
            
            print("✅ Translation processing completed")
        else:
            print(f"\n2️⃣ SKIPPING TRANSLATION PROCESSING")
            print("-" * 30)
            print("Looking for existing translation results...")
        
        # Step 3: Find most recent results files
        print(f"\n3️⃣ ANALYSIS PREPARATION")
        print("-" * 30)
        
        # Find result files
        result_files = []
        if os.path.exists(output_dir):
            for file_name in os.listdir(output_dir):
                if (file_name.endswith('.json') and 
                    'translation_results' in file_name and 
                    'intermediate' not in file_name and
                    'summary' not in file_name):
                    result_files.append(os.path.join(output_dir, file_name))
        
        if not result_files:
            print("❌ No translation result files found!")
            return False
        
        result_files.sort(key=os.path.getmtime, reverse=True)  # Most recent first
        print(f"📊 Found {len(result_files)} result files")
        
        # Step 4: Analysis and Reporting
        print(f"\n4️⃣ ANALYSIS AND REPORTING")
        print("-" * 30)
        
        analysis_results = []
        
        for i, result_file in enumerate(result_files, 1):
            print(f"\n📊 Analyzing file {i}/{len(result_files)}: {os.path.basename(result_file)}")
            
            try:
                report = analyze_translation_results(result_file, analysis_dir)
                if report:
                    analysis_results.append({
                        "file": result_file,
                        "report": report,
                        "file_type": "dictionary" if "dictionary" in result_file else "non_dictionary"
                    })
                    print(f"✅ Analysis completed for {os.path.basename(result_file)}")
                else:
                    print(f"❌ Analysis failed for {os.path.basename(result_file)}")
                    
            except Exception as e:
                print(f"❌ Analysis error for {os.path.basename(result_file)}: {e}")
        
        # Step 5: Generate Combined Summary
        print(f"\n5️⃣ COMBINED SUMMARY")
        print("-" * 30)
        
        if analysis_results:
            generate_combined_summary(analysis_results, analysis_dir, timestamp)
            print("✅ Combined summary generated")
        else:
            print("❌ No successful analyses to summarize")
        
        # Step 6: Final Report
        print(f"\n6️⃣ FINAL REPORT")
        print("-" * 30)
        
        print(f"🎉 ANALYSIS PIPELINE COMPLETED!")
        print(f"📁 Translation results: {output_dir}")
        print(f"📊 Analysis reports: {analysis_dir}")
        print(f"📈 Processed {len(analysis_results)} result files")
        
        # Show key findings
        if analysis_results:
            print(f"\n🔍 KEY FINDINGS:")
            for result in analysis_results:
                file_type = result["file_type"]
                report = result["report"]
                avg_score = report.summary_stats.get('average_translatability_score', 0)
                total_terms = report.total_terms
                
                print(f"   • {file_type.title()} Terms: {total_terms:,} terms analyzed")
                print(f"     Average translatability: {avg_score:.3f}")
                
                categories = report.detailed_analysis.get('translatability_categories', {})
                if categories:
                    highly = categories.get('highly_translatable', {}).get('count', 0)
                    poorly = categories.get('poorly_translatable', {}).get('count', 0)
                    print(f"     Highly translatable: {highly:,} ({(highly/total_terms)*100:.1f}%)")
                    print(f"     Poorly translatable: {poorly:,} ({(poorly/total_terms)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_combined_summary(analysis_results, output_dir, timestamp):
    """Generate a combined summary of all analyses"""
    
    combined_summary = {
        "summary_info": {
            "timestamp": timestamp,
            "total_files_analyzed": len(analysis_results),
            "analysis_types": [result["file_type"] for result in analysis_results]
        },
        "combined_statistics": {},
        "comparative_analysis": {},
        "overall_findings": []
    }
    
    # Aggregate statistics
    total_terms = sum(result["report"].total_terms for result in analysis_results)
    avg_translatability_scores = []
    
    file_summaries = {}
    
    for result in analysis_results:
        file_type = result["file_type"]
        report = result["report"]
        
        file_summaries[file_type] = {
            "total_terms": report.total_terms,
            "avg_translatability": report.summary_stats.get('average_translatability_score', 0),
            "categories": report.detailed_analysis.get('translatability_categories', {}),
            "top_recommendations": report.recommendations[:3]
        }
        
        avg_translatability_scores.append(report.summary_stats.get('average_translatability_score', 0))
    
    combined_summary["combined_statistics"] = {
        "total_terms_analyzed": total_terms,
        "overall_avg_translatability": sum(avg_translatability_scores) / len(avg_translatability_scores) if avg_translatability_scores else 0,
        "file_summaries": file_summaries
    }
    
    # Comparative analysis
    if len(analysis_results) >= 2:
        dict_result = next((r for r in analysis_results if r["file_type"] == "dictionary"), None)
        non_dict_result = next((r for r in analysis_results if r["file_type"] == "non_dictionary"), None)
        
        if dict_result and non_dict_result:
            dict_score = dict_result["report"].summary_stats.get('average_translatability_score', 0)
            non_dict_score = non_dict_result["report"].summary_stats.get('average_translatability_score', 0)
            
            combined_summary["comparative_analysis"] = {
                "dictionary_vs_non_dictionary": {
                    "dictionary_avg_translatability": dict_score,
                    "non_dictionary_avg_translatability": non_dict_score,
                    "difference": abs(dict_score - non_dict_score),
                    "interpretation": (
                        "Dictionary terms are more translatable" if dict_score > non_dict_score 
                        else "Non-dictionary terms are more translatable" if non_dict_score > dict_score
                        else "Similar translatability"
                    )
                }
            }
    
    # Overall findings
    findings = []
    
    if total_terms > 0:
        overall_avg = combined_summary["combined_statistics"]["overall_avg_translatability"]
        
        if overall_avg > 0.7:
            findings.append("Terms show high overall translatability across languages")
        elif overall_avg < 0.3:
            findings.append("Terms show low overall translatability - many may be technical/borrowed terms")
        else:
            findings.append("Terms show moderate translatability with mixed patterns")
        
        # Add findings from individual analyses
        for result in analysis_results:
            key_recommendations = result["report"].recommendations[:2]
            findings.extend([f"{result['file_type'].title()}: {rec}" for rec in key_recommendations])
    
    combined_summary["overall_findings"] = findings
    
    # Save combined summary
    summary_file = os.path.join(output_dir, f"combined_analysis_summary_{timestamp}.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, indent=2, ensure_ascii=False)
    
    # Generate text summary
    text_summary_file = os.path.join(output_dir, f"combined_summary_{timestamp}.txt")
    
    with open(text_summary_file, 'w', encoding='utf-8') as f:
        f.write("COMBINED TRANSLATION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Total Terms Analyzed: {total_terms:,}\n")
        f.write(f"Files Analyzed: {len(analysis_results)}\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Average Translatability: {combined_summary['combined_statistics']['overall_avg_translatability']:.3f}\n\n")
        
        for file_type, summary in file_summaries.items():
            f.write(f"{file_type.upper()} TERMS\n")
            f.write(f"Terms: {summary['total_terms']:,}\n")
            f.write(f"Avg Translatability: {summary['avg_translatability']:.3f}\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 30 + "\n")
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n")
    
    print(f"📄 Combined summary saved: {summary_file}")
    print(f"📄 Text summary saved: {text_summary_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NLLB Translation Analysis Pipeline")
    parser.add_argument("--max-terms", type=int, default=None, 
                       help="Maximum terms to process per file (default: all)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use for translation (default: auto)")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip translation processing, only run analysis on existing results")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with limited terms (100 per file)")
    
    args = parser.parse_args()
    
    # Test mode override
    if args.test_mode:
        args.max_terms = 100
        print("🧪 Running in test mode (100 terms per file)")
    
    success = run_full_analysis(
        max_terms_per_file=args.max_terms,
        device=args.device,
        skip_processing=args.skip_processing
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
