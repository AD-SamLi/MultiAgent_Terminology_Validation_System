#!/usr/bin/env python3
"""
Enhanced Translation Analysis Runner
Supports resumable processing, parallel translation, and session management
"""

import os
import sys
import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

from enhanced_translation_processor import (
    EnhancedTermTranslationProcessor, 
    ParallelConfig, 
    process_with_resume
)
from translation_analyzer import analyze_translation_results


def list_available_sessions():
    """List all available sessions for resuming"""
    print("📋 AVAILABLE SESSIONS")
    print("=" * 50)
    
    processor = EnhancedTermTranslationProcessor()
    sessions = processor.list_sessions()
    
    if not sessions:
        print("❌ No sessions found")
        return
    
    for i, session in enumerate(sessions, 1):
        print(f"{i}. Session: {session['session_id']}")
        print(f"   Type: {session['file_type']}")
        print(f"   Progress: {session['progress']}")
        print(f"   Last Update: {session['last_update']}")
        print()


def setup_directories():
    """Create necessary directories"""
    directories = [
        "translation_results",
        "analysis_reports", 
        "logs",
        "checkpoints"
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
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        else:
            print("💻 GPU not available, using CPU")
            return "cpu"
    except ImportError:
        print("⚠️  PyTorch not installed, using CPU")
        return "cpu"


def run_enhanced_analysis(max_workers: int = None, chunk_size: int = 5, 
                         device: str = "auto", model_size: str = "small",
                         resume_session: str = None, skip_processing: bool = False,
                         use_multiprocessing: bool = True):
    """
    Run enhanced translation analysis with parallel processing and resuming
    
    Args:
        max_workers: Maximum number of parallel workers
        chunk_size: Number of terms per processing chunk
        device: Device to use for translation
        model_size: Model size to use
        resume_session: Session ID to resume from
        skip_processing: Skip translation processing, only run analysis
        use_multiprocessing: Use multiprocessing vs threading
    """
    
    print("🌟 ENHANCED NLLB TRANSLATION ANALYSIS")
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
    print(f"🔄 Resume session: {resume_session or 'New session'}")
    print(f"👥 Max workers: {max_workers}")
    print(f"📦 Chunk size: {chunk_size}")
    print(f"🎮 Device: {device}")
    print(f"🧠 Model size: {model_size}")
    print(f"⚡ Use multiprocessing: {use_multiprocessing}")
    print(f"⏭️  Skip processing: {skip_processing}")
    
    try:
        # Step 1: Setup
        print(f"\n1️⃣ SETUP")
        print("-" * 30)
        setup_directories()
        
        if not check_requirements():
            return False
        
        actual_device = check_gpu_availability() if device == "auto" else device
        
        # Determine optimal worker count
        if max_workers is None:
            cpu_count = mp.cpu_count()
            if actual_device == "cuda":
                # For GPU, use fewer workers to avoid memory conflicts
                max_workers = min(4, cpu_count // 2)
            else:
                # For CPU, can use more workers
                max_workers = min(cpu_count, 8)
        
        print(f"🔧 Using {max_workers} parallel workers")
        
        # Step 2: Translation Processing (if not skipping)
        if not skip_processing:
            print(f"\n2️⃣ ENHANCED PARALLEL TRANSLATION PROCESSING")
            print("-" * 30)
            
            # Configure parallel processing
            parallel_config = ParallelConfig(
                max_workers=max_workers,
                batch_size=8,
                chunk_size=chunk_size,
                use_multiprocessing=use_multiprocessing
            )
            
            # Run enhanced processing
            process_with_resume(
                dictionary_file=dictionary_file,
                non_dictionary_file=non_dictionary_file,
                output_dir=output_dir,
                parallel_config=parallel_config,
                resume_session=resume_session,
                device=actual_device,
                model_size=model_size
            )
            
            print("✅ Enhanced translation processing completed")
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
        
        # Step 5: Final Report
        print(f"\n5️⃣ FINAL REPORT")
        print("-" * 30)
        
        print(f"🎉 ENHANCED ANALYSIS COMPLETED!")
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
        
        # Show session management info
        print(f"\n📋 SESSION MANAGEMENT:")
        print(f"   • Checkpoints saved in: checkpoints/")
        print(f"   • Use --resume <session_id> to resume interrupted processing")
        print(f"   • Use --list-sessions to see available sessions")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point with enhanced options"""
    parser = argparse.ArgumentParser(description="Enhanced NLLB Translation Analysis Pipeline")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=5,
                       help="Number of terms per processing chunk (default: 5)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use for translation (default: auto)")
    parser.add_argument("--model-size", choices=["small", "medium", "1.3B", "3.3B"], default="small",
                       help="Model size to use (default: small)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Session ID to resume from")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip translation processing, only run analysis on existing results")
    parser.add_argument("--list-sessions", action="store_true",
                       help="List available sessions for resuming")
    parser.add_argument("--use-threading", action="store_true",
                       help="Use threading instead of multiprocessing")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with limited parallel workers")
    
    args = parser.parse_args()
    
    # List sessions if requested
    if args.list_sessions:
        list_available_sessions()
        return
    
    # Test mode adjustments
    if args.test_mode:
        args.max_workers = 2
        args.chunk_size = 3
        print("🧪 Running in test mode (limited parallelism)")
    
    success = run_enhanced_analysis(
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        device=args.device,
        model_size=args.model_size,
        resume_session=args.resume,
        skip_processing=args.skip_processing,
        use_multiprocessing=not args.use_threading
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

