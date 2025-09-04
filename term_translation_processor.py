#!/usr/bin/env python3
"""
Term Translation Processor
Processes term candidate files and translates them to all 200 NLLB languages
Analyzes translatability patterns and generates comprehensive reports
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, Counter

from nllb_translation_tool import NLLBTranslationTool, TranslationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TermAnalysisResult:
    """Container for term analysis results"""
    term: str
    frequency: int
    total_languages: int
    same_languages: int
    translated_languages: int
    error_languages: int
    translatability_score: float  # 0-1, higher = more translatable
    same_language_codes: List[str]
    translated_language_codes: List[str]
    error_language_codes: List[str]
    sample_translations: Dict[str, str]  # lang_code -> translation
    analysis_timestamp: str
    processing_time_seconds: float

@dataclass
class ProcessingStats:
    """Container for processing statistics"""
    total_terms: int
    processed_terms: int
    failed_terms: int
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    terms_per_second: float = 0.0

class TermTranslationProcessor:
    """
    Main processor for translating term candidates to all NLLB languages
    """
    
    def __init__(self, device: str = "auto", batch_size: int = 8, max_workers: int = 2, model_size: str = "small"):
        """
        Initialize the term translation processor
        
        Args:
            device: Device for NLLB model ("auto", "cuda", "cpu")
            batch_size: Batch size for translation operations
            max_workers: Maximum number of worker threads for parallel processing
            model_size: Model size ("small", "medium", "1.3B", "3.3B")
        """
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model_size = model_size
        self.source_lang = "eng_Latn"  # Assuming English source
        
        # Initialize translation tool
        print("🔧 Initializing NLLB Translation Tool...")
        print(f"📋 Model size: {model_size}")
        self.translator = NLLBTranslationTool(model_name=model_size, device=device, batch_size=batch_size)
        
        # Get available languages (excluding source language)
        all_languages = self.translator.get_available_languages()
        self.target_languages = [lang for lang in all_languages if lang != self.source_lang]
        
        print(f"✅ Processor initialized")
        print(f"🎯 Target languages: {len(self.target_languages)}")
        print(f"🔢 Batch size: {batch_size}")
        print(f"👥 Max workers: {max_workers}")
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.processed_count = 0
        self.failed_count = 0
    
    def load_term_candidates(self, file_path: str) -> List[Dict]:
        """Load term candidates from JSON file"""
        print(f"📁 Loading term candidates from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract terms from the JSON structure
            if 'dictionary_terms' in data:
                terms = data['dictionary_terms']
                term_type = 'dictionary'
            elif 'non_dictionary_terms' in data:
                terms = data['non_dictionary_terms']
                term_type = 'non_dictionary'
            else:
                raise ValueError("Unknown JSON structure - expected 'dictionary_terms' or 'non_dictionary_terms'")
            
            print(f"✅ Loaded {len(terms)} {term_type} terms")
            print(f"📊 Analysis info: {data.get('analysis_info', {})}")
            
            return terms
            
        except Exception as e:
            print(f"❌ Failed to load terms from {file_path}: {e}")
            raise
    
    def analyze_term_translatability(self, term_data: Dict) -> TermAnalysisResult:
        """
        Analyze translatability of a single term across all languages
        
        Args:
            term_data: Dictionary containing term information
            
        Returns:
            TermAnalysisResult object
        """
        term = term_data.get('term', '')
        frequency = term_data.get('frequency', 0)
        
        if not term or not term.strip():
            return TermAnalysisResult(
                term=term,
                frequency=frequency,
                total_languages=0,
                same_languages=0,
                translated_languages=0,
                error_languages=0,
                translatability_score=0.0,
                same_language_codes=[],
                translated_language_codes=[],
                error_language_codes=[],
                sample_translations={},
                analysis_timestamp=datetime.now().isoformat(),
                processing_time_seconds=0.0
            )
        
        start_time = time.time()
        
        try:
            # Translate to all target languages
            results = self.translator.translate_to_all_languages(term, self.source_lang)
            
            # Analyze results
            same_languages = []
            translated_languages = []
            error_languages = []
            sample_translations = {}
            
            for lang_code, result in results.items():
                if result.error:
                    error_languages.append(lang_code)
                elif result.is_same:
                    same_languages.append(lang_code)
                else:
                    translated_languages.append(lang_code)
                    # Keep some sample translations (max 10 for storage efficiency)
                    if len(sample_translations) < 10:
                        sample_translations[lang_code] = result.translated_text
            
            # Calculate translatability score
            total_valid = len(same_languages) + len(translated_languages)
            translatability_score = len(translated_languages) / total_valid if total_valid > 0 else 0.0
            
            processing_time = time.time() - start_time
            
            # Update thread-safe counters
            with self.lock:
                self.processed_count += 1
            
            return TermAnalysisResult(
                term=term,
                frequency=frequency,
                total_languages=len(results),
                same_languages=len(same_languages),
                translated_languages=len(translated_languages),
                error_languages=len(error_languages),
                translatability_score=translatability_score,
                same_language_codes=same_languages,
                translated_language_codes=translated_languages,
                error_language_codes=error_languages,
                sample_translations=sample_translations,
                analysis_timestamp=datetime.now().isoformat(),
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            
            logger.error(f"Failed to analyze term '{term}': {e}")
            
            return TermAnalysisResult(
                term=term,
                frequency=frequency,
                total_languages=0,
                same_languages=0,
                translated_languages=0,
                error_languages=0,
                translatability_score=0.0,
                same_language_codes=[],
                translated_language_codes=[],
                error_language_codes=[],
                sample_translations={},
                analysis_timestamp=datetime.now().isoformat(),
                processing_time_seconds=time.time() - start_time
            )
    
    def process_terms_batch(self, terms: List[Dict], output_file: str, 
                           start_index: int = 0, max_terms: Optional[int] = None) -> ProcessingStats:
        """
        Process a batch of terms for translation analysis
        
        Args:
            terms: List of term dictionaries
            output_file: Output file path for results
            start_index: Starting index for processing (for resuming)
            max_terms: Maximum number of terms to process (None for all)
            
        Returns:
            ProcessingStats object
        """
        # Determine terms to process
        terms_to_process = terms[start_index:]
        if max_terms:
            terms_to_process = terms_to_process[:max_terms]
        
        total_terms = len(terms_to_process)
        
        print(f"🚀 Starting batch processing of {total_terms} terms")
        print(f"📁 Output file: {output_file}")
        print(f"🎯 Processing terms {start_index} to {start_index + total_terms - 1}")
        
        # Initialize stats
        stats = ProcessingStats(
            total_terms=total_terms,
            processed_terms=0,
            failed_terms=0,
            start_time=datetime.now()
        )
        
        # Reset counters
        with self.lock:
            self.processed_count = 0
            self.failed_count = 0
        
        results = []
        
        # Process terms with progress tracking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_term = {
                executor.submit(self.analyze_term_translatability, term): term 
                for term in terms_to_process
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_term), 1):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress reporting
                    if i % 10 == 0 or i == total_terms:
                        elapsed = (datetime.now() - stats.start_time).total_seconds()
                        rate = i / elapsed if elapsed > 0 else 0
                        eta = (total_terms - i) / rate if rate > 0 else 0
                        
                        print(f"📊 Progress: {i}/{total_terms} ({i/total_terms*100:.1f}%) | "
                              f"Rate: {rate:.1f} terms/sec | ETA: {eta/60:.1f} min")
                    
                    # Save intermediate results every 50 terms
                    if i % 50 == 0:
                        self._save_intermediate_results(results, output_file, i, total_terms)
                        
                except Exception as e:
                    logger.error(f"Failed to process term: {e}")
                    with self.lock:
                        self.failed_count += 1
        
        # Final statistics
        stats.end_time = datetime.now()
        stats.processing_time_seconds = (stats.end_time - stats.start_time).total_seconds()
        stats.processed_terms = self.processed_count
        stats.failed_terms = self.failed_count
        stats.terms_per_second = stats.processed_terms / stats.processing_time_seconds if stats.processing_time_seconds > 0 else 0
        
        # Save final results
        self._save_final_results(results, output_file, stats)
        
        print(f"✅ Batch processing completed!")
        print(f"📊 Processed: {stats.processed_terms}/{stats.total_terms} terms")
        print(f"❌ Failed: {stats.failed_terms} terms")
        print(f"⏱️  Total time: {stats.processing_time_seconds/60:.1f} minutes")
        print(f"🚀 Rate: {stats.terms_per_second:.2f} terms/second")
        
        return stats
    
    def _save_intermediate_results(self, results: List[TermAnalysisResult], 
                                 output_file: str, current_count: int, total_count: int):
        """Save intermediate results to prevent data loss"""
        intermediate_file = f"{output_file}.intermediate_{current_count}"
        
        try:
            output_data = {
                "processing_info": {
                    "status": "in_progress",
                    "processed_terms": current_count,
                    "total_terms": total_count,
                    "progress_percentage": (current_count / total_count) * 100,
                    "timestamp": datetime.now().isoformat(),
                    "source_language": self.source_lang,
                    "target_languages_count": len(self.target_languages)
                },
                "results": [asdict(result) for result in results]
            }
            
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved intermediate results to {intermediate_file}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def _save_final_results(self, results: List[TermAnalysisResult], 
                          output_file: str, stats: ProcessingStats):
        """Save final results with comprehensive analysis"""
        
        # Generate analysis summary
        analysis_summary = self._generate_analysis_summary(results)
        
        output_data = {
            "processing_info": {
                "status": "completed",
                "total_terms": stats.total_terms,
                "processed_terms": stats.processed_terms,
                "failed_terms": stats.failed_terms,
                "processing_time_seconds": stats.processing_time_seconds,
                "terms_per_second": stats.terms_per_second,
                "start_time": stats.start_time.isoformat(),
                "end_time": stats.end_time.isoformat() if stats.end_time else None,
                "source_language": self.source_lang,
                "target_languages_count": len(self.target_languages),
                "device_used": self.device,
                "batch_size": self.batch_size
            },
            "analysis_summary": analysis_summary,
            "results": [asdict(result) for result in results]
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Final results saved to: {output_file}")
            
            # Also save a summary report
            summary_file = output_file.replace('.json', '_summary.json')
            summary_data = {
                "processing_info": output_data["processing_info"],
                "analysis_summary": analysis_summary
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary report saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    def _generate_analysis_summary(self, results: List[TermAnalysisResult]) -> Dict:
        """Generate comprehensive analysis summary"""
        if not results:
            return {}
        
        # Basic statistics
        total_results = len(results)
        translatability_scores = [r.translatability_score for r in results if r.total_languages > 0]
        
        if not translatability_scores:
            return {"error": "No valid results to analyze"}
        
        # Translatability distribution
        highly_translatable = len([s for s in translatability_scores if s >= 0.8])
        moderately_translatable = len([s for s in translatability_scores if 0.3 <= s < 0.8])
        poorly_translatable = len([s for s in translatability_scores if s < 0.3])
        
        # Language analysis
        same_language_counter = Counter()
        translated_language_counter = Counter()
        error_language_counter = Counter()
        
        for result in results:
            for lang in result.same_language_codes:
                same_language_counter[lang] += 1
            for lang in result.translated_language_codes:
                translated_language_counter[lang] += 1
            for lang in result.error_language_codes:
                error_language_counter[lang] += 1
        
        # Most/least translatable terms
        sorted_results = sorted(results, key=lambda x: x.translatability_score, reverse=True)
        most_translatable = [
            {"term": r.term, "score": r.translatability_score, "frequency": r.frequency}
            for r in sorted_results[:10]
        ]
        least_translatable = [
            {"term": r.term, "score": r.translatability_score, "frequency": r.frequency}
            for r in sorted_results[-10:]
        ]
        
        return {
            "overview": {
                "total_terms_analyzed": total_results,
                "average_translatability_score": sum(translatability_scores) / len(translatability_scores),
                "median_translatability_score": sorted(translatability_scores)[len(translatability_scores)//2],
                "highly_translatable_terms": highly_translatable,
                "moderately_translatable_terms": moderately_translatable,
                "poorly_translatable_terms": poorly_translatable
            },
            "translatability_distribution": {
                "highly_translatable_percent": (highly_translatable / total_results) * 100,
                "moderately_translatable_percent": (moderately_translatable / total_results) * 100,
                "poorly_translatable_percent": (poorly_translatable / total_results) * 100
            },
            "language_analysis": {
                "top_languages_keeping_terms_same": dict(same_language_counter.most_common(10)),
                "top_languages_translating_terms": dict(translated_language_counter.most_common(10)),
                "top_languages_with_errors": dict(error_language_counter.most_common(10))
            },
            "term_examples": {
                "most_translatable": most_translatable,
                "least_translatable": least_translatable
            }
        }


def process_term_files(dictionary_file: str, non_dictionary_file: str, 
                      output_dir: str = "translation_results", 
                      max_terms_per_file: Optional[int] = None,
                      device: str = "auto",
                      model_size: str = "small"):
    """
    Process both dictionary and non-dictionary term files
    
    Args:
        dictionary_file: Path to dictionary terms JSON file
        non_dictionary_file: Path to non-dictionary terms JSON file
        output_dir: Directory for output files
        max_terms_per_file: Maximum terms to process per file (None for all)
        device: Device for translation model
        model_size: Model size ("small", "medium", "1.3B", "3.3B")
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize processor
    processor = TermTranslationProcessor(device=device, batch_size=8, max_workers=2, model_size=model_size)
    
    # Process dictionary terms
    if os.path.exists(dictionary_file):
        print(f"\n📚 PROCESSING DICTIONARY TERMS")
        print("=" * 50)
        
        dictionary_terms = processor.load_term_candidates(dictionary_file)
        dict_output_file = os.path.join(output_dir, f"dictionary_terms_translation_results_{timestamp}.json")
        
        dict_stats = processor.process_terms_batch(
            dictionary_terms, 
            dict_output_file,
            max_terms=max_terms_per_file
        )
        
        print(f"✅ Dictionary terms processing completed")
    
    # Process non-dictionary terms  
    if os.path.exists(non_dictionary_file):
        print(f"\n📖 PROCESSING NON-DICTIONARY TERMS")
        print("=" * 50)
        
        non_dict_terms = processor.load_term_candidates(non_dictionary_file)
        non_dict_output_file = os.path.join(output_dir, f"non_dictionary_terms_translation_results_{timestamp}.json")
        
        non_dict_stats = processor.process_terms_batch(
            non_dict_terms,
            non_dict_output_file, 
            max_terms=max_terms_per_file
        )
        
        print(f"✅ Non-dictionary terms processing completed")
    
    print(f"\n🎉 ALL PROCESSING COMPLETED!")
    print(f"📁 Results saved in: {output_dir}")


if __name__ == "__main__":
    # Configuration
    DICTIONARY_FILE = "Fast_Dictionary_Terms_20250903_123659.json"
    NON_DICTIONARY_FILE = "Fast_Non_Dictionary_Terms_20250903_123659.json"
    OUTPUT_DIR = "translation_results"
    
    # For testing, limit to first 100 terms per file
    # Set to None to process all terms
    MAX_TERMS_PER_FILE = 100  # Change to None for full processing
    
    print("🌟 TERM TRANSLATION PROCESSOR")
    print("=" * 60)
    print(f"📚 Dictionary file: {DICTIONARY_FILE}")
    print(f"📖 Non-dictionary file: {NON_DICTIONARY_FILE}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🔢 Max terms per file: {MAX_TERMS_PER_FILE}")
    
    try:
        process_term_files(
            dictionary_file=DICTIONARY_FILE,
            non_dictionary_file=NON_DICTIONARY_FILE,
            output_dir=OUTPUT_DIR,
            max_terms_per_file=MAX_TERMS_PER_FILE,
            device="auto"  # Will use GPU if available
        )
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
