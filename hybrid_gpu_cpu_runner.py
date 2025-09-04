#!/usr/bin/env python3
"""
Hybrid GPU-CPU Translation Runner
Uses GPU for translation pipeline while CPU handles term processing in parallel
This approach maximizes both GPU utilization and CPU throughput
"""

import os
import sys
import json
import time
import threading
import queue
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from nllb_translation_tool import NLLBTranslationTool, TranslationResult
from translation_analyzer import analyze_translation_results


class HybridGPUCPURunner:
    """
    Hybrid runner that uses GPU for translation while CPU processes terms in parallel
    """
    
    def __init__(self, device: str = "auto", model_size: str = "auto", 
                 cpu_workers: int = 4, checkpoint_interval: int = 100):
        """
        Initialize hybrid runner
        
        Args:
            device: Device for GPU translation
            model_size: Model size to use ("auto" for maximum that fits GPU)
            cpu_workers: Number of CPU workers for term processing
            checkpoint_interval: Save checkpoint every N terms
        """
        self.device = device
        self.model_size = self._determine_optimal_model_size(model_size)
        self.cpu_workers = cpu_workers
        self.checkpoint_interval = checkpoint_interval
        self.source_lang = "eng_Latn"
        
        # Thread-safe queues for communication
        self.term_queue = queue.Queue(maxsize=200)  # Terms to process
        self.result_queue = queue.Queue()  # Completed results
        self.checkpoint_queue = queue.Queue()  # Checkpointing data
        
        # Synchronization
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        
        # Session management
        self.session_id = self._generate_session_id()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print("🔧 Initializing Hybrid GPU-CPU Translation Runner...")
        print(f"📋 Session ID: {self.session_id}")
        print(f"🎮 GPU Model: {model_size}")
        print(f"👥 CPU Workers: {cpu_workers}")
        print(f"💾 Checkpoint interval: {checkpoint_interval}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"hybrid_{timestamp}"
    
    def _initialize_gpu_translator(self):
        """Initialize the GPU translation pipeline"""
        print("🎮 Initializing GPU translation pipeline...")
        
        # Determine optimal batch size based on model size
        batch_sizes = {
            "small": 16,
            "medium": 12,
            "1.3B": 10,
            "3.3B": 6
        }
        
        optimal_batch_size = batch_sizes.get(self.model_size, 8)
        print(f"🔢 Using batch size: {optimal_batch_size} for {self.model_size} model")
        
        self.gpu_translator = NLLBTranslationTool(
            model_name=self.model_size, 
            device=self.device, 
            batch_size=optimal_batch_size
        )
        
        # Get target languages
        all_languages = self.gpu_translator.get_available_languages()
        self.target_languages = [lang for lang in all_languages if lang != self.source_lang]
        
        print(f"✅ GPU translator ready with {len(self.target_languages)} target languages")
    
    def _gpu_translation_worker(self):
        """
        GPU worker thread - handles all translation operations
        Processes terms from queue and puts results back
        """
        print("🎮 Starting GPU translation worker...")
        
        try:
            self._initialize_gpu_translator()
            
            while not self.stop_event.is_set():
                try:
                    # Get term from queue (timeout to check stop event)
                    term_data = self.term_queue.get(timeout=1.0)
                    
                    if term_data is None:  # Sentinel value to stop
                        break
                    
                    # Extract term info
                    term_index, term, frequency = term_data
                    
                    if not term or not term.strip():
                        # Handle empty term
                        result = self._create_empty_result(term, frequency, "Empty term")
                        self.result_queue.put((term_index, result))
                        self.term_queue.task_done()
                        continue
                    
                    start_time = time.time()
                    
                    try:
                        # GPU translation - this is the expensive operation
                        translation_results = self.gpu_translator.translate_to_all_languages(term, self.source_lang)
                        
                        # Analyze results (CPU work)
                        analysis_result = self._analyze_translation_results(
                            term, frequency, translation_results, start_time
                        )
                        
                        self.result_queue.put((term_index, analysis_result))
                        
                        # Update stats
                        with self.stats_lock:
                            self.processed_count += 1
                        
                    except Exception as e:
                        # Handle translation error
                        error_result = self._create_empty_result(term, frequency, str(e))
                        self.result_queue.put((term_index, error_result))
                        
                        with self.stats_lock:
                            self.failed_count += 1
                    
                    self.term_queue.task_done()
                    
                except queue.Empty:
                    continue  # Check stop event and try again
                    
        except Exception as e:
            print(f"❌ GPU worker error: {e}")
        
        print("🎮 GPU translation worker stopped")
    
    def _determine_optimal_model_size(self, requested_size: str) -> str:
        """
        Determine the optimal model size based on available GPU memory
        
        Args:
            requested_size: User requested size or "auto"
            
        Returns:
            Optimal model size string
        """
        if requested_size != "auto":
            return requested_size
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                print("💻 No CUDA available, using smallest model for CPU")
                return "small"
            
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU Memory Available: {gpu_memory_gb:.1f} GB")
            
            # Model memory requirements (approximate, including overhead)
            model_memory_requirements = {
                "small": 2.8,      # facebook/nllb-200-1.3B
                "medium": 4.5,     # facebook/nllb-200-1.3B with larger batch
                "1.3B": 4.8,       # facebook/nllb-200-1.3B optimized
                "3.3B": 7.5        # facebook/nllb-200-3.3B
            }
            
            # Leave 1GB buffer for other operations
            available_memory = gpu_memory_gb - 1.0
            
            # Find largest model that fits
            optimal_size = "small"  # fallback
            
            for size, memory_req in sorted(model_memory_requirements.items(), 
                                         key=lambda x: x[1], reverse=True):
                if memory_req <= available_memory:
                    optimal_size = size
                    print(f"✅ Selected model size: {size} (requires {memory_req}GB)")
                    break
                else:
                    print(f"⚠️  Skipping {size} model (requires {memory_req}GB > {available_memory:.1f}GB available)")
            
            return optimal_size
            
        except Exception as e:
            print(f"❌ Error determining model size: {e}")
            print("🔄 Falling back to small model")
            return "small"
    
    def _analyze_translation_results(self, term: str, frequency: int, 
                                   translation_results: Dict, start_time: float) -> Dict:
        """
        Analyze translation results (CPU-intensive work)
        """
        same_languages = []
        translated_languages = []
        error_languages = []
        sample_translations = {}
        
        for lang_code, result in translation_results.items():
            if result.error:
                error_languages.append(lang_code)
            elif result.is_same:
                same_languages.append(lang_code)
            else:
                translated_languages.append(lang_code)
                # Keep sample translations (limit to 10 for storage)
                if len(sample_translations) < 10:
                    sample_translations[lang_code] = result.translated_text
        
        # Calculate translatability score
        total_valid = len(same_languages) + len(translated_languages)
        translatability_score = len(translated_languages) / total_valid if total_valid > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        return {
            "term": term,
            "frequency": frequency,
            "total_languages": len(translation_results),
            "same_languages": len(same_languages),
            "translated_languages": len(translated_languages),
            "error_languages": len(error_languages),
            "translatability_score": translatability_score,
            "same_language_codes": same_languages,
            "translated_language_codes": translated_languages,
            "error_language_codes": error_languages,
            "sample_translations": sample_translations,
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time
        }
    
    def _create_empty_result(self, term: str, frequency: int, error_msg: str) -> Dict:
        """Create empty result for failed/empty terms"""
        return {
            "term": term,
            "frequency": frequency,
            "total_languages": 0,
            "same_languages": 0,
            "translated_languages": 0,
            "error_languages": 0,
            "translatability_score": 0.0,
            "same_language_codes": [],
            "translated_language_codes": [],
            "error_language_codes": [],
            "sample_translations": {},
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time_seconds": 0.0,
            "error": error_msg
        }
    
    def _cpu_coordinator_worker(self, terms: List[Dict], output_file: str, file_type: str, 
                                  start_index: int = 0, existing_results: List[Dict] = None):
        """
        CPU coordinator - manages term feeding and result collection
        Runs in separate thread to coordinate the pipeline
        """
        print(f"👥 Starting CPU coordinator for {len(terms)} terms...")
        
        # Results storage - initialize with existing results if resuming
        all_results = [None] * len(terms)  # Pre-allocate with indices
        
        if existing_results:
            for i, result in enumerate(existing_results):
                if i < len(all_results):
                    all_results[i] = result
            print(f"📂 Loaded {len(existing_results)} existing results")
        
        # Start feeding terms to GPU (skip already processed ones)
        def feed_terms():
            for i, term_data in enumerate(terms):
                if i < start_index:  # Skip already processed terms
                    continue
                    
                term = term_data.get('term', '')
                frequency = term_data.get('frequency', 0)
                self.term_queue.put((i, term, frequency))
            
            # Signal end of terms
            self.term_queue.put(None)
        
        # Start term feeder thread
        feeder_thread = threading.Thread(target=feed_terms)
        feeder_thread.start()
        
        # Collect results
        completed_results = len(existing_results) if existing_results else 0
        last_checkpoint = completed_results
        
        while completed_results < len(terms):
            try:
                # Get result from GPU worker
                term_index, result = self.result_queue.get(timeout=5.0)
                all_results[term_index] = result
                completed_results += 1
                
                # Progress reporting
                if completed_results % 10 == 0 or completed_results == len(terms):
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    rate = completed_results / elapsed if elapsed > 0 else 0
                    eta = (len(terms) - completed_results) / rate if rate > 0 else 0
                    
                    with self.stats_lock:
                        processed = self.processed_count
                        failed = self.failed_count
                    
                    print(f"📊 Progress: {completed_results}/{len(terms)} ({(completed_results/len(terms))*100:.1f}%) | "
                          f"Rate: {rate:.3f} terms/sec | ETA: {eta/3600:.1f} hours | "
                          f"Success: {processed} | Failed: {failed}")
                
                # More frequent checkpoint saving (every 10 terms or at checkpoint interval)
                if (completed_results % 10 == 0 and completed_results > last_checkpoint) or \
                   (completed_results - last_checkpoint >= self.checkpoint_interval):
                    self._save_checkpoint(file_type, completed_results - 1, 
                                        [r for r in all_results if r is not None], len(terms))
                    last_checkpoint = completed_results
                
                self.result_queue.task_done()
                
            except queue.Empty:
                print("⏳ Waiting for GPU results...")
                continue
        
        # Wait for feeder to complete
        feeder_thread.join()
        
        # Final checkpoint and results
        self._save_checkpoint(file_type, len(terms) - 1, all_results, len(terms))
        self._save_final_results(all_results, output_file, file_type)
        
        print(f"✅ CPU coordinator completed - {completed_results} results collected")
    
    def _save_checkpoint(self, file_type: str, processed_index: int, 
                        results: List[Dict], total_terms: int):
        """Save processing checkpoint"""
        checkpoint_data = {
            "session_id": self.session_id,
            "file_type": file_type,
            "processed_index": processed_index,
            "total_terms": total_terms,
            "results_count": len(results),
            "start_time": self.start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "model_size": self.model_size,
            "device": self.device,
            "cpu_workers": self.cpu_workers
        }
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{self.session_id}_{file_type}_checkpoint.json")
        results_file = checkpoint_file.replace('_checkpoint.json', '_results.json')
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {processed_index + 1}/{total_terms} terms")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    def _generate_language_analysis(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive language-specific analysis
        
        Args:
            results: List of translation results
            
        Returns:
            Dictionary with detailed language analysis
        """
        from collections import defaultdict, Counter
        
        # Initialize counters
        lang_same_count = defaultdict(int)  # How many terms stay same per language
        lang_translated_count = defaultdict(int)  # How many terms translated per language
        lang_error_count = defaultdict(int)  # How many errors per language
        
        # Term-specific language data
        term_language_data = []
        
        # Language family patterns
        language_families = {
            'Romance': ['fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'ron_Latn', 'cat_Latn'],
            'Germanic': ['deu_Latn', 'nld_Latn', 'dan_Latn', 'swe_Latn', 'nob_Latn', 'nno_Latn'],
            'Slavic': ['rus_Cyrl', 'pol_Latn', 'ces_Latn', 'slk_Latn', 'bul_Cyrl', 'hrv_Latn'],
            'Arabic': ['arb_Arab', 'ary_Arab', 'arz_Arab', 'acm_Arab', 'apc_Arab', 'ajp_Arab'],
            'Indic': ['hin_Deva', 'ben_Beng', 'guj_Gujr', 'pan_Guru', 'mar_Deva', 'npi_Deva'],
            'East_Asian': ['zho_Hans', 'zho_Hant', 'jpn_Jpan', 'kor_Hang'],
            'African': ['swa_Latn', 'hau_Latn', 'yor_Latn', 'ibo_Latn', 'amh_Ethi', 'som_Latn']
        }
        
        family_same_count = defaultdict(int)
        family_translated_count = defaultdict(int)
        family_total_count = defaultdict(int)
        
        # Process each result
        for result in results:
            if result.get('error'):
                continue
                
            term = result.get('term', '')
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
            
            # Store term-specific data
            term_lang_data = {
                'term': term,
                'frequency': result.get('frequency', 0),
                'translatability_score': result.get('translatability_score', 0.0),
                'languages_keeping_same': same_langs,
                'languages_translating': translated_langs,
                'languages_with_errors': error_langs,
                'same_count': len(same_langs),
                'translated_count': len(translated_langs),
                'error_count': len(error_langs)
            }
            term_language_data.append(term_lang_data)
            
            # Language family analysis
            for family, family_langs in language_families.items():
                family_same = sum(1 for lang in same_langs if lang in family_langs)
                family_translated = sum(1 for lang in translated_langs if lang in family_langs)
                family_total = len([lang for lang in family_langs if lang in self.target_languages])
                
                family_same_count[family] += family_same
                family_translated_count[family] += family_translated
                family_total_count[family] += family_total
        
        # Calculate language statistics
        all_languages = set(lang_same_count.keys()) | set(lang_translated_count.keys())
        language_stats = {}
        
        for lang in all_languages:
            same_count = lang_same_count[lang]
            translated_count = lang_translated_count[lang]
            error_count = lang_error_count[lang]
            total_processed = same_count + translated_count + error_count
            
            if total_processed > 0:
                language_stats[lang] = {
                    'terms_keeping_same': same_count,
                    'terms_translated': translated_count,
                    'terms_with_errors': error_count,
                    'total_terms_processed': total_processed,
                    'same_percentage': (same_count / total_processed) * 100,
                    'translated_percentage': (translated_count / total_processed) * 100,
                    'error_percentage': (error_count / total_processed) * 100,
                    'borrowing_tendency': 'high' if (same_count / total_processed) > 0.7 else 
                                        'medium' if (same_count / total_processed) > 0.3 else 'low'
                }
        
        # Top languages by behavior
        top_same_languages = dict(Counter(lang_same_count).most_common(20))
        top_translated_languages = dict(Counter(lang_translated_count).most_common(20))
        most_borrowing_languages = sorted(
            [(lang, stats['same_percentage']) for lang, stats in language_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:20]
        most_translating_languages = sorted(
            [(lang, stats['translated_percentage']) for lang, stats in language_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:20]
        
        # Language family analysis
        family_analysis = {}
        for family in language_families.keys():
            total_terms = len(results)
            same_total = family_same_count[family]
            translated_total = family_translated_count[family]
            
            if total_terms > 0:
                family_analysis[family] = {
                    'total_same_instances': same_total,
                    'total_translated_instances': translated_total,
                    'average_same_percentage': (same_total / (total_terms * len(language_families[family]))) * 100,
                    'average_translated_percentage': (translated_total / (total_terms * len(language_families[family]))) * 100,
                    'borrowing_tendency': 'high' if (same_total / (same_total + translated_total)) > 0.6 else 
                                        'medium' if (same_total / (same_total + translated_total)) > 0.3 else 'low'
                }
        
        return {
            'summary': {
                'total_languages_analyzed': len(all_languages),
                'total_terms_analyzed': len(results),
                'languages_with_high_borrowing': len([l for l, s in language_stats.items() if s['same_percentage'] > 70]),
                'languages_with_high_translation': len([l for l, s in language_stats.items() if s['translated_percentage'] > 70])
            },
            'language_statistics': language_stats,
            'top_rankings': {
                'most_terms_keeping_same': top_same_languages,
                'most_terms_translating': top_translated_languages,
                'highest_borrowing_percentage': dict(most_borrowing_languages),
                'highest_translation_percentage': dict(most_translating_languages)
            },
            'language_family_analysis': family_analysis,
            'term_language_details': term_language_data[:100]  # First 100 terms for detailed view
        }
    
    def _save_final_results(self, results: List[Dict], output_file: str, file_type: str):
        """Save final results with processing info and detailed language analysis"""
        
        end_time = datetime.now()
        processing_time = (end_time - self.start_time).total_seconds()
        
        # Generate comprehensive language analysis
        language_analysis = self._generate_language_analysis(results)
        
        output_data = {
            "processing_info": {
                "status": "completed",
                "session_id": self.session_id,
                "file_type": file_type,
                "total_terms": len(results),
                "processing_time_seconds": processing_time,
                "terms_per_second": len(results) / processing_time if processing_time > 0 else 0,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "source_language": self.source_lang,
                "target_languages_count": len(self.target_languages),
                "device_used": self.device,
                "model_size": self.model_size,
                "cpu_workers": self.cpu_workers,
                "architecture": "hybrid_gpu_cpu"
            },
            "language_analysis": language_analysis,
            "results": results
        }
        
        try:
            os.makedirs("translation_results", exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Final results saved to: {output_file}")
            
            # Save summary
            summary_file = output_file.replace('.json', '_summary.json')
            summary_data = {
                "processing_info": output_data["processing_info"],
                "result_count": len(results),
                "performance_metrics": {
                    "terms_per_second": len(results) / processing_time,
                    "total_hours": processing_time / 3600,
                    "gpu_efficiency": "hybrid_pipeline"
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Failed to save final results: {e}")
    
    def find_resumable_sessions(self, file_type: str) -> Optional[str]:
        """Find the most recent resumable session for given file type"""
        import glob
        
        checkpoint_pattern = os.path.join(self.checkpoint_dir, f"*_{file_type}_checkpoint.json")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        # Find most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            session_id = checkpoint_data["session_id"]
            processed = checkpoint_data.get("processed_index", -1) + 1
            total = checkpoint_data.get("total_terms", 0)
            
            if processed < total:
                print(f"🔄 Found resumable session: {session_id}")
                print(f"📊 Progress: {processed}/{total} terms ({(processed/total)*100:.1f}%)")
                return session_id
        
        except Exception as e:
            print(f"⚠️  Error reading checkpoint: {e}")
        
        return None
    
    def load_checkpoint_data(self, file_type: str, session_id: str) -> Optional[Dict]:
        """Load checkpoint data for resuming"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{session_id}_{file_type}_checkpoint.json")
        results_file = checkpoint_file.replace('_checkpoint.json', '_results.json')
        
        if os.path.exists(checkpoint_file) and os.path.exists(results_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                return {
                    "checkpoint": checkpoint_data,
                    "results": results,
                    "start_index": checkpoint_data.get("processed_index", -1) + 1
                }
            except Exception as e:
                print(f"❌ Error loading checkpoint data: {e}")
        
        return None
    
    def process_terms_hybrid(self, terms: List[Dict], output_file: str, file_type: str, 
                           auto_resume: bool = True) -> Dict:
        """
        Process terms using hybrid GPU-CPU pipeline
        
        Args:
            terms: List of term dictionaries
            output_file: Output file path
            file_type: Type of terms being processed
            auto_resume: Whether to automatically resume from checkpoint
            
        Returns:
            Processing statistics
        """
        
        # Check for resumable session
        resume_session_id = None
        start_index = 0
        existing_results = []
        
        if auto_resume:
            resume_session_id = self.find_resumable_sessions(file_type)
            if resume_session_id:
                checkpoint_data = self.load_checkpoint_data(file_type, resume_session_id)
                if checkpoint_data:
                    self.session_id = resume_session_id  # Use existing session
                    start_index = checkpoint_data["start_index"]
                    existing_results = checkpoint_data["results"]
                    self.start_time = datetime.fromisoformat(checkpoint_data["checkpoint"]["start_time"])
                    print(f"🔄 Resuming session: {resume_session_id}")
                    print(f"📊 Starting from term {start_index + 1}/{len(terms)}")
        
        if not resume_session_id:
            print(f"🚀 Starting new hybrid GPU-CPU processing of {len(terms)} terms")
            self.start_time = datetime.now()
        
        print(f"📁 Output file: {output_file}")
        print(f"🎮 GPU: Translation pipeline")
        print(f"👥 CPU: {self.cpu_workers} workers for coordination")
        
        # Start GPU worker thread
        gpu_thread = threading.Thread(target=self._gpu_translation_worker)
        gpu_thread.start()
        
        # Start CPU coordinator (runs in main thread) - pass resume data
        self._cpu_coordinator_worker(terms, output_file, file_type, start_index, existing_results)
        
        # Stop GPU worker
        self.stop_event.set()
        self.term_queue.put(None)  # Sentinel to stop GPU worker
        gpu_thread.join()
        
        processing_time = (datetime.now() - self.start_time).total_seconds()
        
        with self.stats_lock:
            processed = self.processed_count
            failed = self.failed_count
        
        print(f"🎉 Hybrid processing completed!")
        print(f"📊 Processed: {processed}/{len(terms)} terms")
        print(f"❌ Failed: {failed} terms")
        print(f"⏱️  Total time: {processing_time/3600:.1f} hours")
        print(f"🚀 Rate: {processed/processing_time:.3f} terms/second")
        
        return {
            "status": "completed",
            "processed": processed,
            "failed": failed,
            "processing_time": processing_time,
            "session_id": self.session_id,
            "architecture": "hybrid_gpu_cpu"
        }


def load_term_candidates(file_path: str) -> List[Dict]:
    """Load term candidates from JSON file"""
    print(f"📁 Loading term candidates from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'dictionary_terms' in data:
            terms = data['dictionary_terms']
            term_type = 'dictionary'
        elif 'non_dictionary_terms' in data:
            terms = data['non_dictionary_terms']
            term_type = 'non_dictionary'
        else:
            raise ValueError("Unknown JSON structure")
        
        print(f"✅ Loaded {len(terms)} {term_type} terms")
        return terms
        
    except Exception as e:
        print(f"❌ Failed to load terms from {file_path}: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hybrid GPU-CPU Translation Analysis")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device for GPU translation (default: auto)")
    parser.add_argument("--model-size", choices=["auto", "small", "medium", "1.3B", "3.3B"], default="auto",
                       help="Model size (default: auto - maximum that fits GPU)")
    parser.add_argument("--cpu-workers", type=int, default=4,
                       help="Number of CPU workers (default: 4)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                       help="Save checkpoint every N terms (default: 100)")
    parser.add_argument("--max-terms", type=int, default=None,
                       help="Maximum terms to process per file")
    parser.add_argument("--test-run", action="store_true",
                       help="Run with limited terms for testing")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh without resuming from checkpoints")
    
    args = parser.parse_args()
    
    # Test run adjustments
    if args.test_run:
        args.max_terms = 50
        args.checkpoint_interval = 20
        print("🧪 Running in test mode")
    
    # Initialize hybrid runner
    runner = HybridGPUCPURunner(
        device=args.device,
        model_size=args.model_size,
        cpu_workers=args.cpu_workers,
        checkpoint_interval=args.checkpoint_interval
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process dictionary terms
    dict_file = "Fast_Dictionary_Terms_20250903_123659.json"
    if os.path.exists(dict_file):
        print(f"\n📚 PROCESSING DICTIONARY TERMS")
        print("=" * 50)
        
        dictionary_terms = load_term_candidates(dict_file)
        
        if args.max_terms:
            dictionary_terms = dictionary_terms[:args.max_terms]
            print(f"🔢 Limited to {args.max_terms} terms")
        
        dict_output = f"translation_results/hybrid_dictionary_results_{timestamp}.json"
        
        runner.process_terms_hybrid(dictionary_terms, dict_output, 'dictionary', 
                                   auto_resume=not args.no_resume)
    
    # Process non-dictionary terms
    non_dict_file = "Fast_Non_Dictionary_Terms_20250903_123659.json"
    if os.path.exists(non_dict_file):
        print(f"\n📖 PROCESSING NON-DICTIONARY TERMS")
        print("=" * 50)
        
        non_dict_terms = load_term_candidates(non_dict_file)
        
        if args.max_terms:
            non_dict_terms = non_dict_terms[:args.max_terms]
            print(f"🔢 Limited to {args.max_terms} terms")
        
        non_dict_output = f"translation_results/hybrid_non_dictionary_results_{timestamp}.json"
        
        # Reset counters for second file
        runner.processed_count = 0
        runner.failed_count = 0
        
        runner.process_terms_hybrid(non_dict_terms, non_dict_output, 'non_dictionary',
                                   auto_resume=not args.no_resume)
    
    print(f"\n🎉 HYBRID PROCESSING COMPLETED!")
    print(f"📋 Session ID: {runner.session_id}")
    print(f"🚀 Architecture: GPU Translation + CPU Coordination")


if __name__ == "__main__":
    main()
