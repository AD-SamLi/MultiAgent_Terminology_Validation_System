#!/usr/bin/env python3
"""
Enhanced Term Translation Processor with Resumable and Parallel Processing
Supports checkpoint-based resuming and multi-GPU/multi-process translation
"""

import os
import json
import time
import logging
import pickle
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict, Counter
import hashlib

from nllb_translation_tool import NLLBTranslationTool, TranslationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/translation_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingCheckpoint:
    """Container for processing checkpoint data"""
    session_id: str
    file_type: str  # 'dictionary' or 'non_dictionary'
    total_terms: int
    processed_terms: int
    failed_terms: int
    last_processed_index: int
    start_time: datetime
    last_update_time: datetime
    results: List[Dict]
    processing_config: Dict

@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    max_workers: int = 4
    batch_size: int = 8
    chunk_size: int = 10  # Terms per chunk for parallel processing
    use_multiprocessing: bool = True  # vs threading
    gpu_devices: List[int] = None  # List of GPU device IDs to use

class EnhancedTermTranslationProcessor:
    """
    Enhanced processor with resumable and parallel translation capabilities
    """
    
    def __init__(self, device: str = "auto", model_size: str = "small", 
                 parallel_config: ParallelConfig = None):
        """
        Initialize the enhanced translation processor
        
        Args:
            device: Device for NLLB model ("auto", "cuda", "cpu")
            model_size: Model size ("small", "medium", "1.3B", "3.3B")
            parallel_config: Configuration for parallel processing
        """
        self.device = device
        self.model_size = model_size
        self.source_lang = "eng_Latn"
        
        # Parallel processing configuration
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Session management
        self.session_id = self._generate_session_id()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize translation tool
        print("🔧 Initializing Enhanced NLLB Translation Processor...")
        print(f"📋 Session ID: {self.session_id}")
        print(f"📋 Model size: {model_size}")
        print(f"🔄 Parallel config: {self.parallel_config.max_workers} workers, chunk size {self.parallel_config.chunk_size}")
        
        self.translator = NLLBTranslationTool(model_name=model_size, device=device, batch_size=self.parallel_config.batch_size)
        
        # Get available languages
        all_languages = self.translator.get_available_languages()
        self.target_languages = [lang for lang in all_languages if lang != self.source_lang]
        
        print(f"✅ Enhanced processor initialized")
        print(f"🎯 Target languages: {len(self.target_languages)}")
        
        # Thread-safe counters
        self.lock = threading.Lock()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def _get_checkpoint_file(self, file_type: str) -> str:
        """Get checkpoint file path for given file type"""
        return os.path.join(self.checkpoint_dir, f"{self.session_id}_{file_type}_checkpoint.pkl")
    
    def _save_checkpoint(self, checkpoint: ProcessingCheckpoint):
        """Save processing checkpoint"""
        checkpoint_file = self._get_checkpoint_file(checkpoint.file_type)
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, file_type: str) -> Optional[ProcessingCheckpoint]:
        """Load processing checkpoint if exists"""
        checkpoint_file = self._get_checkpoint_file(file_type)
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"📂 Checkpoint loaded: {checkpoint_file}")
                return checkpoint
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
        return None
    
    def _find_existing_session(self, file_type: str) -> Optional[str]:
        """Find existing session for resuming"""
        checkpoint_pattern = f"*_{file_type}_checkpoint.pkl"
        checkpoint_files = list(Path(self.checkpoint_dir).glob(checkpoint_pattern))
        
        if checkpoint_files:
            # Get most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            session_id = latest_checkpoint.stem.replace(f"_{file_type}_checkpoint", "")
            return session_id
        return None
    
    def resume_session(self, file_type: str, session_id: str = None) -> bool:
        """Resume processing from existing session"""
        if session_id:
            self.session_id = session_id
        else:
            # Find most recent session
            found_session = self._find_existing_session(file_type)
            if found_session:
                self.session_id = found_session
            else:
                print(f"❌ No existing session found for {file_type} terms")
                return False
        
        checkpoint = self._load_checkpoint(file_type)
        if checkpoint:
            print(f"🔄 Resuming session: {self.session_id}")
            print(f"📊 Progress: {checkpoint.processed_terms}/{checkpoint.total_terms} terms")
            print(f"⏱️  Last update: {checkpoint.last_update_time}")
            return True
        return False
    
    def translate_term_chunk(self, term_chunk: List[Dict], chunk_id: int) -> List[Dict]:
        """
        Translate a chunk of terms (for parallel processing)
        
        Args:
            term_chunk: List of term dictionaries
            chunk_id: Unique identifier for this chunk
            
        Returns:
            List of translation results
        """
        try:
            # Create a separate translator instance for this process/thread
            translator = NLLBTranslationTool(model_name=self.model_size, device=self.device, batch_size=4)
            
            chunk_results = []
            
            for term_data in term_chunk:
                term = term_data.get('term', '')
                frequency = term_data.get('frequency', 0)
                
                if not term or not term.strip():
                    continue
                
                start_time = time.time()
                
                try:
                    # Translate to all target languages
                    results = translator.translate_to_all_languages(term, self.source_lang)
                    
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
                            # Keep some sample translations
                            if len(sample_translations) < 10:
                                sample_translations[lang_code] = result.translated_text
                    
                    # Calculate translatability score
                    total_valid = len(same_languages) + len(translated_languages)
                    translatability_score = len(translated_languages) / total_valid if total_valid > 0 else 0.0
                    
                    processing_time = time.time() - start_time
                    
                    result_data = {
                        "term": term,
                        "frequency": frequency,
                        "total_languages": len(results),
                        "same_languages": len(same_languages),
                        "translated_languages": len(translated_languages),
                        "error_languages": len(error_languages),
                        "translatability_score": translatability_score,
                        "same_language_codes": same_languages,
                        "translated_language_codes": translated_languages,
                        "error_language_codes": error_languages,
                        "sample_translations": sample_translations,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "processing_time_seconds": processing_time,
                        "chunk_id": chunk_id
                    }
                    
                    chunk_results.append(result_data)
                    
                    logger.info(f"✅ Chunk {chunk_id}: Processed '{term}' (score: {translatability_score:.3f})")
                    
                except Exception as e:
                    logger.error(f"❌ Chunk {chunk_id}: Failed to process '{term}': {e}")
                    
                    error_result = {
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
                        "processing_time_seconds": time.time() - start_time,
                        "error": str(e),
                        "chunk_id": chunk_id
                    }
                    chunk_results.append(error_result)
            
            logger.info(f"🎯 Chunk {chunk_id}: Completed {len(chunk_results)} terms")
            return chunk_results
            
        except Exception as e:
            logger.error(f"❌ Chunk {chunk_id}: Critical error: {e}")
            return []
    
    def process_terms_parallel(self, terms: List[Dict], output_file: str, 
                             file_type: str, resume: bool = True) -> Dict:
        """
        Process terms with parallel translation and resumable checkpoints
        
        Args:
            terms: List of term dictionaries
            output_file: Output file path for results
            file_type: Type of terms ('dictionary' or 'non_dictionary')
            resume: Whether to attempt resuming from checkpoint
            
        Returns:
            Processing statistics
        """
        
        # Try to resume from checkpoint
        checkpoint = None
        start_index = 0
        existing_results = []
        
        if resume:
            checkpoint = self._load_checkpoint(file_type)
            if checkpoint:
                print(f"🔄 Resuming from checkpoint...")
                start_index = checkpoint.last_processed_index + 1
                existing_results = checkpoint.results
                print(f"📊 Resuming from term {start_index}/{len(terms)}")
        
        # Determine terms to process
        terms_to_process = terms[start_index:]
        total_terms = len(terms)
        
        if not terms_to_process:
            print("✅ All terms already processed!")
            return {"status": "completed", "processed": len(existing_results)}
        
        print(f"🚀 Starting parallel processing of {len(terms_to_process)} terms")
        print(f"📁 Output file: {output_file}")
        print(f"🎯 Processing terms {start_index} to {total_terms - 1}")
        
        # Initialize checkpoint
        if not checkpoint:
            checkpoint = ProcessingCheckpoint(
                session_id=self.session_id,
                file_type=file_type,
                total_terms=total_terms,
                processed_terms=len(existing_results),
                failed_terms=0,
                last_processed_index=start_index - 1,
                start_time=datetime.now(),
                last_update_time=datetime.now(),
                results=existing_results,
                processing_config={
                    "model_size": self.model_size,
                    "device": self.device,
                    "parallel_config": asdict(self.parallel_config)
                }
            )
        
        # Split terms into chunks for parallel processing
        chunk_size = self.parallel_config.chunk_size
        chunks = []
        for i in range(0, len(terms_to_process), chunk_size):
            chunk = terms_to_process[i:i + chunk_size]
            chunks.append((chunk, start_index + i // chunk_size))
        
        print(f"📦 Created {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        all_results = existing_results.copy()
        processed_count = len(existing_results)
        failed_count = 0
        
        executor_class = ProcessPoolExecutor if self.parallel_config.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.parallel_config.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self.translate_term_chunk, chunk_data, chunk_id): (chunk_data, chunk_id)
                for chunk_data, chunk_id in chunks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_data, chunk_id = future_to_chunk[future]
                
                try:
                    chunk_results = future.result()
                    
                    if chunk_results:
                        all_results.extend(chunk_results)
                        processed_count += len(chunk_results)
                        
                        # Update checkpoint
                        checkpoint.processed_terms = processed_count
                        checkpoint.last_processed_index = start_index + (chunk_id * chunk_size) + len(chunk_data) - 1
                        checkpoint.last_update_time = datetime.now()
                        checkpoint.results = all_results
                        
                        # Save checkpoint every few chunks
                        if chunk_id % 5 == 0:  # Save every 5 chunks
                            self._save_checkpoint(checkpoint)
                        
                        # Progress reporting
                        progress_percent = (processed_count / total_terms) * 100
                        elapsed = (datetime.now() - checkpoint.start_time).total_seconds()
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_terms - processed_count) / rate if rate > 0 else 0
                        
                        print(f"📊 Progress: {processed_count}/{total_terms} ({progress_percent:.1f}%) | "
                              f"Rate: {rate:.3f} terms/sec | ETA: {eta/3600:.1f} hours")
                    else:
                        failed_count += len(chunk_data)
                        logger.error(f"❌ Chunk {chunk_id} failed completely")
                        
                except Exception as e:
                    failed_count += len(chunk_data)
                    logger.error(f"❌ Failed to process chunk {chunk_id}: {e}")
        
        # Final checkpoint save
        checkpoint.processed_terms = processed_count
        checkpoint.failed_terms = failed_count
        checkpoint.last_update_time = datetime.now()
        checkpoint.results = all_results
        self._save_checkpoint(checkpoint)
        
        # Save final results
        self._save_final_results(all_results, output_file, checkpoint)
        
        print(f"✅ Parallel processing completed!")
        print(f"📊 Processed: {processed_count}/{total_terms} terms")
        print(f"❌ Failed: {failed_count} terms")
        
        processing_time = (checkpoint.last_update_time - checkpoint.start_time).total_seconds()
        print(f"⏱️  Total time: {processing_time/3600:.1f} hours")
        print(f"🚀 Rate: {processed_count/processing_time:.3f} terms/second")
        
        return {
            "status": "completed",
            "processed": processed_count,
            "failed": failed_count,
            "processing_time": processing_time,
            "session_id": self.session_id
        }
    
    def _save_final_results(self, results: List[Dict], output_file: str, checkpoint: ProcessingCheckpoint):
        """Save final results with comprehensive analysis"""
        
        output_data = {
            "processing_info": {
                "status": "completed",
                "session_id": checkpoint.session_id,
                "total_terms": checkpoint.total_terms,
                "processed_terms": checkpoint.processed_terms,
                "failed_terms": checkpoint.failed_terms,
                "processing_time_seconds": (checkpoint.last_update_time - checkpoint.start_time).total_seconds(),
                "terms_per_second": checkpoint.processed_terms / (checkpoint.last_update_time - checkpoint.start_time).total_seconds(),
                "start_time": checkpoint.start_time.isoformat(),
                "end_time": checkpoint.last_update_time.isoformat(),
                "source_language": self.source_lang,
                "target_languages_count": len(self.target_languages),
                "device_used": self.device,
                "model_size": self.model_size,
                "parallel_config": checkpoint.processing_config.get("parallel_config", {})
            },
            "results": results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Final results saved to: {output_file}")
            
            # Also save a summary report
            summary_file = output_file.replace('.json', '_summary.json')
            summary_data = {
                "processing_info": output_data["processing_info"],
                "result_count": len(results)
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary report saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final results: {e}")
    
    def list_sessions(self) -> List[Dict]:
        """List all available sessions for resuming"""
        sessions = []
        checkpoint_files = list(Path(self.checkpoint_dir).glob("*_checkpoint.pkl"))
        
        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                sessions.append({
                    "session_id": checkpoint.session_id,
                    "file_type": checkpoint.file_type,
                    "progress": f"{checkpoint.processed_terms}/{checkpoint.total_terms}",
                    "last_update": checkpoint.last_update_time.isoformat(),
                    "checkpoint_file": str(checkpoint_file)
                })
            except Exception as e:
                logger.error(f"❌ Failed to read checkpoint {checkpoint_file}: {e}")
        
        return sessions


def load_term_candidates(file_path: str) -> List[Dict]:
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
        return terms
        
    except Exception as e:
        print(f"❌ Failed to load terms from {file_path}: {e}")
        raise


def process_with_resume(dictionary_file: str, non_dictionary_file: str, 
                       output_dir: str = "translation_results",
                       parallel_config: ParallelConfig = None,
                       resume_session: str = None,
                       device: str = "auto",
                       model_size: str = "small"):
    """
    Process term files with resumable and parallel processing
    
    Args:
        dictionary_file: Path to dictionary terms JSON file
        non_dictionary_file: Path to non-dictionary terms JSON file
        output_dir: Directory for output files
        parallel_config: Configuration for parallel processing
        resume_session: Session ID to resume (None for new session)
        device: Device for translation model
        model_size: Model size
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize enhanced processor
    if parallel_config is None:
        parallel_config = ParallelConfig(
            max_workers=min(4, mp.cpu_count()),
            batch_size=8,
            chunk_size=5,  # Smaller chunks for better progress tracking
            use_multiprocessing=True
        )
    
    processor = EnhancedTermTranslationProcessor(
        device=device,
        model_size=model_size,
        parallel_config=parallel_config
    )
    
    # Resume session if specified
    if resume_session:
        processor.session_id = resume_session
    
    # Process dictionary terms
    if os.path.exists(dictionary_file):
        print(f"\n📚 PROCESSING DICTIONARY TERMS")
        print("=" * 50)
        
        dictionary_terms = load_term_candidates(dictionary_file)
        dict_output_file = os.path.join(output_dir, f"dictionary_terms_translation_results_{timestamp}.json")
        
        # Try to resume
        resume_success = False
        if resume_session:
            resume_success = processor.resume_session('dictionary', resume_session)
        
        dict_stats = processor.process_terms_parallel(
            dictionary_terms, 
            dict_output_file,
            'dictionary',
            resume=resume_success or not resume_session
        )
        
        print(f"✅ Dictionary terms processing completed")
    
    # Process non-dictionary terms  
    if os.path.exists(non_dictionary_file):
        print(f"\n📖 PROCESSING NON-DICTIONARY TERMS")
        print("=" * 50)
        
        non_dict_terms = load_term_candidates(non_dictionary_file)
        non_dict_output_file = os.path.join(output_dir, f"non_dictionary_terms_translation_results_{timestamp}.json")
        
        # Try to resume
        resume_success = False
        if resume_session:
            resume_success = processor.resume_session('non_dictionary', resume_session)
        
        non_dict_stats = processor.process_terms_parallel(
            non_dict_terms,
            non_dict_output_file,
            'non_dictionary', 
            resume=resume_success or not resume_session
        )
        
        print(f"✅ Non-dictionary terms processing completed")
    
    print(f"\n🎉 ALL PROCESSING COMPLETED!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📋 Session ID: {processor.session_id}")


if __name__ == "__main__":
    # Example usage
    parallel_config = ParallelConfig(
        max_workers=4,
        batch_size=8,
        chunk_size=5,
        use_multiprocessing=True
    )
    
    process_with_resume(
        dictionary_file="Fast_Dictionary_Terms_20250903_123659.json",
        non_dictionary_file="Fast_Non_Dictionary_Terms_20250903_123659.json",
        output_dir="translation_results",
        parallel_config=parallel_config,
        device="auto",
        model_size="small"
    )

