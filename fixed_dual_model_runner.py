#!/usr/bin/env python3
"""
🔧 FIXED DUAL-MODEL ULTRA-FAST RUNNER
=====================================

BREAKTHROUGH SOLUTION: 2x NLLB Models with Sequential Loading!
- 🎮 Sequential model loading (prevents GPU competition)
- 💻 Smart CPU worker distribution
- 📦 Conservative batch sizes for stability
- ⚡ ELIMINATES queue bottleneck completely!
- 🛡️ Robust error handling and recovery

FIXES APPLIED:
- Sequential GPU worker initialization
- Reduced batch sizes (32 per model)
- Better queue management (50 per queue)
- Improved error handling
- Resource cleanup between loads
"""

import os
import sys
import json
import time
import queue
import threading
import torch
import gc
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

# Add the current directory to Python path for imports
sys.path.append('/home/samli/Documents/Python/Term_Verify')

from nllb_translation_tool import NLLBTranslationTool

@dataclass
class FixedDualModelConfig:
    """Fixed configuration for stable dual model setup"""
    model_size: str = "1.3B"
    gpu_workers: int = 2              # 2 GPU workers (one per model)
    cpu_workers: int = 12             # Reduced for stability
    gpu_batch_size: int = 32          # Reduced from 48 for stability
    max_queue_size: int = 50          # Smaller queues per GPU
    checkpoint_interval: int = 45     # More frequent saves
    model_load_delay: int = 10        # Seconds between model loads
    
class FixedDualModelRunner:
    """Fixed dual-model runner with sequential loading and better stability"""
    
    def __init__(self, config: FixedDualModelConfig = None, resume_session: str = None):
        self.config = config or FixedDualModelConfig()
        self.session_id = resume_session or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Debug: Show what session we're using
        if resume_session:
            print(f"🔄 RESUMING from session: {resume_session}")
        else:
            print(f"🆕 STARTING new session: {self.session_id}")
        
        # Dual GPU queues (one per model)
        self.gpu_queue_1 = queue.Queue(maxsize=self.config.max_queue_size)
        self.gpu_queue_2 = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        
        # Load balancer for distributing work
        self.next_gpu = 0  # Round-robin between GPUs
        self.gpu_lock = threading.Lock()  # Prevent race conditions
        
        # Thread control
        self.stop_event = threading.Event()
        self.gpu1_ready = threading.Event()
        self.gpu2_ready = threading.Event()
        
        # Progress tracking
        self.processed_terms = 0
        self.failed_terms = 0
        self.total_terms = 0
        self.processed_terms_set: Set[str] = set()
        self.results = []  # Always initialize as list
        
        # Performance tracking
        self.start_time = None
        self.last_checkpoint_time = time.time()
        
        # Load existing progress if resuming
        if resume_session:
            self._load_checkpoint()
        
        print(f"🔧 FIXED DUAL-MODEL RUNNER INITIALIZED")
        print(f"   • Session: {self.session_id}")
        print(f"   • GPU Workers: {self.config.gpu_workers} (sequential loading)")
        print(f"   • CPU Workers: {self.config.cpu_workers}")
        print(f"   • Batch Size: {self.config.gpu_batch_size} per model")
        print(f"   • Queue Size: {self.config.max_queue_size} per GPU")
        print(f"   • Load Delay: {self.config.model_load_delay}s between models")
    
    def _load_checkpoint(self):
        """Load existing checkpoint and results (supports both formats)"""
        print(f"🔍 Looking for checkpoint with session_id: {self.session_id}")
        try:
            # Try fixed dual-model format first
            fixed_checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/fixed_dual_{self.session_id}_checkpoint.json"
            fixed_results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/fixed_dual_{self.session_id}_results.json"
            print(f"🔍 Checking for: {fixed_checkpoint_file}")
            print(f"🔍 File exists: {os.path.exists(fixed_checkpoint_file)}")
            
            # Try other formats as fallback
            ultra_checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_checkpoint.json"
            ultra_results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_results.json"
            
            dual_checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_checkpoint.json"
            dual_results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_results.json"
            
            checkpoint_loaded = False
            results_loaded = False
            
            # Try to load checkpoint (priority order)
            for checkpoint_file, format_name in [
                (fixed_checkpoint_file, "fixed-dual-model"),
                (ultra_checkpoint_file, "ultra-fast"),
                (dual_checkpoint_file, "dual-model")
            ]:
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    checkpoint_loaded = True
                    print(f"📂 Loading from {format_name} checkpoint...")
                    break
            
            if checkpoint_loaded:
                self.processed_terms = checkpoint_data.get('processed_terms', 0)
                self.failed_terms = checkpoint_data.get('failed_terms', 0)
                self.total_terms = checkpoint_data.get('total_terms', 0)
                print(f"📂 Loaded checkpoint: {self.processed_terms} processed, {self.failed_terms} failed")
            
            # Try to load results (priority order)
            for results_file, format_name in [
                (fixed_results_file, "fixed-dual-model"),
                (ultra_results_file, "ultra-fast"),
                (dual_results_file, "dual-model")
            ]:
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        self.results = json.load(f)
                    results_loaded = True
                    print(f"📂 Loading from {format_name} results...")
                    break
            
            if results_loaded:
                # Ensure results is a list, not dict
                if isinstance(self.results, dict):
                    print(f"🔧 Converting results dict to list format...")
                    if 'results' in self.results:
                        self.results = self.results['results']
                    else:
                        # Convert dict values to list if needed
                        self.results = list(self.results.values()) if self.results else []
                
                # Build processed terms set for deduplication
                for result in self.results:
                    if isinstance(result, dict) and 'term' in result:
                        self.processed_terms_set.add(result['term'])
                
                print(f"📂 Loaded {len(self.results)} previous results")
                print(f"🔧 Converting to fixed dual-model format...")
            
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting fresh...")
    
    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load dictionary and non-dictionary terms"""
        # Load dictionary terms
        try:
            with open('/home/samli/Documents/Python/Term_Verify/Dictionary_Terms_Found.json', 'r', encoding='utf-8') as f:
                dict_data = json.load(f)
            
            if isinstance(dict_data, dict) and 'dictionary_terms' in dict_data:
                dict_terms_list = dict_data['dictionary_terms']
                if isinstance(dict_terms_list, list):
                    dict_terms = [item['term'] for item in dict_terms_list if isinstance(item, dict) and 'term' in item]
                else:
                    dict_terms = list(dict_terms_list.keys()) if isinstance(dict_terms_list, dict) else []
            else:
                dict_terms = dict_data
        except Exception as e:
            print(f"⚠️  Warning: Could not load dictionary terms: {e}")
            dict_terms = []
        
        # Load non-dictionary terms
        try:
            with open('/home/samli/Documents/Python/Term_Verify/Non_Dictionary_Terms.json', 'r', encoding='utf-8') as f:
                non_dict_data = json.load(f)
            
            if isinstance(non_dict_data, dict) and 'non_dictionary_terms' in non_dict_data:
                non_dict_terms_list = non_dict_data['non_dictionary_terms']
                if isinstance(non_dict_terms_list, list):
                    non_dict_terms = [item['term'] for item in non_dict_terms_list if isinstance(item, dict) and 'term' in item]
                else:
                    non_dict_terms = list(non_dict_terms_list.keys()) if isinstance(non_dict_terms_list, dict) else []
            else:
                non_dict_terms = non_dict_data
        except Exception as e:
            print(f"⚠️  Warning: Could not load non-dictionary terms: {e}")
            non_dict_terms = []
        
        # Filter out already processed terms
        dict_terms = [term for term in dict_terms if term not in self.processed_terms_set]
        non_dict_terms = [term for term in non_dict_terms if term not in self.processed_terms_set]
        
        print(f"📊 Data loaded: {len(dict_terms)} dictionary + {len(non_dict_terms)} non-dictionary = {len(dict_terms) + len(non_dict_terms)} terms")
        
        return dict_terms, non_dict_terms
    
    def _gpu_translation_worker(self, worker_id: int, gpu_queue: queue.Queue, ready_event: threading.Event):
        """Fixed GPU worker with sequential loading and better error handling"""
        print(f"🎮 Initializing GPU worker {worker_id}...")
        
        # Wait for previous worker if this is worker 2
        if worker_id == 2:
            print(f"🎮 GPU worker {worker_id} waiting for worker 1 to be ready...")
            self.gpu1_ready.wait(timeout=120)  # Wait up to 2 minutes
            print(f"🎮 GPU worker {worker_id} starting model load after delay...")
            time.sleep(self.config.model_load_delay)  # Delay between loads
        
        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            print(f"🎮 GPU worker {worker_id} loading model...")
            
            # Initialize translation tool with conservative settings
            translator = NLLBTranslationTool(
                model_name=self.config.model_size,
                batch_size=self.config.gpu_batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"✅ GPU worker {worker_id} model loaded: {self.config.model_size} (Batch: {self.config.gpu_batch_size})")
            
            # Signal that this worker is ready
            ready_event.set()
            
            if worker_id == 1:
                print(f"🎮 GPU worker 1 ready, signaling worker 2 can start...")
            
            batch_count = 0
            
            while not self.stop_event.is_set():
                try:
                    # Get batch of work
                    batch_items = []
                    timeout = 2.0  # Longer timeout for stability
                    
                    # Collect batch items
                    for _ in range(self.config.gpu_batch_size):
                        try:
                            item = gpu_queue.get(timeout=timeout)
                            if item is None:  # Shutdown signal
                                print(f"🎮 GPU worker {worker_id} received shutdown signal")
                                return
                            batch_items.append(item)
                            timeout = 0.2  # Faster subsequent gets
                        except queue.Empty:
                            break
                    
                    if not batch_items:
                        continue
                    
                    batch_count += 1
                    print(f"🎮 GPU-{worker_id} processing batch {batch_count} ({len(batch_items)} items)")
                    
                    # Process each term in the batch
                    for term, target_languages in batch_items:
                        try:
                            # Track start time for analysis
                            start_time = time.time()
                            
                            # Translate to all target languages
                            translations = {}
                            
                            # Process in smaller sub-batches for memory efficiency
                            sub_batch_size = 15  # Smaller for stability
                            for i in range(0, len(target_languages), sub_batch_size):
                                sub_langs = target_languages[i:i + sub_batch_size]
                                
                                for lang in sub_langs:
                                    try:
                                        result = translator.translate_text(
                                            text=term,
                                            src_lang='eng_Latn',
                                            tgt_lang=lang
                                        )
                                        translations[lang] = result.translated_text
                                        print(f"✅ GPU-{worker_id} {lang}: Translated")
                                    except Exception as e:
                                        print(f"❌ GPU-{worker_id} {lang}: Failed - {str(e)[:50]}")
                                        translations[lang] = f"ERROR: {str(e)[:100]}"
                                
                                # Frequent memory cleanup for stability
                                if i % 30 == 0:
                                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # Analyze translation results (same as hybrid runner)
                            analysis_result = self._analyze_translation_results(
                                term, translations, start_time, worker_id
                            )
                            
                            # Create result with full analysis
                            result = analysis_result
                            
                            self.result_queue.put(result, timeout=10.0)
                            
                        except Exception as e:
                            print(f"❌ GPU-{worker_id} processing error for '{term}': {e}")
                            error_result = self._create_empty_result(term, str(e), worker_id)
                            self.result_queue.put(error_result, timeout=10.0)
                    
                    # Conservative memory management
                    if batch_count % 2 == 0:  # Very frequent cleanup
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        
                except Exception as e:
                    print(f"❌ GPU-{worker_id} batch error: {e}")
                    time.sleep(1)  # Brief pause on error
                    continue
                    
        except Exception as e:
            print(f"💥 GPU worker {worker_id} failed to initialize: {e}")
            ready_event.set()  # Signal ready even on failure to prevent deadlock
        finally:
            print(f"🎮 GPU worker {worker_id} shutting down")
            ready_event.set()  # Ensure event is set
    
    def _analyze_translation_results(self, term: str, translations: Dict[str, str], 
                                   start_time: float, worker_id: int) -> Dict:
        """
        Analyze translation results (same logic as hybrid runner)
        Determines which languages keep the term the same vs translate it
        """
        same_languages = []
        translated_languages = []
        error_languages = []
        sample_translations = {}
        
        # Convert frequency to 0 since we don't track frequency in this version
        frequency = 0
        
        for lang_code, translation in translations.items():
            if translation.startswith("ERROR:"):
                error_languages.append(lang_code)
            elif translation.strip().lower() == term.strip().lower():
                same_languages.append(lang_code)
            else:
                translated_languages.append(lang_code)
                # Keep sample translations (limit to 10 for storage)
                if len(sample_translations) < 10:
                    sample_translations[lang_code] = translation
        
        # Calculate translatability score
        total_valid = len(same_languages) + len(translated_languages)
        translatability_score = len(translated_languages) / total_valid if total_valid > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        return {
            "term": term,
            "frequency": frequency,
            "total_languages": len(translations),
            "same_languages": len(same_languages),
            "translated_languages": len(translated_languages),
            "error_languages": len(error_languages),
            "translatability_score": translatability_score,
            "same_language_codes": same_languages,
            "translated_language_codes": translated_languages,
            "error_language_codes": error_languages,
            "sample_translations": sample_translations,
            "all_translations": translations,  # Keep all translation text
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "gpu_worker": worker_id,
            "status": 'completed'
        }
    
    def _create_empty_result(self, term: str, error_msg: str, worker_id: int) -> Dict:
        """Create empty result for failed/empty terms (same as hybrid runner)"""
        return {
            "term": term,
            "frequency": 0,
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
            "gpu_worker": worker_id,
            "status": 'failed',
            "error": error_msg
        }

    def _get_target_languages(self) -> List[str]:
        """Get list of target languages for translation"""
        return [
            'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab',
            'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn',
            'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt',
            'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn',
            'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn',
            'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn',
            'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn',
            'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn',
            'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn',
            'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn',
            'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn',
            'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn',
            'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva',
            'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn',
            'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck',
            'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn',
            'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml',
            'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn',
            'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab',
            'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans',
            'zho_Hant', 'zul_Latn'
        ]
    
    def _cpu_worker(self, worker_id: int, terms: List[str]):
        """CPU worker with intelligent load balancing and better error handling"""
        target_languages = self._get_target_languages()
        processed_count = 0
        
        # Wait for both GPU workers to be ready
        print(f"👷 CPU worker {worker_id} waiting for GPU workers to be ready...")
        gpu1_ready = self.gpu1_ready.wait(timeout=180)  # 3 minutes max
        gpu2_ready = self.gpu2_ready.wait(timeout=180)  # 3 minutes max
        
        if not (gpu1_ready and gpu2_ready):
            print(f"⚠️  CPU worker {worker_id}: Not all GPU workers ready, proceeding anyway...")
        else:
            print(f"✅ CPU worker {worker_id}: Both GPU workers ready, starting processing...")
        
        for term in terms:
            if self.stop_event.is_set():
                break
            
            # Skip if already processed
            if term in self.processed_terms_set:
                continue
            
            # Smart load balancing with thread safety
            with self.gpu_lock:
                selected_queue = self.gpu_queue_1 if self.next_gpu == 0 else self.gpu_queue_2
                selected_gpu_id = 1 if self.next_gpu == 0 else 2
                self.next_gpu = (self.next_gpu + 1) % 2  # Round-robin
            
            try:
                # Send to selected GPU queue with longer timeout
                work_item = (term, target_languages)
                selected_queue.put(work_item, timeout=30.0)  # Longer timeout
                processed_count += 1
                
                if processed_count % 5 == 0:
                    print(f"👷 CPU-{worker_id}: Sent {processed_count} terms (→ GPU-{selected_gpu_id})")
                
            except queue.Full:
                # Try the other queue if first is full
                alternate_queue = self.gpu_queue_2 if selected_queue == self.gpu_queue_1 else self.gpu_queue_1
                alternate_gpu_id = 2 if selected_gpu_id == 1 else 1
                
                try:
                    alternate_queue.put(work_item, timeout=15.0)
                    processed_count += 1
                    print(f"👷 CPU-{worker_id}: Switched to GPU-{alternate_gpu_id} (queue {selected_gpu_id} full)")
                except queue.Full:
                    print(f"⚠️  CPU worker {worker_id}: Both GPU queues full, waiting...")
                    time.sleep(0.5)  # Longer wait
                    continue
            except Exception as e:
                print(f"❌ CPU worker {worker_id} error: {e}")
                continue
        
        print(f"✅ CPU worker {worker_id} completed: {processed_count} terms sent")
    
    def _result_collector(self):
        """Collect and save results with fixed dual-model progress tracking"""
        print("📊 Starting fixed dual-model result collector...")
        
        last_save_time = time.time()
        save_interval = self.config.checkpoint_interval
        
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                # Get result with timeout
                result = self.result_queue.get(timeout=3.0)
                
                # Ensure results is still a list (safety check)
                if not isinstance(self.results, list):
                    print(f"⚠️  Results corrupted, reinitializing as list...")
                    self.results = []
                
                # Process result
                self.results.append(result)
                term = result.get('term', '')
                
                if result.get('status') == 'completed':
                    self.processed_terms += 1
                    self.processed_terms_set.add(term)
                    
                    # Show progress with GPU worker info
                    gpu_worker = result.get('gpu_worker', '?')
                    if self.processed_terms % 5 == 0:
                        progress = (self.processed_terms / self.total_terms * 100) if self.total_terms > 0 else 0
                        rate = self.processed_terms / (time.time() - self.start_time) if self.start_time else 0
                        print(f"📈 Progress: {self.processed_terms}/{self.total_terms} ({progress:.1f}%) | Rate: {rate:.3f} terms/sec | GPU-{gpu_worker}")
                else:
                    self.failed_terms += 1
                
                # More frequent checkpoint saves
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self._save_checkpoint()
                    last_save_time = current_time
                
                # Performance summary every 50 terms
                if self.processed_terms % 50 == 0 and self.processed_terms > 0:
                    total_processed = self.processed_terms + self.failed_terms
                    rate = total_processed / (time.time() - self.start_time) if self.start_time else 0
                    eta_hours = (self.total_terms - total_processed) / (rate * 3600) if rate > 0 else 0
                    
                    # System stats
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    
                    print(f"🔧 FIXED DUAL-MODEL PROGRESS:")
                    print(f"   • Completed: {self.processed_terms}/{self.total_terms} ({(self.processed_terms/self.total_terms*100):.1f}%)")
                    print(f"   • Rate: {rate:.3f} terms/sec | ETA: {eta_hours:.1f} hours")
                    print(f"   • Success: {self.processed_terms} | Failed: {self.failed_terms}")
                    print(f"   • System: CPU: {cpu_percent:.1f}% | RAM: {memory_info.percent:.1f}%")
                    print(f"   🛡️ STABLE DUAL-GPU PROCESSING...")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Result collector error: {e}")
                continue
        
        # Final save
        self._save_checkpoint()
        print("📊 Fixed dual-model result collector finished")
    
    def _generate_language_analysis(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive language-specific analysis (same as hybrid runner)
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
            'African': ['swh_Latn', 'hau_Latn', 'yor_Latn', 'ibo_Latn', 'amh_Ethi', 'som_Latn']
        }
        
        family_same_count = defaultdict(int)
        family_translated_count = defaultdict(int)
        family_total_count = defaultdict(int)
        
        # Get target languages for family analysis
        target_languages = self._get_target_languages()
        
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
                family_total = len([lang for lang in family_langs if lang in target_languages])
                
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

    def _save_final_results_with_analysis(self, total_time: float, language_analysis: Dict):
        """Save final results with comprehensive analysis (same format as hybrid runner)"""
        
        end_time = datetime.now()
        target_languages = self._get_target_languages()
        
        output_data = {
            "processing_info": {
                "status": "completed",
                "session_id": self.session_id,
                "file_type": "fixed_dual_model",
                "total_terms": len(self.results),
                "processing_time_seconds": total_time,
                "terms_per_second": len(self.results) / total_time if total_time > 0 else 0,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": end_time.isoformat(),
                "source_language": "eng_Latn",
                "target_languages_count": len(target_languages),
                "device_used": "cuda" if torch.cuda.is_available() else "cpu",
                "model_size": self.config.model_size,
                "gpu_workers": self.config.gpu_workers,
                "cpu_workers": self.config.cpu_workers,
                "architecture": "fixed_dual_model_gpu_cpu"
            },
            "language_analysis": language_analysis,
            "results": self.results
        }
        
        try:
            # Save comprehensive results file
            final_results_file = f"/home/samli/Documents/Python/Term_Verify/translation_results/fixed_dual_model_results_{self.session_id}.json"
            os.makedirs(os.path.dirname(final_results_file), exist_ok=True)
            
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Comprehensive results saved to: {final_results_file}")
            
            # Save summary file
            summary_file = final_results_file.replace('.json', '_summary.json')
            summary_data = {
                "processing_info": output_data["processing_info"],
                "language_summary": language_analysis["summary"],
                "result_count": len(self.results),
                "performance_metrics": {
                    "terms_per_second": len(self.results) / total_time if total_time > 0 else 0,
                    "total_hours": total_time / 3600,
                    "gpu_efficiency": "dual_model_pipeline",
                    "success_rate": (self.processed_terms / (self.processed_terms + self.failed_terms) * 100) if (self.processed_terms + self.failed_terms) > 0 else 0
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Failed to save final results: {e}")

    def _save_checkpoint(self):
        """Save checkpoint with fixed dual-model tracking"""
        try:
            checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/fixed_dual_{self.session_id}_checkpoint.json"
            results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/fixed_dual_{self.session_id}_results.json"
            
            # Ensure checkpoints directory exists
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            
            # Save checkpoint
            checkpoint_data = {
                'session_id': self.session_id,
                'processed_terms': self.processed_terms,
                'failed_terms': self.failed_terms,
                'total_terms': self.total_terms,
                'checkpoint_time': time.time(),
                'processing_rate': self.processed_terms / (time.time() - self.start_time) if self.start_time else 0,
                'config': {
                    'model_size': self.config.model_size,
                    'gpu_workers': self.config.gpu_workers,
                    'cpu_workers': self.config.cpu_workers,
                    'gpu_batch_size': self.config.gpu_batch_size,
                    'model_load_delay': self.config.model_load_delay
                },
                'runner_type': 'fixed_dual_model'
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Save results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Fixed dual-model checkpoint saved: {self.processed_terms} terms")
            
        except Exception as e:
            print(f"⚠️  Checkpoint save error: {e}")
    
    def run_fixed_dual_model_processing(self):
        """Run fixed dual-model ultra-fast processing"""
        print("🔧 STARTING FIXED DUAL-MODEL PROCESSING")
        print("=" * 60)
        
        try:
            # Load data
            dict_terms, non_dict_terms = self._load_data()
            all_terms = dict_terms + non_dict_terms
            self.total_terms = len(all_terms)
            
            if self.total_terms == 0:
                print("✅ All terms already processed!")
                return
            
            print(f"🎯 Processing {self.total_terms} terms with FIXED DUAL-MODEL configuration:")
            print(f"   • GPU Models: 2x {self.config.model_size} (Sequential loading)")
            print(f"   • Batch Size: {self.config.gpu_batch_size} per model")
            print(f"   • CPU Workers: {self.config.cpu_workers}")
            print(f"   • Queue Size: {self.config.max_queue_size} per GPU")
            print(f"   • Load Delay: {self.config.model_load_delay}s between models")
            print(f"   • Expected: Stable 2x speed improvement!")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print("🎮 GPU memory cleared")
            
            self.start_time = time.time()
            
            # Start GPU workers with sequential loading
            gpu_threads = []
            
            print("🔧 Starting GPU workers with sequential loading...")
            
            gpu_thread_1 = threading.Thread(
                target=self._gpu_translation_worker,
                args=(1, self.gpu_queue_1, self.gpu1_ready),
                name="FixedDualGPU-1"
            )
            gpu_thread_1.start()
            gpu_threads.append(gpu_thread_1)
            
            gpu_thread_2 = threading.Thread(
                target=self._gpu_translation_worker,
                args=(2, self.gpu_queue_2, self.gpu2_ready),
                name="FixedDualGPU-2"
            )
            gpu_thread_2.start()
            gpu_threads.append(gpu_thread_2)
            
            print(f"🎮 Started 2 GPU workers with sequential initialization")
            
            # Start result collector
            collector_thread = threading.Thread(
                target=self._result_collector,
                name="FixedDualCollector"
            )
            collector_thread.start()
            
            # Start CPU workers with ThreadPoolExecutor
            print(f"👷 Starting {self.config.cpu_workers} CPU workers...")
            
            # Distribute terms among CPU workers
            terms_per_worker = len(all_terms) // self.config.cpu_workers
            
            with ThreadPoolExecutor(max_workers=self.config.cpu_workers, thread_name_prefix="FixedDualCPU") as executor:
                futures = []
                
                for i in range(self.config.cpu_workers):
                    start_idx = i * terms_per_worker
                    end_idx = start_idx + terms_per_worker if i < self.config.cpu_workers - 1 else len(all_terms)
                    worker_terms = all_terms[start_idx:end_idx]
                    
                    future = executor.submit(self._cpu_worker, i + 1, worker_terms)
                    futures.append(future)
                
                print(f"✅ All workers started successfully!")
                print(f"🔧 FIXED DUAL-MODEL PROCESSING ACTIVE - STABLE & FAST!")
                
                # Wait for CPU workers to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"❌ Worker failed: {e}")
                
                print("✅ All CPU workers completed")
            
            # Signal GPU workers to stop
            self.gpu_queue_1.put(None)  # Shutdown signal
            self.gpu_queue_2.put(None)  # Shutdown signal
            
            # Wait for GPU workers
            for thread in gpu_threads:
                thread.join(timeout=60)
                if thread.is_alive():
                    print(f"⚠️  GPU thread {thread.name} did not stop gracefully")
            
            # Stop result collector
            self.stop_event.set()
            collector_thread.join(timeout=30)
            
            # Final statistics and comprehensive analysis
            total_time = time.time() - self.start_time
            total_processed = self.processed_terms + self.failed_terms
            
            # Generate comprehensive language analysis
            print("🔍 Generating comprehensive language analysis...")
            language_analysis = self._generate_language_analysis(self.results)
            
            # Save final results with analysis (same format as hybrid runner)
            self._save_final_results_with_analysis(total_time, language_analysis)
            
            print("\n🎉 FIXED DUAL-MODEL PROCESSING COMPLETED!")
            print("=" * 60)
            print(f"📊 Final Statistics:")
            print(f"   • Total Processed: {total_processed}/{self.total_terms}")
            print(f"   • Success Rate: {(self.processed_terms/total_processed*100):.1f}%")
            print(f"   • Processing Time: {total_time/3600:.1f} hours")
            print(f"   • Average Rate: {total_processed/total_time:.3f} terms/sec")
            print(f"   • Languages Analyzed: {language_analysis['summary']['total_languages_analyzed']}")
            print(f"   • High Borrowing Languages: {language_analysis['summary']['languages_with_high_borrowing']}")
            print(f"   • High Translation Languages: {language_analysis['summary']['languages_with_high_translation']}")
            print(f"   • Stable Dual-Model Performance!")
            print(f"   • Results saved to: fixed_dual_{self.session_id}_results.json")
            
        except KeyboardInterrupt:
            print("\n⏹️  Fixed dual-model processing interrupted by user")
            self.stop_event.set()
            self._save_checkpoint()
        except Exception as e:
            print(f"\n💥 Fixed dual-model processing failed: {e}")
            self.stop_event.set()
            self._save_checkpoint()
            raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--resume-from" and len(sys.argv) > 2:
        session_id = sys.argv[2]
        print(f"🔧 Starting fixed dual-model runner from session: {session_id}")
        config = FixedDualModelConfig()
        runner = FixedDualModelRunner(config=config, resume_session=session_id)
        runner.run_fixed_dual_model_processing()
    else:
        # Start fresh
        print("🔧 Starting fresh fixed dual-model processing...")
        config = FixedDualModelConfig()
        runner = FixedDualModelRunner(config=config)
        runner.run_fixed_dual_model_processing()
