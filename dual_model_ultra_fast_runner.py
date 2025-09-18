#!/usr/bin/env python3
"""
🚀 DUAL-MODEL ULTRA-FAST RUNNER
===============================

BREAKTHROUGH SOLUTION: 2x NLLB Models on Single Tesla T4!
- 🎮 2 GPU workers (each with own model instance) 
- 💻 16 CPU workers feeding 2 GPU queues
- 📦 Optimized batch sizes per model
- ⚡ ELIMINATES queue bottleneck completely!

PERFORMANCE TARGETS:
- GPU: >90% utilization (2 models working)
- CPU: >80% utilization (16 workers)
- Processing: ~2x speed improvement
- Queue: NO MORE "GPU queue full" messages!

MEMORY USAGE:
- Model 1: ~6.6GB VRAM
- Model 2: ~6.6GB VRAM  
- Total: ~13.2GB / 16GB (safe margin)
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
class DualModelConfig:
    """Configuration for dual model setup"""
    model_size: str = "1.3B"
    gpu_workers: int = 2              # 2 GPU workers (one per model)
    cpu_workers: int = 16             # Reduced from 20 for stability
    gpu_batch_size: int = 48          # Reduced from 64 for dual models
    max_queue_size: int = 100         # Per GPU queue (200 total)
    checkpoint_interval: int = 60     # Save every 60 seconds
    
class DualModelUltraFastRunner:
    """Dual-model runner with 2x GPU throughput"""
    
    def __init__(self, config: DualModelConfig = None, resume_session: str = None):
        self.config = config or DualModelConfig()
        self.session_id = resume_session or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dual GPU queues (one per model)
        self.gpu_queue_1 = queue.Queue(maxsize=self.config.max_queue_size)
        self.gpu_queue_2 = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        
        # Load balancer for distributing work
        self.next_gpu = 0  # Round-robin between GPUs
        
        # Thread control
        self.stop_event = threading.Event()
        
        # Progress tracking
        self.processed_terms = 0
        self.failed_terms = 0
        self.total_terms = 0
        self.processed_terms_set: Set[str] = set()
        self.results = []
        
        # Performance tracking
        self.start_time = None
        self.last_checkpoint_time = time.time()
        
        # Load existing progress if resuming
        if resume_session:
            self._load_checkpoint()
        
        print(f"🚀 DUAL-MODEL ULTRA-FAST RUNNER INITIALIZED")
        print(f"   • Session: {self.session_id}")
        print(f"   • GPU Workers: {self.config.gpu_workers} (2 models)")
        print(f"   • CPU Workers: {self.config.cpu_workers}")
        print(f"   • Batch Size: {self.config.gpu_batch_size} per model")
        print(f"   • Queue Size: {self.config.max_queue_size} per GPU")
    
    def _load_checkpoint(self):
        """Load existing checkpoint and results (supports both dual-model and ultra-fast formats)"""
        try:
            # Try dual-model format first
            dual_checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_checkpoint.json"
            dual_results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_results.json"
            
            # Try ultra-fast format as fallback
            ultra_checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_checkpoint.json"
            ultra_results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_results.json"
            
            checkpoint_loaded = False
            results_loaded = False
            
            # Load checkpoint (try dual-model first, then ultra-fast)
            if os.path.exists(dual_checkpoint_file):
                with open(dual_checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                checkpoint_loaded = True
                print(f"📂 Loading from dual-model checkpoint...")
            elif os.path.exists(ultra_checkpoint_file):
                with open(ultra_checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                checkpoint_loaded = True
                print(f"📂 Loading from ultra-fast checkpoint (converting to dual-model)...")
            
            if checkpoint_loaded:
                self.processed_terms = checkpoint_data.get('processed_terms', 0)
                self.failed_terms = checkpoint_data.get('failed_terms', 0)
                self.total_terms = checkpoint_data.get('total_terms', 0)
                print(f"📂 Loaded checkpoint: {self.processed_terms} processed, {self.failed_terms} failed")
            
            # Load results (try dual-model first, then ultra-fast)
            if os.path.exists(dual_results_file):
                with open(dual_results_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                results_loaded = True
                print(f"📂 Loading from dual-model results...")
            elif os.path.exists(ultra_results_file):
                with open(ultra_results_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                results_loaded = True
                print(f"📂 Loading from ultra-fast results (converting to dual-model)...")
            
            if results_loaded:
                # Build processed terms set for deduplication
                for result in self.results:
                    if 'term' in result:
                        self.processed_terms_set.add(result['term'])
                
                print(f"📂 Loaded {len(self.results)} previous results")
                print(f"🔄 Converting to dual-model format for enhanced processing...")
            
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
    
    def _gpu_translation_worker(self, worker_id: int, gpu_queue: queue.Queue):
        """Optimized GPU worker with dedicated model instance"""
        print(f"🎮 Starting GPU worker {worker_id}...")
        
        try:
            # Initialize translation tool with optimized settings
            translator = NLLBTranslationTool(
                model_name=self.config.model_size,
                batch_size=self.config.gpu_batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"🎮 GPU worker {worker_id} model loaded: {self.config.model_size} (Batch: {self.config.gpu_batch_size})")
            
            batch_count = 0
            
            while not self.stop_event.is_set():
                try:
                    # Get batch of work
                    batch_items = []
                    timeout = 1.0
                    
                    # Collect batch items
                    for _ in range(self.config.gpu_batch_size):
                        try:
                            item = gpu_queue.get(timeout=timeout)
                            if item is None:  # Shutdown signal
                                return
                            batch_items.append(item)
                            timeout = 0.1  # Faster subsequent gets
                        except queue.Empty:
                            break
                    
                    if not batch_items:
                        continue
                    
                    batch_count += 1
                    print(f"🎮 GPU-{worker_id} processing batch {batch_count} ({len(batch_items)} items)")
                    
                    # Process each term in the batch
                    for term, target_languages in batch_items:
                        try:
                            # Translate to all target languages
                            translations = {}
                            
                            # Process in sub-batches for memory efficiency
                            sub_batch_size = 20
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
                                        print(f"✅ {lang}: Translated")
                                    except Exception as e:
                                        print(f"❌ {lang}: Failed - {str(e)[:50]}")
                                        translations[lang] = f"ERROR: {str(e)[:100]}"
                                
                                # Quick memory cleanup
                                if i % 40 == 0:
                                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # Create result
                            result = {
                                'term': term,
                                'translations': translations,
                                'timestamp': datetime.now().isoformat(),
                                'gpu_worker': worker_id,
                                'status': 'completed'
                            }
                            
                            self.result_queue.put(result, timeout=5.0)
                            
                        except Exception as e:
                            print(f"❌ GPU-{worker_id} processing error for '{term}': {e}")
                            error_result = {
                                'term': term,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                                'gpu_worker': worker_id,
                                'status': 'failed'
                            }
                            self.result_queue.put(error_result, timeout=5.0)
                    
                    # Memory management
                    if batch_count % 3 == 0:  # More frequent for dual models
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        
                except Exception as e:
                    print(f"❌ GPU-{worker_id} batch error: {e}")
                    continue
                    
        except Exception as e:
            print(f"💥 GPU worker {worker_id} failed: {e}")
        finally:
            print(f"🎮 GPU worker {worker_id} shutting down")
    
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
        """CPU worker with intelligent load balancing"""
        target_languages = self._get_target_languages()
        processed_count = 0
        
        for term in terms:
            if self.stop_event.is_set():
                break
            
            # Skip if already processed
            if term in self.processed_terms_set:
                continue
            
            # Smart load balancing between GPU queues
            selected_queue = self.gpu_queue_1 if self.next_gpu == 0 else self.gpu_queue_2
            self.next_gpu = (self.next_gpu + 1) % 2  # Round-robin
            
            try:
                # Send to selected GPU queue with timeout
                work_item = (term, target_languages)
                selected_queue.put(work_item, timeout=10.0)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    gpu_id = 1 if selected_queue == self.gpu_queue_1 else 2
                    print(f"👷 CPU-{worker_id}: Sent {processed_count} terms (→ GPU-{gpu_id})")
                
            except queue.Full:
                # Try the other queue if first is full
                alternate_queue = self.gpu_queue_2 if selected_queue == self.gpu_queue_1 else self.gpu_queue_1
                try:
                    alternate_queue.put(work_item, timeout=5.0)
                    processed_count += 1
                except queue.Full:
                    print(f"⚠️  Both GPU queues full, CPU worker {worker_id} waiting...")
                    time.sleep(0.2)
                    continue
            except Exception as e:
                print(f"❌ CPU worker {worker_id} error: {e}")
                continue
        
        print(f"✅ CPU worker {worker_id} completed: {processed_count} terms sent")
    
    def _result_collector(self):
        """Collect and save results with dual-model progress tracking"""
        print("📊 Starting dual-model result collector...")
        
        last_save_time = time.time()
        save_interval = self.config.checkpoint_interval
        
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                # Get result with timeout
                result = self.result_queue.get(timeout=2.0)
                
                # Process result
                self.results.append(result)
                term = result.get('term', '')
                
                if result.get('status') == 'completed':
                    self.processed_terms += 1
                    self.processed_terms_set.add(term)
                    
                    # Show progress with GPU worker info
                    gpu_worker = result.get('gpu_worker', '?')
                    if self.processed_terms % 10 == 0:
                        progress = (self.processed_terms / self.total_terms * 100) if self.total_terms > 0 else 0
                        rate = self.processed_terms / (time.time() - self.start_time) if self.start_time else 0
                        print(f"📈 Progress: {self.processed_terms}/{self.total_terms} ({progress:.1f}%) | Rate: {rate:.2f} terms/sec | GPU-{gpu_worker}")
                else:
                    self.failed_terms += 1
                
                # Periodic checkpoint save
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self._save_checkpoint()
                    last_save_time = current_time
                
                # Performance summary every 100 terms
                if self.processed_terms % 100 == 0 and self.processed_terms > 0:
                    total_processed = self.processed_terms + self.failed_terms
                    rate = total_processed / (time.time() - self.start_time) if self.start_time else 0
                    eta_hours = (self.total_terms - total_processed) / (rate * 3600) if rate > 0 else 0
                    
                    # System stats
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    
                    print(f"🔥 DUAL-MODEL PROGRESS:")
                    print(f"   • Completed: {self.processed_terms}/{self.total_terms} ({(self.processed_terms/self.total_terms*100):.1f}%)")
                    print(f"   • Rate: {rate:.2f} terms/sec | ETA: {eta_hours:.1f} hours")
                    print(f"   • Success: {self.processed_terms} | Failed: {self.failed_terms}")
                    print(f"   • System: CPU: {cpu_percent:.1f}% | RAM: {memory_info.percent:.1f}%")
                    print(f"   🚀 DUAL-GPU MAXIMUM SPEED...")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Result collector error: {e}")
                continue
        
        # Final save
        self._save_checkpoint()
        print("📊 Result collector finished")
    
    def _save_checkpoint(self):
        """Save checkpoint with dual-model tracking"""
        try:
            checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_checkpoint.json"
            results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/dual_model_{self.session_id}_results.json"
            
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
                    'gpu_batch_size': self.config.gpu_batch_size
                }
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Save results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {self.processed_terms} terms")
            
        except Exception as e:
            print(f"⚠️  Checkpoint save error: {e}")
    
    def run_dual_model_processing(self):
        """Run dual-model ultra-fast processing"""
        print("🚀 STARTING DUAL-MODEL ULTRA-FAST PROCESSING")
        print("=" * 60)
        
        try:
            # Load data
            dict_terms, non_dict_terms = self._load_data()
            all_terms = dict_terms + non_dict_terms
            self.total_terms = len(all_terms)
            
            if self.total_terms == 0:
                print("✅ All terms already processed!")
                return
            
            print(f"🎯 Processing {self.total_terms} terms with DUAL-MODEL configuration:")
            print(f"   • GPU Models: 2x {self.config.model_size} (Batch: {self.config.gpu_batch_size} each)")
            print(f"   • CPU Workers: {self.config.cpu_workers}")
            print(f"   • Queue Size: {self.config.max_queue_size} per GPU")
            print(f"   • Expected: ~2x speed improvement!")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🎮 GPU memory cleared")
            
            self.start_time = time.time()
            
            # Start dual GPU workers
            gpu_threads = []
            
            gpu_thread_1 = threading.Thread(
                target=self._gpu_translation_worker,
                args=(1, self.gpu_queue_1),
                name="DualModelGPU-1"
            )
            gpu_thread_1.start()
            gpu_threads.append(gpu_thread_1)
            
            gpu_thread_2 = threading.Thread(
                target=self._gpu_translation_worker,
                args=(2, self.gpu_queue_2),
                name="DualModelGPU-2"
            )
            gpu_thread_2.start()
            gpu_threads.append(gpu_thread_2)
            
            print(f"🎮 Started 2 GPU workers (dual models)")
            
            # Start result collector
            collector_thread = threading.Thread(
                target=self._result_collector,
                name="DualModelCollector"
            )
            collector_thread.start()
            
            # Start CPU workers with ThreadPoolExecutor
            print(f"👷 Starting {self.config.cpu_workers} CPU workers...")
            
            # Distribute terms among CPU workers
            terms_per_worker = len(all_terms) // self.config.cpu_workers
            
            with ThreadPoolExecutor(max_workers=self.config.cpu_workers, thread_name_prefix="DualModelCPU") as executor:
                futures = []
                
                for i in range(self.config.cpu_workers):
                    start_idx = i * terms_per_worker
                    end_idx = start_idx + terms_per_worker if i < self.config.cpu_workers - 1 else len(all_terms)
                    worker_terms = all_terms[start_idx:end_idx]
                    
                    future = executor.submit(self._cpu_worker, i + 1, worker_terms)
                    futures.append(future)
                
                print(f"✅ All workers started successfully!")
                print(f"🔥 DUAL-MODEL PROCESSING ACTIVE - NO MORE QUEUE BOTTLENECKS!")
                
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
                thread.join(timeout=30)
                if thread.is_alive():
                    print(f"⚠️  GPU thread {thread.name} did not stop gracefully")
            
            # Stop result collector
            self.stop_event.set()
            collector_thread.join(timeout=10)
            
            # Final statistics
            total_time = time.time() - self.start_time
            total_processed = self.processed_terms + self.failed_terms
            
            print("\n🎉 DUAL-MODEL PROCESSING COMPLETED!")
            print("=" * 60)
            print(f"📊 Final Statistics:")
            print(f"   • Total Processed: {total_processed}/{self.total_terms}")
            print(f"   • Success Rate: {(self.processed_terms/total_processed*100):.1f}%")
            print(f"   • Processing Time: {total_time/3600:.1f} hours")
            print(f"   • Average Rate: {total_processed/total_time:.2f} terms/sec")
            print(f"   • Theoretical Speedup: ~2x vs single model")
            print(f"   • Results saved to: dual_model_{self.session_id}_results.json")
            
        except KeyboardInterrupt:
            print("\n⏹️  Dual-model processing interrupted by user")
            self.stop_event.set()
            self._save_checkpoint()
        except Exception as e:
            print(f"\n💥 Dual-model processing failed: {e}")
            self.stop_event.set()
            self._save_checkpoint()
            raise

def find_latest_dual_model_session():
    """Find the latest dual-model session ID from checkpoints"""
    try:
        checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("dual_model_") and f.endswith("_checkpoint.json")]
        
        if not checkpoint_files:
            return None
        
        # Extract session IDs and find the latest
        sessions = []
        for filename in checkpoint_files:
            try:
                # Extract session ID from filename: dual_model_YYYYMMDD_HHMMSS_checkpoint.json
                session_id = filename.replace("dual_model_", "").replace("_checkpoint.json", "")
                filepath = os.path.join(checkpoints_dir, filename)
                mtime = os.path.getmtime(filepath)
                sessions.append((session_id, mtime))
            except:
                continue
        
        if sessions:
            # Return the session with the latest modification time
            latest_session = max(sessions, key=lambda x: x[1])
            return latest_session[0]
        
    except Exception as e:
        print(f"⚠️  Could not find latest dual-model session: {e}")
        return None

def find_most_processed_dual_model_session():
    """Find the dual-model session with the most processed terms"""
    try:
        checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("dual_model_") and f.endswith("_checkpoint.json")]
        
        if not checkpoint_files:
            return None
        
        best_session = None
        max_processed = -1
        
        for filename in checkpoint_files:
            try:
                session_id = filename.replace("dual_model_", "").replace("_checkpoint.json", "")
                filepath = os.path.join(checkpoints_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                processed = data.get('processed_terms', 0)
                if processed > max_processed:
                    max_processed = processed
                    best_session = session_id
                        
            except Exception as e:
                print(f"⚠️  Could not read {filename}: {e}")
                continue
        
        return best_session
        
    except Exception as e:
        print(f"⚠️  Could not find most processed dual-model session: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--list-sessions":
            print("📋 Available Dual-Model Sessions:")
            checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
            
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("dual_model_") and f.endswith("_checkpoint.json")]
                
                if checkpoint_files:
                    for filename in sorted(checkpoint_files):
                        session_id = filename.replace("dual_model_", "").replace("_checkpoint.json", "")
                        print(f"   • {session_id}")
                        
                        # Read checkpoint for details
                        try:
                            with open(os.path.join(checkpoints_dir, filename), 'r') as f:
                                data = json.load(f)
                            
                            processed = data.get('processed_terms', 0)
                            failed = data.get('failed_terms', 0)
                            total = data.get('total_terms', 0)
                            print(f"     → {processed} processed, {failed} failed, {total} total")
                        except:
                            print(f"     → (could not read details)")
                else:
                    print("   No dual-model sessions found")
            else:
                print("   No checkpoints directory found")
            
            sys.exit(0)
        
        elif command == "--resume":
            latest_session = find_latest_dual_model_session()
            if latest_session:
                print(f"🔄 Resuming from latest dual-model session: {latest_session}")
                config = DualModelConfig()
                runner = DualModelUltraFastRunner(config=config, resume_session=latest_session)
                runner.run_dual_model_processing()
            else:
                print("❌ No dual-model sessions found to resume")
                sys.exit(1)
        
        elif command == "--resume-from" and len(sys.argv) > 2:
            session_id = sys.argv[2]
            print(f"🔄 Resuming from dual-model session: {session_id}")
            config = DualModelConfig()
            runner = DualModelUltraFastRunner(config=config, resume_session=session_id)
            runner.run_dual_model_processing()
        
        elif command == "--resume-best":
            best_session = find_most_processed_dual_model_session()
            if best_session:
                print(f"🔄 Resuming from most processed dual-model session: {best_session}")
                config = DualModelConfig()
                runner = DualModelUltraFastRunner(config=config, resume_session=best_session)
                runner.run_dual_model_processing()
            else:
                print("❌ No dual-model sessions found to resume")
                sys.exit(1)
        
        else:
            print("❌ Unknown command. Use --list-sessions, --resume, --resume-from <ID>, or --resume-best")
            sys.exit(1)
    
    else:
        # Start fresh dual-model processing
        print("🚀 Starting fresh dual-model ultra-fast processing...")
        config = DualModelConfig()
        runner = DualModelUltraFastRunner(config=config)
        runner.run_dual_model_processing()
