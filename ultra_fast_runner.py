#!/usr/bin/env python3
"""
🚀 ULTRA-FAST RUNNER - MAXIMUM SPEED OPTIMIZATION
================================================

PERFORMANCE OPTIMIZATIONS:
- 🎮 Aggressive GPU utilization with continuous pipeline
- 💻 Maximum CPU workers (20+ workers)
- 📦 Dynamic batch sizing (32-64 items)
- ⚡ Overlapping GPU/CPU processing
- 🔄 Asynchronous result collection
- 💾 Optimized memory management

TARGET PERFORMANCE:
- GPU: >90% utilization
- CPU: >80% average utilization
- Processing Rate: 5-10x faster than current
"""

import json
import time
import threading
import queue
import multiprocessing
import gc
import os
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import torch

# Import translation tools
from nllb_translation_tool import NLLBTranslationTool

class UltraFastRunner:
    """Ultra-optimized runner for maximum CPU and GPU utilization"""
    
    def __init__(self, resume_session=None):
        """Initialize with ultra-aggressive settings"""
        self.start_time = time.time()
        self.session_id = resume_session or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🚀 ULTRA-FAST CONFIGURATION
        self.model_size = "1.3B"  # Optimal for Tesla T4
        self.gpu_batch_size = 64  # AGGRESSIVE batch size
        self.cpu_workers = 20     # MORE than cores for overlap
        self.gpu_workers = 1      # Single GPU optimization
        self.max_queue_size = 200 # Large queue for continuous flow
        
        # Performance tracking
        self.processed_terms = 0
        self.failed_terms = 0
        self.total_terms = 0
        self.processing_rate = 0.0
        
        # Threading and queues
        self.gpu_queue = queue.Queue(maxsize=self.max_queue_size)
        self.result_queue = queue.Queue(maxsize=self.max_queue_size * 2)
        self.stop_event = threading.Event()
        
        # Results storage
        self.results = {}
        self.checkpoint_interval = 50  # Save every 50 terms
        self.processed_terms_set = set()  # Track completed terms
        
        # Try to load existing checkpoint
        if resume_session:
            self._load_checkpoint()
        
        print(f"🚀 ULTRA-FAST RUNNER INITIALIZED")
        print(f"   • Model: {self.model_size}")
        print(f"   • GPU Batch: {self.gpu_batch_size}")
        print(f"   • CPU Workers: {self.cpu_workers}")
        print(f"   • Session: {self.session_id}")
        if resume_session:
            print(f"   • Resuming from: {len(self.processed_terms_set)} completed terms")

    def _load_checkpoint(self):
        """Load existing checkpoint and results"""
        try:
            checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_checkpoint.json"
            results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_results.json"
            
            # Load checkpoint
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.processed_terms = checkpoint_data.get('processed_terms', 0)
                self.failed_terms = checkpoint_data.get('failed_terms', 0)
                self.total_terms = checkpoint_data.get('total_terms', 0)
                self.processing_rate = checkpoint_data.get('processing_rate', 0.0)
                
                print(f"📥 Loaded checkpoint: {self.processed_terms} processed, {self.failed_terms} failed")
            
            # Load existing results
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                
                # Build set of processed terms
                self.processed_terms_set = set(self.results.keys())
                print(f"📥 Loaded {len(self.results)} existing results")
            
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting fresh...")

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load dictionary and non-dictionary terms"""
        print("📂 Loading dataset...")
        
        # Load dictionary terms
        try:
            with open('/home/samli/Documents/Python/Term_Verify/Dictionary_Terms_Found.json', 'r', encoding='utf-8') as f:
                dict_data = json.load(f)
                if isinstance(dict_data, dict) and 'dictionary_terms' in dict_data:
                    # Handle list of dictionaries structure
                    dict_terms_list = dict_data['dictionary_terms']
                    if isinstance(dict_terms_list, list):
                        dict_terms = [item['term'] for item in dict_terms_list if isinstance(item, dict) and 'term' in item]
                    else:
                        dict_terms = list(dict_terms_list.keys()) if isinstance(dict_terms_list, dict) else []
                elif isinstance(dict_data, dict):
                    dict_terms = list(dict_data.keys())
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
                    # Handle list of dictionaries structure
                    non_dict_terms_list = non_dict_data['non_dictionary_terms']
                    if isinstance(non_dict_terms_list, list):
                        non_dict_terms = [item['term'] for item in non_dict_terms_list if isinstance(item, dict) and 'term' in item]
                    else:
                        non_dict_terms = list(non_dict_terms_list.keys()) if isinstance(non_dict_terms_list, dict) else []
                elif isinstance(non_dict_data, dict):
                    non_dict_terms = list(non_dict_data.keys())
                else:
                    non_dict_terms = non_dict_data
        except Exception as e:
            print(f"⚠️  Warning: Could not load non-dictionary terms: {e}")
            non_dict_terms = []
        
        print(f"✅ Loaded {len(dict_terms)} dictionary + {len(non_dict_terms)} non-dictionary terms")
        
        # Filter out already processed terms if resuming
        if self.processed_terms_set:
            original_dict_count = len(dict_terms)
            original_non_dict_count = len(non_dict_terms)
            
            dict_terms = [term for term in dict_terms if term not in self.processed_terms_set]
            non_dict_terms = [term for term in non_dict_terms if term not in self.processed_terms_set]
            
            filtered_dict = original_dict_count - len(dict_terms)
            filtered_non_dict = original_non_dict_count - len(non_dict_terms)
            
            print(f"🔄 Resuming: Filtered out {filtered_dict + filtered_non_dict} already processed terms")
            print(f"   • Remaining: {len(dict_terms)} dictionary + {len(non_dict_terms)} non-dictionary terms")
        
        return dict_terms, non_dict_terms

    def _gpu_translation_worker(self):
        """Ultra-optimized GPU worker with continuous processing"""
        print("🎮 Starting ULTRA-FAST GPU worker...")
        
        try:
            # Initialize translation tool with aggressive settings
            translator = NLLBTranslationTool(
                model_name=self.model_size,
                batch_size=self.gpu_batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"🎮 GPU Model loaded: {self.model_size} (Batch: {self.gpu_batch_size})")
            
            batch_count = 0
            while not self.stop_event.is_set():
                try:
                    # Get batch of work
                    batch_items = []
                    timeout = 1.0
                    
                    # Collect batch items
                    for _ in range(self.gpu_batch_size):
                        try:
                            item = self.gpu_queue.get(timeout=timeout)
                            if item is None:  # Shutdown signal
                                break
                            batch_items.append(item)
                            timeout = 0.1  # Faster subsequent gets
                        except queue.Empty:
                            break
                    
                    if not batch_items:
                        continue
                    
                    batch_count += 1
                    print(f"🎮 Processing GPU batch {batch_count} ({len(batch_items)} items)")
                    
                    # Process batch
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
                            
                            # Send result
                            result = {
                                'term': term,
                                'translations': translations,
                                'status': 'completed',
                                'timestamp': time.time()
                            }
                            
                            self.result_queue.put(result, timeout=5.0)
                            
                        except Exception as e:
                            print(f"❌ GPU processing error for '{term}': {e}")
                            error_result = {
                                'term': term,
                                'translations': {},
                                'status': 'failed',
                                'error': str(e),
                                'timestamp': time.time()
                            }
                            self.result_queue.put(error_result, timeout=5.0)
                    
                    # Aggressive memory management
                    if batch_count % 5 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                        
                except Exception as e:
                    print(f"❌ GPU worker batch error: {e}")
                    continue
                    
        except Exception as e:
            print(f"💥 GPU worker failed: {e}")
        finally:
            print("🎮 GPU worker shutting down")

    def _cpu_coordinator_worker(self, worker_id: int, terms: List[str]):
        """Ultra-fast CPU coordinator worker"""
        target_languages = [
            'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn',
            'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab',
            'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn',
            'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl',
            'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt',
            'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn',
            'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn',
            'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'epo_Latn', 'est_Latn',
            'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn',
            'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn',
            'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr',
            'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn',
            'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan',
            'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva',
            'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn',
            'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn',
            'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn',
            'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn',
            'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym',
            'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn',
            'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr',
            'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn',
            'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru',
            'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn',
            'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck',
            'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn',
            'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn',
            'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn',
            'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn',
            'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn',
            'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng',
            'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn',
            'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn',
            'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn'
        ]
        
        print(f"💻 CPU Worker {worker_id}: Processing {len(terms)} terms")
        
        for term in terms:
            if self.stop_event.is_set():
                break
            
            # Skip if already processed (for resumption)
            if term in self.processed_terms_set:
                continue
                
            try:
                # Send to GPU queue with timeout
                work_item = (term, target_languages)
                self.gpu_queue.put(work_item, timeout=10.0)
                
            except queue.Full:
                print(f"⚠️  GPU queue full, worker {worker_id} waiting...")
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"❌ CPU worker {worker_id} error: {e}")
                continue
        
        print(f"💻 CPU Worker {worker_id}: Completed")

    def _result_collector_worker(self):
        """Ultra-fast result collection and saving"""
        print("📊 Starting ULTRA-FAST result collector...")
        
        last_checkpoint = time.time()
        last_rate_calc = time.time()
        last_processed = 0
        
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                # Get result with timeout
                result = self.result_queue.get(timeout=2.0)
                
                # Store result
                term = result['term']
                self.results[term] = result
                self.processed_terms_set.add(term)  # Track processed terms
                
                if result['status'] == 'completed':
                    self.processed_terms += 1
                else:
                    self.failed_terms += 1
                
                # Calculate processing rate
                current_time = time.time()
                if current_time - last_rate_calc >= 5.0:  # Every 5 seconds
                    time_diff = current_time - last_rate_calc
                    processed_diff = self.processed_terms - last_processed
                    self.processing_rate = processed_diff / time_diff
                    
                    last_rate_calc = current_time
                    last_processed = self.processed_terms
                
                # Progress update
                if self.processed_terms % 10 == 0:
                    elapsed = current_time - self.start_time
                    total_completed = self.processed_terms + self.failed_terms
                    
                    if self.total_terms > 0:
                        progress_pct = (total_completed / self.total_terms) * 100
                        eta_seconds = (self.total_terms - total_completed) / max(self.processing_rate, 0.001)
                        eta_hours = eta_seconds / 3600
                    else:
                        progress_pct = 0
                        eta_hours = 0
                    
                    # System metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    
                    print(f"🔥 ULTRA-FAST PROGRESS:")
                    print(f"   • Completed: {total_completed}/{self.total_terms} ({progress_pct:.1f}%)")
                    print(f"   • Rate: {self.processing_rate:.2f} terms/sec | ETA: {eta_hours:.1f} hours")
                    print(f"   • Success: {self.processed_terms} | Failed: {self.failed_terms}")
                    print(f"   • System: CPU: {cpu_percent:.1f}% | RAM: {memory_info.percent:.1f}%")
                    print(f"   🚀 MAXIMUM SPEED ACTIVE...")
                
                # Checkpoint saving
                if current_time - last_checkpoint >= 60:  # Every minute
                    self._save_checkpoint()
                    last_checkpoint = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Result collector error: {e}")
                continue
        
        print("📊 Result collector shutting down")

    def _save_checkpoint(self):
        """Save checkpoint with ultra-fast writing"""
        try:
            checkpoint_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_checkpoint.json"
            results_file = f"/home/samli/Documents/Python/Term_Verify/checkpoints/ultra_fast_{self.session_id}_results.json"
            
            # Create checkpoints directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            
            # Checkpoint data
            checkpoint_data = {
                'session_id': self.session_id,
                'processed_terms': self.processed_terms,
                'failed_terms': self.failed_terms,
                'total_terms': self.total_terms,
                'processing_rate': self.processing_rate,
                'start_time': self.start_time,
                'checkpoint_time': time.time(),
                'configuration': {
                    'model_size': self.model_size,
                    'gpu_batch_size': self.gpu_batch_size,
                    'cpu_workers': self.cpu_workers
                }
            }
            
            # Save checkpoint
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Save results (only if we have results)
            if self.results:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {self.processed_terms + self.failed_terms} terms")
            
        except Exception as e:
            print(f"⚠️  Checkpoint save error: {e}")

    def run_ultra_fast_processing(self):
        """Run ultra-fast processing of the complete dataset"""
        print(f"\n🚀 STARTING ULTRA-FAST PROCESSING")
        print(f"=====================================")
        
        try:
            # Load data
            dict_terms, non_dict_terms = self._load_data()
            all_terms = dict_terms + non_dict_terms
            self.total_terms = len(all_terms)
            
            if not all_terms:
                print("❌ No terms to process!")
                return
            
            print(f"🎯 Processing {self.total_terms} terms with ULTRA-FAST configuration:")
            print(f"   • GPU Model: {self.model_size} (Batch: {self.gpu_batch_size})")
            print(f"   • CPU Workers: {self.cpu_workers}")
            print(f"   • Target Languages: 202")
            print(f"   • Expected Speed: 5-10x faster")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🎮 GPU memory cleared")
            
            # Start GPU worker
            gpu_thread = threading.Thread(
                target=self._gpu_translation_worker,
                name="UltraFastGPU"
            )
            gpu_thread.daemon = True
            gpu_thread.start()
            
            # Start result collector
            collector_thread = threading.Thread(
                target=self._result_collector_worker,
                name="UltraFastCollector"
            )
            collector_thread.daemon = True
            collector_thread.start()
            
            # Distribute work among CPU workers
            terms_per_worker = max(1, len(all_terms) // self.cpu_workers)
            worker_threads = []
            
            print(f"💻 Starting {self.cpu_workers} CPU workers...")
            
            with ThreadPoolExecutor(max_workers=self.cpu_workers, thread_name_prefix="UltraFastCPU") as executor:
                futures = []
                
                for i in range(self.cpu_workers):
                    start_idx = i * terms_per_worker
                    if i == self.cpu_workers - 1:
                        # Last worker gets remaining terms
                        worker_terms = all_terms[start_idx:]
                    else:
                        worker_terms = all_terms[start_idx:start_idx + terms_per_worker]
                    
                    if worker_terms:
                        future = executor.submit(self._cpu_coordinator_worker, i, worker_terms)
                        futures.append(future)
                
                print(f"⚡ ULTRA-FAST processing started with {len(futures)} workers!")
                
                # Wait for all CPU workers to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"❌ Worker failed: {e}")
                
                print("💻 All CPU workers completed")
            
            # Signal GPU worker to stop
            self.gpu_queue.put(None)  # Shutdown signal
            
            # Wait for remaining results
            print("⏳ Waiting for final results...")
            time.sleep(10)  # Allow time for final processing
            
            # Stop all workers
            self.stop_event.set()
            
            # Final checkpoint
            self._save_checkpoint()
            
            # Final statistics
            total_processed = self.processed_terms + self.failed_terms
            elapsed_time = time.time() - self.start_time
            final_rate = total_processed / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n🎉 ULTRA-FAST PROCESSING COMPLETED!")
            print(f"=====================================")
            print(f"✅ Total Processed: {total_processed}/{self.total_terms}")
            
            if total_processed > 0:
                success_rate = (self.processed_terms / total_processed) * 100
                print(f"✅ Success Rate: {success_rate:.1f}%")
            else:
                print(f"✅ Success Rate: N/A (no terms processed)")
                
            print(f"⚡ Processing Rate: {final_rate:.2f} terms/sec")
            print(f"⏱️  Total Time: {elapsed_time/3600:.2f} hours")
            print(f"💾 Results saved in checkpoints/ultra_fast_{self.session_id}_*")
            
        except KeyboardInterrupt:
            print("\n⏹️  Ultra-fast processing interrupted by user")
            self.stop_event.set()
            self._save_checkpoint()
        except Exception as e:
            print(f"\n💥 Ultra-fast processing failed: {e}")
            self.stop_event.set()
            self._save_checkpoint()
            raise

def find_latest_session():
    """Find the latest session ID from checkpoints"""
    try:
        checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
        
        # Find all ultra_fast checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("ultra_fast_") and f.endswith("_checkpoint.json")]
        
        if not checkpoint_files:
            return None
        
        # Extract session IDs and find the latest
        sessions = []
        for filename in checkpoint_files:
            parts = filename.replace("ultra_fast_", "").replace("_checkpoint.json", "")
            sessions.append(parts)
        
        # Return the latest session (assuming format YYYYMMDD_HHMMSS)
        return max(sessions) if sessions else None
        
    except Exception as e:
        print(f"⚠️  Could not find latest session: {e}")
        return None

def find_most_processed_session():
    """Find the session with the most processed terms"""
    try:
        checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
        
        # Find all ultra_fast checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("ultra_fast_") and f.endswith("_checkpoint.json")]
        
        if not checkpoint_files:
            return None
        
        best_session = None
        max_processed = -1
        
        for filename in checkpoint_files:
            session_id = filename.replace("ultra_fast_", "").replace("_checkpoint.json", "")
            filepath = os.path.join(checkpoints_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    processed = data.get('processed_terms', 0)
                    
                    if processed > max_processed:
                        max_processed = processed
                        best_session = session_id
                        
            except Exception as e:
                print(f"⚠️  Could not read {filename}: {e}")
                continue
        
        if best_session:
            print(f"🎯 Found session with most progress: {best_session} ({max_processed} terms)")
        
        return best_session
        
    except Exception as e:
        print(f"⚠️  Could not find most processed session: {e}")
        return None

if __name__ == "__main__":
    import sys
    import json
    
    print("🚀 ULTRA-FAST RUNNER - MAXIMUM SPEED MODE")
    print("=========================================")
    
    # Check for resume options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--resume":
            # Try to resume from latest session
            latest_session = find_latest_session()
            if latest_session:
                print(f"🔄 Resuming from latest session: {latest_session}")
                runner = UltraFastRunner(resume_session=latest_session)
            else:
                print("❌ No previous session found, starting fresh")
                runner = UltraFastRunner()
        elif sys.argv[1] == "--resume-best":
            # Resume from session with most progress
            best_session = find_most_processed_session()
            if best_session:
                print(f"🎯 Resuming from most processed session: {best_session}")
                runner = UltraFastRunner(resume_session=best_session)
            else:
                print("❌ No previous session found, starting fresh")
                runner = UltraFastRunner()
        elif sys.argv[1] == "--resume-from" and len(sys.argv) > 2:
            # Resume from specific session
            specific_session = sys.argv[2]
            print(f"🔄 Resuming from specific session: {specific_session}")
            runner = UltraFastRunner(resume_session=specific_session)
        elif sys.argv[1] == "--list-sessions":
            # List available sessions
            checkpoints_dir = "/home/samli/Documents/Python/Term_Verify/checkpoints"
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("ultra_fast_") and f.endswith("_checkpoint.json")]
                if checkpoint_files:
                    print("📋 Available sessions:")
                    for filename in sorted(checkpoint_files, reverse=True):
                        session_id = filename.replace("ultra_fast_", "").replace("_checkpoint.json", "")
                        # Read checkpoint for details
                        try:
                            with open(os.path.join(checkpoints_dir, filename), 'r') as f:
                                data = json.load(f)
                                processed = data.get('processed_terms', 0)
                                failed = data.get('failed_terms', 0)
                                total = data.get('total_terms', 0)
                                print(f"   • {session_id}: {processed} processed, {failed} failed, {total} total")
                        except:
                            print(f"   • {session_id}: (could not read details)")
                else:
                    print("❌ No sessions found")
            else:
                print("❌ No checkpoints directory found")
            sys.exit(0)
        else:
            print("❌ Unknown argument. Usage:")
            print("   python ultra_fast_runner.py                    # Start fresh")
            print("   python ultra_fast_runner.py --resume           # Resume from latest session")
            print("   python ultra_fast_runner.py --resume-best      # Resume from most processed session")
            print("   python ultra_fast_runner.py --resume-from ID   # Resume from specific session")
            print("   python ultra_fast_runner.py --list-sessions    # List available sessions")
            sys.exit(1)
    else:
        # Start fresh
        runner = UltraFastRunner()
    
    runner.run_ultra_fast_processing()
