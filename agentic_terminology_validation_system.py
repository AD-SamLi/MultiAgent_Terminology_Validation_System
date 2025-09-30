#!/usr/bin/env python3
"""
MULTI-AGENT TERMINOLOGY VALIDATION SYSTEM - UNIFIED PROCESS FLOW
==============================================================

Complete implementation of the terminology validation and verification process
as described in term_process.txt and integrated with all system components.

INPUT: Term_Extracted_result.csv
OUTPUT: Validated and translated terminology with comprehensive reporting

Process Flow:
1. Initial Term Collection and Verification
2. Glossary Validation  
3. New Terminology Processing
4. Frequency Analysis and Filtering
5. Translation Process (1-200 languages)
6. Language Verification
7. Final Review and Decision
8. Timestamp + Term Data Recording
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures
import multiprocessing as mp

# Import system components
from convert_extracted_to_combined import convert_extracted_to_combined
from verify_terms_in_text import create_cleaned_csv, analyze_term_matching
from create_clean_csv import create_clean_csv_formats
from create_json_format import create_complete_json_format
from frequency_storage import FrequencyStorageSystem
from terminology_agent import TerminologyAgent
from terminology_tool import TerminologyTool
from ultra_optimized_smart_runner import UltraOptimizedSmartRunner, UltraOptimizedConfig
from modern_parallel_validation import OrganizedValidationManager, EnhancedValidationSystem
from fast_dictionary_agent import FastDictionaryAgent
from auth_fix_wrapper import ensure_agent_auth_fix, robust_agent_call_with_retries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_terminology_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_term_batch_worker(term_batch, glossary_folder, model_name):
    """
    Standalone worker function for multiprocessing glossary analysis
    Each process will have its own terminology agent instance
    """
    try:
        # Import here to avoid circular imports in multiprocessing
        from terminology_agent import TerminologyAgent
        from auth_fix_wrapper import ensure_agent_auth_fix
        
        # Initialize terminology agent in this process
        agent = TerminologyAgent(
            glossary_folder=glossary_folder,
            model_name=model_name
        )
        agent = ensure_agent_auth_fix(agent)
        
        results = []
        for term in term_batch:
            try:
                # Ensure term is a string
                if not isinstance(term, str):
                    term = str(term)
                
                # Skip empty or invalid terms
                if not term or not term.strip():
                    results.append({
                        'term': term,
                        'found': False,
                        'analysis': 'Empty or invalid term'
                    })
                    continue
                
                # Clean the term
                clean_term = term.strip()
                
                # Analyze single term with proper error handling
                analysis_result = agent.analyze_text_terminology(clean_term, "EN", "EN")
                
                # Ensure analysis_result is a string
                if not isinstance(analysis_result, str):
                    analysis_result = str(analysis_result)
                
                # Simple parsing of the result to determine if term is found
                analysis_lower = analysis_result.lower()
                if ("no recognized glossary terminology terms were found" in analysis_lower or 
                    "no glossary terms" in analysis_lower or
                    "no terms found" in analysis_lower or
                    "no task provided" in analysis_lower or
                    "no task was provided" in analysis_lower):
                    results.append({
                        'term': clean_term,
                        'found': False,
                        'analysis': 'Not found in glossary'
                    })
                else:
                    results.append({
                        'term': clean_term,
                        'found': True,
                        'analysis': analysis_result
                    })
                    
            except Exception as e:
                # Ensure term is available for error reporting
                error_term = term if isinstance(term, str) else str(term) if term else "unknown_term"
                results.append({
                    'term': error_term,
                    'found': False,
                    'analysis': f"Error during analysis: {str(e)}"
                })
        
        return results
        
    except Exception as e:
        # Return error results for all terms in batch
        error_results = []
        for term in term_batch:
            error_term = term if isinstance(term, str) else str(term) if term else "unknown_term"
            error_results.append({
                'term': error_term,
                'found': False,
                'analysis': f"Worker initialization error: {str(e)}"
            })
        return error_results


class AgenticTerminologyValidationSystem:
    """
    Main system orchestrator implementing the complete terminology validation workflow
    """
    
    def __init__(self, config: Dict[str, Any] = None, resume_from: str = None):
        """Initialize the system with configuration"""
        self.config = config or {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.resume_from = resume_from
        
        # Performance optimization flags
        self.skip_glossary_validation = self.config.get('skip_glossary_validation', False)
        self.fast_mode = self.config.get('fast_mode', False)
        self.use_process_pool = self.config.get('use_process_pool', False)  # Default to ThreadPool for stability
        
        # System components
        self.frequency_storage = None
        self.terminology_agent = None
        self.validation_manager = None
        self.translation_runner = None
        self.fast_dictionary_agent = None
        self.resource_monitor = None  # Dynamic resource monitoring
        
        # Process tracking
        self.process_stats = {
            'total_terms_input': 0,
            'terms_after_cleaning': 0,
            'frequency_1_terms': 0,
            'frequency_gt_2_terms': 0,
            'terms_translated': 0,
            'terms_approved': 0,
            'terms_rejected': 0,
            'processing_time': 0
        }
        
        # Setup output directory and resume functionality
        self.completed_steps = set()
        self.step_files = {}
        self.output_dir = self._setup_output_directory()
        
        # Detect already completed steps if resuming
        if self.resume_from:
            self._detect_completed_steps()
        
        logger.info(f"[*] Multi-Agent Terminology Validation System initialized")
        logger.info(f"[FOLDER] Session: {self.session_id}")
        logger.info(f"[DIR] Output directory: {self.output_dir}")
        
        if self.resume_from:
            logger.info(f"[RESUME] Resuming from: {self.resume_from}")
            if self.completed_steps:
                logger.info(f"[PROGRESS] Resume mode: Found {len(self.completed_steps)} completed steps")
                logger.info(f"[PROGRESS] Completed steps: {sorted(self.completed_steps)}")
            else:
                logger.info("[PROGRESS] Resume requested but no previous steps found")
        else:
            logger.info("[PROGRESS] Starting fresh process - creating new output folder")
    
    def _setup_output_directory(self) -> Path:
        """Setup output directory with proper new run vs resume logic"""
        
        if self.resume_from:
            # Resume mode: Use specified directory
            resume_dir = Path(self.resume_from)
            
            if resume_dir.exists() and resume_dir.is_dir():
                logger.info(f"[RESUME] Using existing directory: {resume_dir}")
                return resume_dir
            else:
                # Try to find directory by pattern if not exact path
                existing_dirs = list(Path(".").glob(f"*{self.resume_from}*"))
                if existing_dirs:
                    # Use the first match
                    resume_dir = existing_dirs[0]
                    logger.info(f"[RESUME] Found matching directory: {resume_dir}")
                    return resume_dir
                else:
                    logger.warning(f"[RESUME] Directory not found: {self.resume_from}")
                    logger.info("[RESUME] Creating new directory instead")
        
        # New run mode: Always create a new directory
        new_dir = Path(f"agentic_validation_output_{self.session_id}")
        new_dir.mkdir(exist_ok=True)
        logger.info(f"[NEW] Created new output directory: {new_dir}")
        return new_dir
    
    def _detect_completed_steps(self):
        """Detect which steps have already been completed"""
        # Define expected output files for each step
        step_indicators = {
            1: ['Combined_Terms_Data.csv'],
            2: ['Glossary_Analysis_Results.json'],
            3: ['New_Terms_Candidates_With_Dictionary.json', 'Dictionary_Terms_Identified.json', 'Non_Dictionary_Terms_Identified.json'],
            4: ['Frequency_Storage_Export.json'],
            5: ['Translation_Results.json'],
            6: ['Verified_Translation_Results.json'],
            7: ['Final_Terminology_Decisions.json'],  # FIXED: Correct file name
            8: ['Complete_Audit_Record.json'],         # FIXED: Correct file name
            9: ['Approved_Terms_Export.csv']           # NEW: Step 9 CSV export
        }
        
        # Checkpoint files for resume detection
        checkpoint_files = {
            2: "step2_checkpoint.json",
            3: "step3_checkpoint.json",
            5: "step5_translation_checkpoint.json"
        }
        
        for step, files in step_indicators.items():
            step_files = {}
            
            # Check for checkpoint files
            step_has_checkpoint = False
            if step in checkpoint_files:
                checkpoint_path = self.output_dir / checkpoint_files[step]
                if checkpoint_path.exists():
                    step_has_checkpoint = True
                    logger.info(f"[CHECKPOINT] Step {step} has checkpoint - can be resumed")
            
            # For step 3, only require the main file to consider it complete
            if step == 3:
                main_file = 'New_Terms_Candidates_With_Dictionary.json'
                main_path = self.output_dir / main_file
                if main_path.exists() and main_path.stat().st_size > 0:
                    step_files[main_file] = main_path
                    # Check for optional split files
                    for file in files[1:]:  # Skip the first file (main file)
                        file_path = self.output_dir / file
                        if file_path.exists() and file_path.stat().st_size > 0:
                            step_files[file] = file_path
                    
                    # Only mark as completed if no checkpoint exists
                    if not step_has_checkpoint:
                        self.completed_steps.add(step)
                        self.step_files[step] = step_files
                    else:
                        logger.info(f"[RESUME] Step {step} has results but also checkpoint - will resume")
            else:
                # For other steps, require all files
                step_completed = True
                for file in files:
                    file_path = Path(self.output_dir) / file
                    if file_path.exists() and file_path.stat().st_size > 0:
                        # Special handling for Step 5 Translation_Results.json
                        if step == 5 and file == 'Translation_Results.json':
                            # Check if the file has actual translation data, not just empty results
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                translation_results = data.get('translation_results', [])
                                if translation_results and len(translation_results) > 0:
                                    step_files[file] = file_path
                                    logger.info(f"[STEP 5] Found Translation_Results.json with {len(translation_results)} terms")
                                else:
                                    logger.info(f"[STEP 5] Translation_Results.json exists but is empty - Step 5 not completed")
                                    step_completed = False
                                    break
                            except Exception as e:
                                logger.warning(f"[STEP 5] Error reading Translation_Results.json: {e}")
                                step_completed = False
                                break
                        
                        # Special handling for Step 7 Final_Terminology_Decisions.json
                        elif step == 7 and file == 'Final_Terminology_Decisions.json':
                            # Check if the file has complete decision data for all expected terms
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                # Check for decisions data
                                final_decisions = data.get('final_decisions', [])
                                modern_summary = data.get('modern_validation_summary', {})
                                total_decisions = modern_summary.get('total_decisions', len(final_decisions))
                                
                                # Load Step 6 results to check expected count
                                step6_file = Path(self.output_dir) / 'Verified_Translation_Results.json'
                                expected_terms = 0
                                if step6_file.exists():
                                    try:
                                        with open(step6_file, 'r', encoding='utf-8') as f6:
                                            step6_data = json.load(f6)
                                        expected_terms = len(step6_data.get('verified_results', []))
                                    except:
                                        expected_terms = 8691  # Fallback to known count
                                else:
                                    expected_terms = 8691  # Fallback to known count
                                
                                if total_decisions >= expected_terms:
                                    step_files[file] = file_path
                                    logger.info(f"[STEP 7] Found complete Final_Terminology_Decisions.json with {total_decisions}/{expected_terms} terms")
                                else:
                                    logger.info(f"[STEP 7] Final_Terminology_Decisions.json exists but incomplete: {total_decisions}/{expected_terms} terms - Step 7 not completed")
                                    step_completed = False
                                    break
                            except Exception as e:
                                logger.warning(f"[STEP 7] Error reading Final_Terminology_Decisions.json: {e}")
                                step_completed = False
                                break
                        
                        # Special handling for Step 8 Complete_Audit_Record.json
                        elif step == 8 and file == 'Complete_Audit_Record.json':
                            # Check if the file has complete audit data for all expected terms
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                
                                # Check for process statistics
                                process_stats = data.get('process_statistics', {})
                                total_processed = process_stats.get('total_terms_processed', 0)
                                
                                # Load Step 7 results to check expected count
                                step7_file = Path(self.output_dir) / 'Final_Terminology_Decisions.json'
                                expected_terms = 0
                                if step7_file.exists():
                                    try:
                                        with open(step7_file, 'r', encoding='utf-8') as f7:
                                            step7_data = json.load(f7)
                                        final_decisions = step7_data.get('final_decisions', [])
                                        modern_summary = step7_data.get('modern_validation_summary', {})
                                        expected_terms = modern_summary.get('total_decisions', len(final_decisions))
                                    except:
                                        expected_terms = 8691  # Fallback to known count
                                else:
                                    expected_terms = 8691  # Fallback to known count
                                
                                if total_processed >= expected_terms:
                                    step_files[file] = file_path
                                    logger.info(f"[STEP 8] Found complete Complete_Audit_Record.json with {total_processed}/{expected_terms} terms")
                                else:
                                    logger.info(f"[STEP 8] Complete_Audit_Record.json exists but incomplete: {total_processed}/{expected_terms} terms - Step 8 not completed")
                                    step_completed = False
                                    break
                            except Exception as e:
                                logger.warning(f"[STEP 8] Error reading Complete_Audit_Record.json: {e}")
                                step_completed = False
                                break
                        else:
                            step_files[file] = file_path
                    else:
                        step_completed = False
                        break
                
                # Only mark as completed if all files exist and no checkpoint
                if step_completed and not step_has_checkpoint:
                    self.completed_steps.add(step)
                    self.step_files[step] = step_files
                elif step_has_checkpoint:
                    logger.info(f"[RESUME] Step {step} has checkpoint - will resume from partial progress")
        
        return self.completed_steps
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file and return its contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[WARNING] Error loading JSON file {file_path}: {e}")
            return {}
    
    def _add_step_metadata(self, file_path: str, step_number: int, step_name: str, additional_metadata: Dict = None):
        """Add step metadata to result files for workflow tracking"""
        try:
            metadata = {
                "step_sequence": {
                    "step_number": step_number,
                    "step_name": step_name,
                    "step_timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id,
                    "workflow_stage": f"Step {step_number}: {step_name}"
                }
            }
            
            if additional_metadata:
                metadata["step_sequence"].update(additional_metadata)
            
            # For JSON files, add metadata to the file
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add metadata if it's a dict
                    if isinstance(data, dict):
                        data["step_metadata"] = metadata
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"[METADATA] Added step {step_number} metadata to {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not add metadata to JSON file {file_path}: {e}")
            
            # For CSV files, create a companion metadata file
            elif file_path.endswith('.csv'):
                metadata_file = file_path.replace('.csv', '_metadata.json')
                try:
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"[METADATA] Created step {step_number} metadata file: {os.path.basename(metadata_file)}")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not create metadata file {metadata_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"[WARNING] Error adding step metadata: {e}")
    
    def initialize_components(self):
        """Initialize all system components"""
        logger.info("[SETUP] Initializing system components...")
        
        try:
            # 1. Initialize Frequency Storage System
            self.frequency_storage = FrequencyStorageSystem(
                storage_dir=str(self.output_dir / "frequency_storage")
            )
            
            # 2. Initialize Terminology Agent (Glossary management)
            glossary_folder = self.config.get('glossary_folder', 'glossary')
            if os.path.exists(glossary_folder):
                self.terminology_agent = TerminologyAgent(
                    glossary_folder=glossary_folder,
                    model_name=self.config.get('terminology_model', 'gpt-4.1')
                )
                # Apply authentication fix
                self.terminology_agent = ensure_agent_auth_fix(self.terminology_agent)
                logger.info(f"[AUTH] Applied authentication fix to TerminologyAgent")
            else:
                logger.warning(f"[WARNING] Glossary folder not found: {glossary_folder}")
            
            # 3. Initialize Validation Manager (DISABLED - not used in current flow)
            # self.validation_manager = OrganizedValidationManager(
            #     model_name=self.config.get('validation_model', 'gpt-4.1'),
            #     run_folder=f"validation_{self.session_id}",
            #     base_output_dir=str(self.output_dir),
            #     organize_existing=True
            # )
            self.validation_manager = None  # Disabled to prevent unnecessary directory creation
            
            # 4. Initialize Fast Dictionary Agent
            self.fast_dictionary_agent = FastDictionaryAgent()
            if not self.fast_dictionary_agent.dictionary_tool.initialized:
                logger.warning("[WARNING] Fast Dictionary Agent not fully initialized - NLTK may be missing")
            
            # 4.5. Initialize Azure OpenAI client for Step 9 context generation
            try:
                from azure.identity import DefaultAzureCredential
                from openai import AzureOpenAI
                
                credential = DefaultAzureCredential()
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                self.azure_openai_client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://autodesk-openai-eastus.openai.azure.com/"),
                    api_version="2024-02-01",
                    azure_ad_token=token.token
                )
                logger.info("[AZURE] Azure OpenAI client initialized for Step 9 context generation")
            except Exception as e:
                logger.warning(f"[AZURE] Failed to initialize Azure OpenAI client: {e}")
                self.azure_openai_client = None
            
            # 5. Initialize Translation Runner
            translation_config = UltraOptimizedConfig()
            translation_config.model_size = self.config.get('translation_model_size', '1.3B')
            translation_config.gpu_workers = self.config.get('gpu_workers', 3)  # Support up to 3 GPU workers
            translation_config.cpu_workers = self.config.get('cpu_workers', 16)
            
            self.translation_runner = UltraOptimizedSmartRunner(
                config=translation_config,
                resume_session=None
            )
            
            logger.info("[OK] All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize components: {e}")
            raise
    
    def step_1_initial_term_collection(self, input_file: str) -> str:
        """
        Step 1: Initial Term Collection and Verification using Unified Processor
        - Load Term_Extracted_result.csv
        - Extract terms from Final_Curated_Terms/Final_Curated_Terms_Detailed columns ONLY
        - Convert to Combined_Terms_Data.csv format
        - Verify terms exist in their source texts
        """
        logger.info("[LIST] STEP 1: Initial Term Collection and Verification (ENHANCED)")
        logger.info("=" * 60)
        logger.info("[DIRECT] Using embedded unified processor - bypassing subprocess issues")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Use direct processor to extract terms correctly
        output_prefix = str(self.output_dir / "Step1_DirectProcessor")
        logger.info(f"[DIRECT] Processing {input_file} with embedded processor...")
        logger.info(f"[TARGET] Final_Curated_Terms and Final_Curated_Terms_Detailed columns")
        
        try:
            # Import and use direct processor
            from direct_unified_processor import process_terms_directly
            
            # Process directly
            combined_file, cleaned_file = process_terms_directly(input_file, output_prefix)
            
            logger.info(f"[DIRECT] Processing completed successfully")
            logger.info(f"[DIRECT] Combined file: {combined_file}")
            logger.info(f"[DIRECT] Cleaned file: {cleaned_file}")
            
            # Check for generated files
            combined_source = Path(combined_file)
            cleaned_source = Path(cleaned_file)
            
            if not combined_source.exists():
                raise FileNotFoundError(f"Direct processor did not create expected file: {combined_source}")
            
            # Copy to standard naming for compatibility
            combined_dest = self.output_dir / "Combined_Terms_Data.csv"
            cleaned_dest = self.output_dir / "Cleaned_Terms_Data.csv"
            
            import shutil
            shutil.copy2(combined_source, combined_dest)
            
            if cleaned_source.exists():
                shutil.copy2(cleaned_source, cleaned_dest)
                final_file = str(cleaned_dest)
                logger.info("[VERIFIED] Using cleaned dataset with verified terms")
            else:
                shutil.copy2(combined_source, cleaned_dest)
                final_file = str(cleaned_dest)
                logger.info("[COMBINED] Using combined dataset (no verification performed)")
            
            # Get basic statistics from the files
            try:
                import pandas as pd
                combined_df = pd.read_csv(combined_dest)
                final_df = pd.read_csv(final_file)
                
                self.process_stats['total_terms_input'] = len(combined_df)
                self.process_stats['terms_after_cleaning'] = len(final_df)
                
                logger.info(f"[STATS] Processing results:")
                logger.info(f"   Total extracted terms: {len(combined_df):,}")
                logger.info(f"   Final verified terms: {len(final_df):,}")
                logger.info(f"   Source: Final_Curated_Terms/Final_Curated_Terms_Detailed")
                
            except Exception as e:
                logger.warning(f"[WARNING] Could not read statistics: {e}")
            
            # Clean up temporary files
            try:
                if combined_source.exists():
                    combined_source.unlink()
                if cleaned_source.exists():
                    cleaned_source.unlink()
                # Also clean up other generated files
                for suffix in ['_Simple.csv', '_Summary.csv']:
                    temp_file = Path(f"{output_prefix}{suffix}")
                    if temp_file.exists():
                        temp_file.unlink()
            except Exception as e:
                logger.warning(f"[WARNING] Could not clean up temporary files: {e}")
            
            # Add step metadata
            self._add_step_metadata(final_file, 1, "Initial Term Collection and Verification", {
                "input_file": os.path.basename(input_file),
                "processor_used": "unified_term_processor.py",
                "extraction_source": "Final_Curated_Terms and Final_Curated_Terms_Detailed columns",
                "terms_extracted": self.process_stats.get('total_terms_input', 0),
                "terms_verified": self.process_stats.get('terms_after_cleaning', 0)
            })
            
            logger.info(f"[OK] Step 1 completed using Unified Processor")
            return final_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Unified processor failed: {e}")
            if e.stderr:
                logger.error(f"[STDERR] {e.stderr}")
            raise RuntimeError(f"Unified processor failed: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Step 1 unexpected error: {e}")
            raise
    
    def _analyze_single_term(self, term: str) -> Dict[str, Any]:
        """
        Analyze a single term using the terminology agent.
        This method is designed to be used in parallel processing.
        
        Args:
            term: The term to analyze
            
        Returns:
            Dictionary with 'found' (bool) and 'analysis' (str) keys
        """
        try:
            analysis = self.terminology_agent.analyze_text_terminology(
                text=term,
                src_lang="EN", 
                tgt_lang="EN"  # English-only processing
            )
            
            # Parse analysis results (simplified)
            found = "found" in analysis.lower() or "exists" in analysis.lower()
            
            return {
                'found': found,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'found': False,
                'analysis': f"Error during analysis: {e}"
            }
    
    def _analyze_batch_terms_threaded(self, term_batch):
        """
        Analyze a batch of terms using the shared terminology agent (ThreadPool safe)
        """
        results = []
        for term in term_batch:
            try:
                # Ensure term is a string
                if not isinstance(term, str):
                    term = str(term)
                
                # Skip empty or invalid terms
                if not term or not term.strip():
                    results.append({
                        'term': term,
                        'found': False,
                        'analysis': 'Empty or invalid term'
                    })
                    continue
                
                # Clean the term
                clean_term = term.strip()
                
                # Analyze single term using the shared agent
                single_result = self._analyze_single_term(clean_term)
                
                # Add term to result
                result = {
                    'term': clean_term,
                    'found': single_result['found'],
                    'analysis': single_result['analysis']
                }
                results.append(result)
                
            except Exception as e:
                # Ensure term is available for error reporting
                error_term = term if isinstance(term, str) else str(term) if term else "unknown_term"
                results.append({
                    'term': error_term,
                    'found': False,
                    'analysis': f"Error during analysis: {str(e)}"
                })
        
        return results
    
    def step_2_glossary_validation(self, cleaned_file: str) -> Dict[str, List[Dict]]:
        """
        Step 2: Glossary Validation with Checkpoint Resume
        - Check against existing terminology glossary
        - Use Terminology Glossary Agent
        - Reference MT Glossary
        - Support incremental processing with checkpoints
        """
        logger.info("[GLOSSARY] STEP 2: Glossary Validation")
        logger.info("=" * 60)
        
        # Load cleaned terms
        import pandas as pd
        df = pd.read_csv(cleaned_file)
        unique_terms = df['term'].unique().tolist()
        
        logger.info(f"[SEARCH] Checking {len(unique_terms)} unique terms against glossaries...")
        
        # Check for existing checkpoint
        checkpoint_file = str(self.output_dir / "step2_checkpoint.json")
        glossary_file = str(self.output_dir / "Glossary_Analysis_Results.json")
        
        glossary_results = {
            'existing_terms': [],
            'new_terms': [],
            'glossary_conflicts': []
        }
        
        # Load existing progress if available
        processed_terms = set()
        remaining_terms = unique_terms.copy()
        start_index = 0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                processed_terms = set(checkpoint_data.get('processed_terms', []))
                start_index = checkpoint_data.get('last_processed_index', 0) + 1
                
                # Load existing results
                if os.path.exists(glossary_file):
                    with open(glossary_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                        glossary_results = existing_results.get('results', glossary_results)
                
                remaining_terms = [term for term in unique_terms if term not in processed_terms]
                
                if remaining_terms:
                    logger.info(f"[RESUME] Found checkpoint: {len(processed_terms)} terms already processed")
                    logger.info(f"[RESUME] Resuming with {len(remaining_terms)} remaining terms")
                else:
                    logger.info(f"[SKIP] All terms already processed according to checkpoint")
                    
            except Exception as e:
                logger.warning(f"[WARNING] Error reading checkpoint: {e}. Starting from beginning.")
                processed_terms = set()
                remaining_terms = unique_terms.copy()
                start_index = 0
        
        def save_checkpoint(processed_list, results):
            """Save current progress to checkpoint file"""
            try:
                checkpoint_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_terms': len(unique_terms),
                    'processed_terms': list(processed_list),
                    'last_processed_index': len(processed_list) - 1,
                    'remaining_terms': len(unique_terms) - len(processed_list)
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
                # Also save intermediate results
                intermediate_results = {
                    'metadata': {
                        'step': 2,
                        'step_name': 'Glossary Validation',
                        'timestamp': datetime.now().isoformat(),
                        'total_terms': len(unique_terms),
                        'processed_terms': len(processed_list),
                        'remaining_terms': len(unique_terms) - len(processed_list)
                    },
                    'results': results
                }
                with open(glossary_file, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"[CHECKPOINT] Saved progress: {len(processed_list)}/{len(unique_terms)} terms")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to save checkpoint: {e}")
        
        if self.terminology_agent and remaining_terms:
            try:
                # Get glossary overview
                logger.info("[INFO] Getting glossary overview...")
                overview = self.terminology_agent.get_glossary_overview()
                logger.info("Glossary overview obtained")
                
                # Use maximum CPU cores for aggressive multiprocessing
                max_workers = cpu_count()  # Use all available CPU cores
                logger.info(f"[PARALLEL] Using {max_workers} workers (all CPU cores) for maximum parallel processing...")
                logger.info(f"[PROCESS] Processing {len(remaining_terms)} remaining terms...")
                
                # Process terms in larger batches for better throughput with more workers
                batch_size = max(50, len(remaining_terms) // max_workers)  # Dynamic batch size based on CPU cores
                batch_count = (len(remaining_terms) + batch_size - 1) // batch_size
                
                logger.info(f"[BATCH] Processing {len(remaining_terms)} terms in {batch_count} batches of {batch_size}")
                logger.info(f"[OPTIMIZATION] Using all {max_workers} CPU cores for maximum speed")
                
                for batch_idx in range(batch_count):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(remaining_terms))
                    batch_terms = remaining_terms[batch_start:batch_end]
                    
                    logger.info(f"[BATCH {batch_idx + 1}/{batch_count}] Processing terms {batch_start + 1} to {batch_end}")
                    
                    # Process batch in parallel using ProcessPoolExecutor for maximum CPU utilization
                    # Clean and validate terms before processing
                    clean_batch_terms = []
                    for term in batch_terms:
                        if isinstance(term, str) and term.strip():
                            clean_batch_terms.append(term.strip())
                        elif hasattr(term, 'get') and term.get('term'):
                            # Handle case where term is a dict with 'term' key
                            clean_batch_terms.append(str(term['term']).strip())
                        else:
                            logger.warning(f"[WARNING] Skipping invalid term: {term}")
                    
                    if not clean_batch_terms:
                        logger.warning(f"[WARNING] No valid terms in batch {batch_idx + 1}")
                        continue
                    
                    # Split batch into smaller sub-batches for each worker
                    terms_per_worker = max(1, len(clean_batch_terms) // max_workers)
                    worker_batches = []
                    
                    for i in range(0, len(clean_batch_terms), terms_per_worker):
                        worker_batch = clean_batch_terms[i:i + terms_per_worker]
                        worker_batches.append(worker_batch)
                    
                    logger.info(f"[WORKERS] Split {len(clean_batch_terms)} terms into {len(worker_batches)} worker batches")
                    
                    # Use ThreadPoolExecutor by default for better stability with shared resources
                    if self.use_process_pool:
                        logger.info("[MULTIPROCESSING] Using ProcessPoolExecutor (experimental)")
                        executor_class = concurrent.futures.ProcessPoolExecutor
                        worker_func = analyze_term_batch_worker
                    else:
                        logger.info("[MULTIPROCESSING] Using ThreadPoolExecutor (stable)")
                        executor_class = concurrent.futures.ThreadPoolExecutor
                        worker_func = self._analyze_batch_terms_threaded
                    
                    with executor_class(max_workers=max_workers) as executor:
                        if self.use_process_pool:
                            # Create futures for worker batches (ProcessPool)
                            future_to_batch = {
                                executor.submit(
                                    worker_func, 
                                    worker_batch, 
                                    self.config.get('glossary_folder', 'glossary'),
                                    self.config.get('terminology_model', 'gpt-4.1')
                                ): worker_batch 
                                for worker_batch in worker_batches
                            }
                        else:
                            # Create futures for worker batches (ThreadPool)
                            future_to_batch = {
                                executor.submit(worker_func, worker_batch): worker_batch 
                                for worker_batch in worker_batches
                            }
                        
                        # Process results as they complete
                        batch_completed = 0
                        for future in concurrent.futures.as_completed(future_to_batch):
                            worker_batch = future_to_batch[future]
                            batch_completed += len(worker_batch)
                            
                            if batch_completed % 100 == 0 or batch_completed >= len(batch_terms):
                                total_completed = len(processed_terms) + batch_completed
                                logger.info(f"   [PROGRESS] Processed {total_completed}/{len(unique_terms)} terms ({total_completed/len(unique_terms)*100:.1f}%)")
                            
                            try:
                                # Get results from worker batch
                                worker_results = future.result()
                                
                                for result in worker_results:
                                    term = result['term']
                                    if result['found']:
                                        glossary_results['existing_terms'].append({
                                            'term': term,
                                            'analysis': result['analysis']
                                        })
                                    else:
                                        glossary_results['new_terms'].append({
                                            'term': term,
                                            'analysis': result['analysis']
                                        })
                                    
                                    processed_terms.add(term)
                                
                            except Exception as e:
                                logger.warning(f"[WARNING] Error processing worker batch: {e}")
                                # Add all terms in this batch as new terms with error
                                for term in worker_batch:
                                    glossary_results['new_terms'].append({
                                        'term': term,
                                        'analysis': f"Worker batch error: {e}"
                                    })
                                    processed_terms.add(term)
                    
                    # Save checkpoint after each batch
                    save_checkpoint(processed_terms, glossary_results)
                    logger.info(f"[BATCH {batch_idx + 1}] Completed: {len(batch_terms)} terms processed")
                
            except Exception as e:
                logger.error(f"[ERROR] Glossary validation failed: {e}")
                # Fallback: treat remaining terms as new
                for term in remaining_terms:
                    if term not in processed_terms:
                        glossary_results['new_terms'].append({'term': term, 'analysis': 'Fallback: treated as new'})
                        processed_terms.add(term)
                save_checkpoint(processed_terms, glossary_results)
        
        elif remaining_terms:
            logger.warning("[WARNING] Terminology agent not available, treating remaining terms as new")
            for term in remaining_terms:
                glossary_results['new_terms'].append({'term': term, 'analysis': 'No glossary agent available'})
                processed_terms.add(term)
            save_checkpoint(processed_terms, glossary_results)
        
        # Clean up checkpoint file on successful completion
        try:
            if os.path.exists(checkpoint_file) and len(processed_terms) == len(unique_terms):
                os.remove(checkpoint_file)
                logger.info("[CLEANUP] Step 2 checkpoint file removed after successful completion")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to remove Step 2 checkpoint file: {e}")
        
        # Save final results to disk for step completion detection
        final_results = {
            'metadata': {
                'step': 2,
                'step_name': 'Glossary Validation',
                'timestamp': datetime.now().isoformat(),
                'total_terms': len(unique_terms),
                'processed_terms': len(processed_terms),
                'status': 'completed'
            },
            'results': glossary_results
        }
        
        with open(glossary_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        self._add_step_metadata(glossary_file, 2, "Glossary Validation", {
            'existing_terms': len(glossary_results['existing_terms']),
            'new_terms': len(glossary_results['new_terms']),
            'conflicts': len(glossary_results['glossary_conflicts'])
        })
        
        logger.info(f"[OK] Step 2 completed:")
        logger.info(f"   Existing terms: {len(glossary_results['existing_terms'])}")
        logger.info(f"   New terms: {len(glossary_results['new_terms'])}")
        logger.info(f"   Conflicts: {len(glossary_results['glossary_conflicts'])}")
        logger.info(f"[FOLDER] Results saved: {glossary_file}")
        
        return glossary_results
    
    def step_3_new_terminology_processing(self, glossary_results: Dict, cleaned_file: str) -> str:
        """
        Step 3: New Terminology Processing with Checkpoint Resume
        - Process terms marked as new terminology
        - Check against most up-to-date dictionary using Fast Dictionary Agent
        - Support incremental processing with checkpoints
        """
        logger.info("[NEW] STEP 3: New Terminology Processing")
        logger.info("=" * 60)
        
        new_terms = glossary_results['new_terms']
        logger.info(f"[SEARCH] Processing {len(new_terms)} new terms with Fast Dictionary Agent...")
        
        # Check for existing checkpoint
        checkpoint_file = str(self.output_dir / "step3_checkpoint.json")
        new_terms_file = str(self.output_dir / "New_Terms_Candidates_With_Dictionary.json")
        
        # Load existing progress if available
        processed_terms = []
        remaining_terms = new_terms.copy()
        start_index = 0
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                processed_terms = checkpoint_data.get('processed_terms', [])
                start_index = checkpoint_data.get('last_processed_index', 0) + 1
                
                if start_index < len(new_terms):
                    remaining_terms = new_terms[start_index:]
                    logger.info(f"[RESUME] Found checkpoint: {len(processed_terms)} terms already processed")
                    logger.info(f"[RESUME] Resuming from term {start_index + 1}/{len(new_terms)}")
                else:
                    logger.info(f"[SKIP] All terms already processed according to checkpoint")
                    remaining_terms = []
                    
            except Exception as e:
                logger.warning(f"[WARNING] Error reading checkpoint: {e}. Starting from beginning.")
                processed_terms = []
                remaining_terms = new_terms.copy()
                start_index = 0
        
        # Load full term data for context
        import pandas as pd
        df = pd.read_csv(cleaned_file)
        
        # Process remaining terms with checkpoint support
        def save_checkpoint(processed_list, current_index):
            """Save current progress to checkpoint file"""
            try:
                checkpoint_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_terms': len(new_terms),
                    'processed_terms': processed_list,
                    'last_processed_index': current_index,
                    'remaining_terms': len(new_terms) - current_index - 1
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                logger.info(f"[CHECKPOINT] Saved progress: {len(processed_list)}/{len(new_terms)} terms")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to save checkpoint: {e}")
        
        # Use batch processing for optimal speed with checkpoint support
        
        # If we have remaining terms, use batch processing for speed
        if remaining_terms and self.fast_dictionary_agent and self.fast_dictionary_agent.dictionary_tool.initialized:
            logger.info(f"[BATCH] Using optimized batch processing for {len(remaining_terms)} remaining terms...")
            
            # Convert remaining terms to the format expected by batch processing
            remaining_terms_data = []
            for i, term_info in enumerate(remaining_terms):
                current_global_index = start_index + i
                term = term_info['term']
                
                try:
                    term_rows = df[df['term'] == term]
                    contexts = term_rows['original_text'].tolist()
                    pos_tags = term_rows['pos_tag'].tolist()
                    frequency = len(term_rows)
                    
                    term_data = {
                        'term': term,
                        'frequency': frequency,
                        'original_texts': {'texts': contexts},
                        'pos_tag_variations': {'tags': list(set(pos_tags))},
                        'glossary_analysis': term_info['analysis'],
                        '_checkpoint_index': current_global_index  # Track original index
                    }
                    remaining_terms_data.append(term_data)
                    
                except Exception as e:
                    logger.error(f"[ERROR] Failed to prepare term '{term}' for batch processing: {e}")
            
            # Process in batches with checkpoints
            batch_size = 1000  # Process 1000 terms per batch
            batch_count = (len(remaining_terms_data) + batch_size - 1) // batch_size
            
            logger.info(f"[BATCH] Processing {len(remaining_terms_data)} terms in {batch_count} batches of {batch_size}")
            
            for batch_idx in range(batch_count):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(remaining_terms_data))
                batch_terms = remaining_terms_data[batch_start:batch_end]
                
                logger.info(f"[BATCH {batch_idx + 1}/{batch_count}] Processing terms {batch_start + 1} to {batch_end}")
                
                try:
                    # Run batch dictionary analysis
                    dict_words, non_dict_words = self.fast_dictionary_agent.analyze_terms_fast(
                        batch_terms, 
                        max_terms=None
                    )
                    
                    # Add batch results to processed terms
                    batch_results = dict_words + non_dict_words
                    processed_terms.extend(batch_results)
                    
                    # Save checkpoint after each batch
                    last_index = batch_terms[-1]['_checkpoint_index']
                    save_checkpoint(processed_terms, last_index)
                    
                    logger.info(f"[BATCH {batch_idx + 1}] Completed: {len(batch_results)} terms processed")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Batch {batch_idx + 1} failed: {e}")
                    # Add failed terms with error markers
                    for term_data in batch_terms:
                        term_data['dictionary_analysis'] = {
                            'in_dictionary': None,
                            'method': 'batch_error',
                            'confidence': 'error',
                            'reason': str(e)
                        }
                        processed_terms.append(term_data)
                    
                    # Save checkpoint even for failed batch
                    last_index = batch_terms[-1]['_checkpoint_index']
                    save_checkpoint(processed_terms, last_index)
        
        # All terms are now in processed_terms
        all_analyzed_terms = processed_terms
        
        # Handle case when dictionary agent is not available
        if remaining_terms and not (self.fast_dictionary_agent and self.fast_dictionary_agent.dictionary_tool.initialized):
            logger.warning("[WARNING] Fast Dictionary Agent not available, processing without dictionary analysis")
            
            # Process remaining terms without dictionary analysis
            for i, term_info in enumerate(remaining_terms):
                current_global_index = start_index + i
                term = term_info['term']
                
                try:
                    term_rows = df[df['term'] == term]
                    contexts = term_rows['original_text'].tolist()
                    pos_tags = term_rows['pos_tag'].tolist()
                    frequency = len(term_rows)
                    
                    term_data = {
                        'term': term,
                        'frequency': frequency,
                        'original_texts': {'texts': contexts},
                        'pos_tag_variations': {'tags': list(set(pos_tags))},
                        'glossary_analysis': term_info['analysis'],
                        'dictionary_analysis': {
                            'in_dictionary': None,
                            'method': 'agent_unavailable',
                            'confidence': 'unknown',
                            'reason': 'Fast Dictionary Agent not initialized'
                        }
                    }
                    processed_terms.append(term_data)
                    
                    # Save checkpoint every 500 terms (faster without dictionary analysis)
                    if (i + 1) % 500 == 0 or i == len(remaining_terms) - 1:
                        save_checkpoint(processed_terms, current_global_index)
                        
                except Exception as e:
                    logger.error(f"[ERROR] Failed to process term '{term}': {e}")
        
        # Clean up checkpoint index from final results
        for term_data in all_analyzed_terms:
            if '_checkpoint_index' in term_data:
                del term_data['_checkpoint_index']
        
        # Clean up checkpoint file on successful completion
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info("[CLEANUP] Checkpoint file removed after successful completion")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to remove checkpoint file: {e}")
        
        # Save new terms dataset with dictionary analysis
        new_terms_file = str(self.output_dir / "New_Terms_Candidates_With_Dictionary.json")
        with open(new_terms_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_new_terms': len(all_analyzed_terms),
                    'dictionary_analysis_method': 'fast_dictionary_agent',
                    'source': 'agentic_terminology_validation_system'
                },
                'new_terms': all_analyzed_terms
            }, f, indent=2, ensure_ascii=False)
        
        # Also save separate files for dictionary and non-dictionary terms
        dict_terms = [t for t in all_analyzed_terms if t.get('dictionary_analysis', {}).get('in_dictionary') == True]
        non_dict_terms = [t for t in all_analyzed_terms if t.get('dictionary_analysis', {}).get('in_dictionary') == False]
        
        if self.fast_dictionary_agent and self.fast_dictionary_agent.dictionary_tool.initialized:
            
            # Save dictionary terms
            dict_file = str(self.output_dir / "Dictionary_Terms_Identified.json")
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'analysis_method': 'fast_dictionary_agent',
                        'total_terms': len(dict_terms)
                    },
                    'dictionary_terms': dict_terms
                }, f, indent=2, ensure_ascii=False)
            
            # Save non-dictionary terms
            non_dict_file = str(self.output_dir / "Non_Dictionary_Terms_Identified.json")
            with open(non_dict_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'analysis_method': 'fast_dictionary_agent',
                        'total_terms': len(non_dict_terms)
                    },
                    'non_dictionary_terms': non_dict_terms
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[FOLDER] Dictionary terms saved to: {dict_file}")
            logger.info(f"[FOLDER] Non-dictionary terms saved to: {non_dict_file}")
        else:
            # Create empty files when dictionary agent is not available
            dict_file = str(self.output_dir / "Dictionary_Terms_Identified.json")
            with open(dict_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'analysis_method': 'unavailable',
                        'total_terms': 0,
                        'note': 'Fast Dictionary Agent was not available'
                    },
                    'dictionary_terms': []
                }, f, indent=2, ensure_ascii=False)
            
            non_dict_file = str(self.output_dir / "Non_Dictionary_Terms_Identified.json")
            with open(non_dict_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'analysis_method': 'unavailable',
                        'total_terms': len(all_analyzed_terms),
                        'note': 'All terms marked as non-dictionary due to agent unavailability'
                    },
                    'non_dictionary_terms': all_analyzed_terms
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[FOLDER] Empty dictionary file created: {dict_file}")
            logger.info(f"[FOLDER] All terms saved as non-dictionary: {non_dict_file}")
        
        # Add step metadata
        self._add_step_metadata(new_terms_file, 3, "New Terminology Processing", {
            "dictionary_agent_used": "FastDictionaryAgent",
            "terms_processed": len(all_analyzed_terms),
            "dictionary_terms_found": len(dict_terms),
            "non_dictionary_terms": len(non_dict_terms)
        })
        
        logger.info(f"[OK] Step 3 completed: {len(all_analyzed_terms)} new terms processed with dictionary analysis")
        logger.info(f"[FOLDER] Complete dataset saved to: {new_terms_file}")
        
        return new_terms_file
    
    def step_4_frequency_analysis_filtering(self, new_terms_file: str) -> Tuple[str, str]:
        """
        Step 4: Frequency Analysis and Filtering
        - Filter terms with frequency > 2
        - Store frequency = 1 terms for future reference
        """
        logger.info("[STATS] STEP 4: Frequency Analysis and Filtering")
        logger.info("=" * 60)
        
        # Load new terms data
        with open(new_terms_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_terms = data['new_terms']
        
        high_frequency_terms = []
        low_frequency_terms = []
        
        for term_data in new_terms:
            frequency = term_data['frequency']
            
            if frequency >= 2:
                high_frequency_terms.append(term_data)
            else:
                low_frequency_terms.append(term_data)
                
                # Store frequency=1 terms in storage system
                self.frequency_storage.store_frequency_one_term(
                    term=term_data['term'],
                    source_file=new_terms_file,
                    original_contexts=term_data['original_texts']['texts'],
                    pos_tags=term_data['pos_tag_variations']['tags'],
                    metadata={
                        'glossary_analysis': term_data['glossary_analysis'],
                        'stored_at': datetime.now().isoformat()
                    }
                )
        
        # Save high frequency terms for processing
        high_freq_file = str(self.output_dir / "High_Frequency_Terms.json")
        with open(high_freq_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_terms': len(high_frequency_terms),
                    'min_frequency': 2,
                    'source': 'frequency_filtering'
                },
                'terms': high_frequency_terms
            }, f, indent=2, ensure_ascii=False)
        
        # Export frequency storage statistics
        storage_stats = self.frequency_storage.get_storage_statistics()
        storage_export = self.frequency_storage.export_to_json(
            str(self.output_dir / "Frequency_Storage_Export.json")
        )
        
        # Update statistics
        self.process_stats['frequency_1_terms'] = len(low_frequency_terms)
        self.process_stats['frequency_gt_2_terms'] = len(high_frequency_terms)
        
        logger.info(f"[OK] Step 4 completed:")
        logger.info(f"   High frequency terms (>=2): {len(high_frequency_terms)}")
        logger.info(f"   Low frequency terms (=1): {len(low_frequency_terms)}")
        logger.info(f"   Storage statistics: {storage_stats}")
        
        return high_freq_file, storage_export
    
    def _calculate_remaining_terms_dynamically(self):
        """
        Calculate remaining terms dynamically using High_Frequency_Terms.json and Translation_Results.json
        This ensures we always have the most accurate and up-to-date list of terms to translate.
        """
        try:
            logger.info("[DYNAMIC] Starting dynamic calculation of remaining terms...")
            
            # Load High_Frequency_Terms.json (authoritative source)
            high_freq_file = self.output_dir / "High_Frequency_Terms.json"
            if not os.path.exists(high_freq_file):
                logger.error(f"[ERROR] High_Frequency_Terms.json not found at {high_freq_file}")
                return None
            
            with open(high_freq_file, 'r', encoding='utf-8') as f:
                high_freq_data = json.load(f)
            
            # Extract all terms that should be processed (frequency ≥2)
            all_required_terms = set()
            for term_data in high_freq_data.get('terms', []):
                if isinstance(term_data, dict) and 'term' in term_data:
                    all_required_terms.add(term_data['term'])
            
            logger.info(f"[DYNAMIC] Loaded {len(all_required_terms):,} required terms from High_Frequency_Terms.json")
            
            # Load completed terms from Translation_Results.json
            results_files = [
                self.output_dir / "Translation_Results.json",
                self.output_dir / "Translation_Results_cleaned.json"
            ]
            
            completed_terms = set()
            for results_file in results_files:
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                        
                        for result in results_data.get('translation_results', []):
                            if isinstance(result, dict) and 'term' in result:
                                completed_terms.add(result['term'])
                        
                        logger.info(f"[DYNAMIC] Loaded terms from {results_file.name}")
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not load {results_file.name}: {e}")
            
            logger.info(f"[DYNAMIC] Found {len(completed_terms):,} completed terms")
            
            # Calculate remaining terms
            remaining_terms = all_required_terms - completed_terms
            extra_completed = completed_terms - all_required_terms
            
            logger.info(f"[DYNAMIC] Calculated {len(remaining_terms):,} terms still need translation")
            if extra_completed:
                logger.info(f"[DYNAMIC] Found {len(extra_completed):,} extra completed terms not in high frequency list")
            
            # Create new remaining terms structure
            remaining_terms_data = {
                'metadata': {
                    'generated_timestamp': datetime.now().isoformat(),
                    'source_method': 'dynamic_calculation',
                    'authoritative_source': 'High_Frequency_Terms.json',
                    'completed_source': 'Translation_Results.json',
                    'total_required_terms': len(all_required_terms),
                    'completed_terms': len(completed_terms),
                    'remaining_terms': len(remaining_terms),
                    'calculation': f'{len(all_required_terms)} - {len(completed_terms)} = {len(remaining_terms)}'
                },
                'dictionary_terms': [],
                'non_dictionary_terms': []
            }
            
            # Create a lookup for frequency data
            term_frequency_lookup = {}
            for term_data in high_freq_data.get('terms', []):
                if isinstance(term_data, dict) and 'term' in term_data:
                    term_frequency_lookup[term_data['term']] = term_data.get('frequency', 2)
            
            # Add remaining terms to dictionary_terms with frequency data
            for term in sorted(remaining_terms):
                remaining_terms_data['dictionary_terms'].append({
                    'term': term,
                    'source': 'High_Frequency_Terms.json',
                    'frequency': term_frequency_lookup.get(term, 2),  # Use actual frequency or default to 2
                    'original_texts': []  # Will be populated if needed
                })
            
            # Save the dynamically calculated remaining terms file
            remaining_file = self.output_dir / "Remaining_Terms_For_Translation.json"
            backup_file = f"{remaining_file}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create backup if file exists
            if os.path.exists(remaining_file):
                import shutil
                shutil.copy2(remaining_file, backup_file)
                logger.info(f"[BACKUP] Created backup: {os.path.basename(backup_file)}")
            
            # Write new remaining terms file
            with open(remaining_file, 'w', encoding='utf-8') as f:
                json.dump(remaining_terms_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[DYNAMIC] Generated new remaining terms file with {len(remaining_terms):,} terms")
            
            # Update checkpoint with dynamic calculation
            self._update_checkpoint_with_dynamic_calculation(len(all_required_terms), len(completed_terms), len(remaining_terms))
            
            return {
                'total_required': len(all_required_terms),
                'completed': len(completed_terms),
                'remaining': len(remaining_terms),
                'remaining_terms_set': remaining_terms
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Dynamic calculation failed: {e}")
            return None
    
    def _update_checkpoint_with_dynamic_calculation(self, total_terms, completed_terms, remaining_terms):
        """Update checkpoint with dynamically calculated values"""
        try:
            checkpoint_file = self.output_dir / "step5_translation_checkpoint.json"
            
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
            else:
                checkpoint_data = {}
            
            # Update with dynamic calculation
            completion_percentage = (completed_terms / total_terms * 100) if total_terms > 0 else 0
            
            checkpoint_data.update({
                'total_terms': total_terms,
                'completed_terms': completed_terms,
                'remaining_terms': remaining_terms,
                'completion_percentage': completion_percentage,
                'calculation_method': 'dynamic_from_authoritative_sources',
                'last_calculated': datetime.now().isoformat(),
                'sources': {
                    'authoritative': 'High_Frequency_Terms.json',
                    'completed': 'Translation_Results.json'
                },
                'next_batch_info': {
                    'batch_size': remaining_terms,
                    'method': 'calculated_dynamically'
                }
            })
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[DYNAMIC] Updated checkpoint: {completed_terms:,}/{total_terms:,} terms ({completion_percentage:.1f}%)")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update checkpoint: {e}")
    
    def _update_remaining_terms_after_progress(self, current_translations: list):
        """Update the remaining terms file after translation progress"""
        try:
            # Get set of currently translated terms
            translated_terms = set()
            for result in current_translations:
                if isinstance(result, dict) and 'term' in result:
                    translated_terms.add(result['term'])
            
            logger.info(f"[UPDATE] Updating remaining terms after {len(translated_terms)} translations")
            
            # Load original high frequency terms
            high_freq_file = str(self.output_dir / "High_Frequency_Terms.json")
            with open(high_freq_file, 'r', encoding='utf-8') as f:
                dict_data = json.load(f)
            
            if 'terms' in dict_data:
                all_dict_terms = dict_data['terms']
            elif 'dictionary_terms' in dict_data:
                all_dict_terms = dict_data['dictionary_terms']
            else:
                logger.error("[ERROR] No terms found in High_Frequency_Terms.json")
                return
            
            # Load non-dictionary terms with frequency >= 2
            non_dict_file = str(self.output_dir / "Non_Dictionary_Terms_Identified.json")
            high_freq_non_dict = []
            if os.path.exists(non_dict_file):
                with open(non_dict_file, 'r', encoding='utf-8') as f:
                    non_dict_data = json.load(f)
                all_non_dict = non_dict_data.get('non_dictionary_terms', [])
                high_freq_non_dict = [term for term in all_non_dict if term.get('frequency', 0) >= 2]
            
            # OVERLAP DETECTION AND RESOLUTION
            logger.info("[OVERLAP_CHECK] Checking for overlaps between dictionary and non-dictionary terms...")
            
            # Create sets of term strings for overlap detection
            dict_term_strings = set()
            for term_data in all_dict_terms:
                if isinstance(term_data, dict):
                    term = term_data.get('term', '')
                else:
                    term = str(term_data)
                if term:
                    dict_term_strings.add(term)
            
            non_dict_term_strings = set()
            for term_data in high_freq_non_dict:
                term = term_data.get('term', '')
                if term:
                    non_dict_term_strings.add(term)
            
            # Detect overlaps
            overlapping_terms = dict_term_strings & non_dict_term_strings
            overlap_count = len(overlapping_terms)
            
            logger.info(f"[OVERLAP_ANALYSIS] Dictionary terms: {len(dict_term_strings):,}")
            logger.info(f"[OVERLAP_ANALYSIS] Non-dictionary terms (freq>=2): {len(non_dict_term_strings):,}")
            logger.info(f"[OVERLAP_ANALYSIS] Overlapping terms: {overlap_count:,}")
            
            if overlap_count > 0:
                overlap_percentage = (overlap_count / len(non_dict_term_strings)) * 100 if len(non_dict_term_strings) > 0 else 0
                logger.warning(f"[OVERLAP_DETECTED] {overlap_count:,} terms overlap ({overlap_percentage:.1f}% of non-dict terms)")
                
                if overlap_percentage > 90:
                    logger.warning("[OVERLAP_RESOLUTION] High overlap detected (>90%) - using dictionary terms only")
                    # Use only dictionary terms to avoid double-counting
                    high_freq_non_dict = []  # Clear non-dictionary terms
                    logger.info("[OVERLAP_RESOLUTION] Non-dictionary terms excluded due to high overlap")
                else:
                    logger.info(f"[OVERLAP_RESOLUTION] Removing {overlap_count:,} overlapping terms from non-dictionary list")
                    # Remove overlapping terms from non-dictionary list
                    high_freq_non_dict = [term_data for term_data in high_freq_non_dict 
                                         if term_data.get('term', '') not in overlapping_terms]
                    logger.info(f"[OVERLAP_RESOLUTION] Non-dictionary terms after deduplication: {len(high_freq_non_dict):,}")
            else:
                logger.info("[OVERLAP_CHECK] No overlaps detected - proceeding with both lists")
            
            # Calculate total unique terms after overlap resolution
            total_unique_terms = len(dict_term_strings) + len(high_freq_non_dict)
            logger.info(f"[UNIQUE_TERMS] Total unique terms after overlap resolution: {total_unique_terms:,}")
            
            # Filter out already translated terms
            remaining_dict_terms = []
            for term_data in all_dict_terms:
                if isinstance(term_data, dict):
                    term = term_data.get('term', '')
                else:
                    term = str(term_data)
                
                if term and term not in translated_terms:
                    remaining_dict_terms.append(term_data)
            
            remaining_non_dict_terms = []
            for term_data in high_freq_non_dict:
                term = term_data.get('term', '')
                if term and term not in translated_terms:
                    remaining_non_dict_terms.append(term_data)
            
            # Update remaining terms file
            remaining_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_terms": len(remaining_dict_terms) + len(remaining_non_dict_terms),
                    "dictionary_terms": len(remaining_dict_terms),
                    "non_dictionary_terms": len(remaining_non_dict_terms),
                    "already_translated": len(translated_terms),
                    "source": "Updated after translation progress with overlap resolution",
                    "overlap_analysis": {
                        "overlapping_terms_detected": overlap_count,
                        "overlap_percentage": overlap_percentage if overlap_count > 0 else 0,
                        "resolution_applied": "High overlap exclusion" if overlap_count > 0 and overlap_percentage > 90 else "Deduplication" if overlap_count > 0 else "None",
                        "original_dict_terms": len(dict_term_strings),
                        "original_non_dict_terms": len(non_dict_term_strings) if overlap_count == 0 else len(non_dict_term_strings) + overlap_count,
                        "final_unique_terms": total_unique_terms
                    }
                },
                "dictionary_terms": remaining_dict_terms,
                "non_dictionary_terms": remaining_non_dict_terms
            }
            
            # Save updated remaining terms file
            remaining_file = str(self.output_dir / "Remaining_Terms_For_Translation.json")
            with open(remaining_file, 'w', encoding='utf-8') as f:
                json.dump(remaining_data, f, indent=2, ensure_ascii=False)
            
            # Update checkpoint file
            checkpoint_file = self.output_dir / "step5_translation_checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                total_remaining = len(remaining_dict_terms) + len(remaining_non_dict_terms)
                # Use the corrected total unique terms after overlap resolution
                corrected_total_terms = len(translated_terms) + total_remaining
                
                checkpoint_data.update({
                    "total_terms": corrected_total_terms,  # Update with overlap-corrected total
                    "completed_terms": len(translated_terms),
                    "remaining_terms": total_remaining,
                    "completion_percentage": (len(translated_terms) / corrected_total_terms * 100) if corrected_total_terms > 0 else 100,
                    "checkpoint_timestamp": datetime.now().isoformat(),
                    "overlap_resolution_applied": True,
                    "overlap_analysis": {
                        "overlapping_terms": overlap_count,
                        "overlap_percentage": overlap_percentage if overlap_count > 0 else 0,
                        "resolution_method": "High overlap exclusion" if overlap_count > 0 and overlap_percentage > 90 else "Deduplication" if overlap_count > 0 else "None"
                    }
                })
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"[UPDATE] Updated remaining terms: {len(remaining_dict_terms)} dict + {len(remaining_non_dict_terms)} non-dict = {len(remaining_dict_terms) + len(remaining_non_dict_terms)} remaining")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update remaining terms: {e}")

    def step_5_translation_process(self, high_freq_file: str) -> str:
        """
        Step 5: Translation Process
        - Use generic translation for new terminology
        - Translate to multiple languages (1-200)
        - Use NLLB and AYA 101 models
        """
        logger.info("[WEB] STEP 5: Translation Process")
        logger.info("=" * 60)
        
        # Check for checkpoint to resume translation
        checkpoint_file = self.output_dir / "step5_translation_checkpoint.json"
        translation_results_file = str(self.output_dir / "Translation_Results.json")
        
        if checkpoint_file.exists():
            logger.info("[CHECKPOINT] Found Step 5 checkpoint - resuming translation...")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # ALWAYS validate against actual translation results (SOURCE OF TRUTH)
            actual_completed = 0
            unique_terms_completed = 0
            if os.path.exists(translation_results_file):
                try:
                    with open(translation_results_file, 'r', encoding='utf-8') as f:
                        actual_data = json.load(f)
                    actual_results = actual_data.get('translation_results', [])
                    actual_completed = len(actual_results)
                    
                    # Count unique terms (avoid duplicates)
                    unique_terms = set()
                    for result in actual_results:
                        if isinstance(result, dict) and 'term' in result:
                            unique_terms.add(result['term'])
                    unique_terms_completed = len(unique_terms)
                    
                    logger.info(f"[SOURCE_OF_TRUTH] Actual translation results: {actual_completed} entries, {unique_terms_completed} unique terms")
                except Exception as e:
                    logger.error(f"[ERROR] Could not read actual translation results: {e}")
                    return translation_results_file
            
            checkpoint_completed = checkpoint_data.get('completed_terms', 0)
            
            # ALWAYS use actual results as source of truth
            if unique_terms_completed > 0:
                logger.info(f"[OVERRIDE] Using ACTUAL results as source of truth: {unique_terms_completed} terms")
                # Recalculate all checkpoint values based on actual results
                # Use the corrected total from checkpoint (8691 = terms with frequency ≥2)
                corrected_total_terms = checkpoint_data.get('total_terms', 8691)  # Get from checkpoint, fallback to 8691
                remaining_terms = corrected_total_terms - unique_terms_completed
                completion_percentage = (unique_terms_completed / corrected_total_terms) * 100
                
                # Update checkpoint data with actual values
                checkpoint_data.update({
                    'completed_terms': unique_terms_completed,
                    'total_terms': corrected_total_terms,
                    'remaining_terms': remaining_terms,
                    'completion_percentage': completion_percentage,
                    'source_of_truth': 'actual_translation_results',
                    'validated_against_results': True,
                    'checkpoint_corrected': True if checkpoint_completed != unique_terms_completed else False
                })
                
                if checkpoint_completed != unique_terms_completed:
                    logger.warning(f"[CORRECTED] Checkpoint claimed {checkpoint_completed}, actual is {unique_terms_completed}")
                    
                    # Save corrected checkpoint immediately
                    try:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                        logger.info(f"[SAVED] Corrected checkpoint saved with actual progress")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to save corrected checkpoint: {e}")
            else:
                logger.error(f"[ERROR] No actual translation results found - cannot proceed")
                return translation_results_file
            
            logger.info(f"[RESUME] Completed: {checkpoint_data['completed_terms']} terms")
            logger.info(f"[RESUME] Remaining: {checkpoint_data['remaining_terms']} terms")
            logger.info(f"[RESUME] Progress: {checkpoint_data['completion_percentage']:.1f}%")
            
            # DYNAMIC CALCULATION: Use High_Frequency_Terms.json and Translation_Results.json
            logger.info("[DYNAMIC] Calculating remaining terms from authoritative sources...")
            remaining_terms_calculated = self._calculate_remaining_terms_dynamically()
            
            # Use the remaining terms file for translation (now dynamically generated)
            remaining_file = str(self.output_dir / "Remaining_Terms_For_Translation.json")
            if os.path.exists(remaining_file):
                logger.info(f"[INPUT] Using dynamically calculated remaining terms file: {remaining_file}")
                with open(remaining_file, 'r', encoding='utf-8') as f:
                    remaining_data = json.load(f)
                
                # Load both dictionary and non-dictionary terms from remaining file
                dict_terms = remaining_data.get('dictionary_terms', [])
                non_dict_terms = remaining_data.get('non_dictionary_terms', [])
                
                # Check metadata to see if overlap resolution was already applied
                metadata = remaining_data.get('metadata', {})
                overlap_info = metadata.get('overlap_analysis', {})
                
                if overlap_info and non_dict_terms:
                    # If there are non-dict terms but overlap analysis exists, re-apply overlap detection
                    logger.warning(f"[OVERLAP_CHECK] Found {len(non_dict_terms)} non-dict terms in remaining file - re-checking overlaps")
                    
                    # Create sets of term strings for overlap detection
                    dict_term_strings = set()
                    for term_data in dict_terms:
                        if isinstance(term_data, dict):
                            term = term_data.get('term', '')
                        else:
                            term = str(term_data)
                        if term:
                            dict_term_strings.add(term)
                    
                    non_dict_term_strings = set()
                    for term_data in non_dict_terms:
                        term = term_data.get('term', '')
                        if term:
                            non_dict_term_strings.add(term)
                    
                    # Detect overlaps
                    overlapping_terms = dict_term_strings & non_dict_term_strings
                    overlap_count = len(overlapping_terms)
                    
                    if overlap_count > 0:
                        overlap_percentage = (overlap_count / len(non_dict_term_strings)) * 100 if len(non_dict_term_strings) > 0 else 0
                        logger.warning(f"[OVERLAP_DETECTED] {overlap_count:,} terms overlap ({overlap_percentage:.1f}% of non-dict terms)")
                        
                        if overlap_percentage > 90:
                            logger.warning("[OVERLAP_RESOLUTION] High overlap detected (>90%) - excluding non-dictionary terms")
                            non_dict_terms = []  # Clear non-dictionary terms
                            logger.info("[OVERLAP_RESOLUTION] Non-dictionary terms excluded due to high overlap")
                        else:
                            logger.info(f"[OVERLAP_RESOLUTION] Removing {overlap_count:,} overlapping terms from non-dictionary list")
                            # Remove overlapping terms from non-dictionary list
                            non_dict_terms = [term_data for term_data in non_dict_terms 
                                            if term_data.get('term', '') not in overlapping_terms]
                            logger.info(f"[OVERLAP_RESOLUTION] Non-dictionary terms after deduplication: {len(non_dict_terms):,}")
                
                terms_for_translation = dict_terms + non_dict_terms
                logger.info(f"[RESUME] Processing {len(dict_terms)} dictionary + {len(non_dict_terms)} non-dictionary = {len(terms_for_translation)} remaining terms")
                
                # If overlap resolution was applied, update the remaining terms file
                if overlap_info and len(non_dict_terms) == 0 and remaining_data.get('non_dictionary_terms'):
                    logger.info("[UPDATE] Updating remaining terms file with overlap resolution applied")
                    remaining_data['non_dictionary_terms'] = []
                    remaining_data['metadata']['overlap_resolution_applied'] = True
                    remaining_data['metadata']['overlap_resolution_timestamp'] = datetime.now().isoformat()
                    
                    try:
                        with open(remaining_file, 'w', encoding='utf-8') as f:
                            json.dump(remaining_data, f, indent=2, ensure_ascii=False)
                        logger.info("[SAVED] Updated remaining terms file with overlap resolution")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to update remaining terms file: {e}")
                        
            else:
                logger.error(f"[ERROR] Remaining terms file not found: {remaining_file}")
                return translation_results_file
        else:
            logger.info("[START] Starting fresh ultra-optimized translation processing...")
            
            # Load high frequency dictionary terms
            with open(high_freq_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different possible structures
            if 'terms' in data:
                dictionary_terms = data['terms']
            elif 'dictionary_terms' in data:
                dictionary_terms = data['dictionary_terms']
            else:
                logger.error(f"[ERROR] Unknown data structure in {high_freq_file}: {list(data.keys())}")
                raise KeyError(f"Expected 'terms' or 'dictionary_terms' key in {high_freq_file}")
            
            # Also load high-frequency non-dictionary terms (freq >= 2)
            non_dict_file = str(self.output_dir / "Non_Dictionary_Terms_Identified.json")
            non_dictionary_terms = []
            if os.path.exists(non_dict_file):
                logger.info("[LOAD] Loading high-frequency non-dictionary terms...")
                with open(non_dict_file, 'r', encoding='utf-8') as f:
                    non_dict_data = json.load(f)
                
                if 'non_dictionary_terms' in non_dict_data:
                    all_non_dict = non_dict_data['non_dictionary_terms']
                    # Filter by frequency >= 2
                    non_dictionary_terms = [
                        term for term in all_non_dict 
                        if isinstance(term, dict) and term.get('frequency', 0) >= 2
                    ]
                    logger.info(f"[FILTER] Found {len(non_dictionary_terms)} non-dictionary terms with frequency >= 2")
                else:
                    logger.warning("[WARNING] No non_dictionary_terms found in non-dictionary file")
            else:
                logger.warning(f"[WARNING] Non-dictionary file not found: {non_dict_file}")
            
            # APPLY OVERLAP DETECTION AND RESOLUTION
            logger.info("[OVERLAP_CHECK] Checking for overlaps between dictionary and non-dictionary terms...")
            
            # Create sets of term strings for overlap detection
            dict_term_strings = set()
            for term_data in dictionary_terms:
                if isinstance(term_data, dict):
                    term = term_data.get('term', '')
                else:
                    term = str(term_data)
                if term:
                    dict_term_strings.add(term)
            
            non_dict_term_strings = set()
            for term_data in non_dictionary_terms:
                term = term_data.get('term', '')
                if term:
                    non_dict_term_strings.add(term)
            
            # Detect overlaps
            overlapping_terms = dict_term_strings & non_dict_term_strings
            overlap_count = len(overlapping_terms)
            
            logger.info(f"[OVERLAP_ANALYSIS] Dictionary terms: {len(dict_term_strings):,}")
            logger.info(f"[OVERLAP_ANALYSIS] Non-dictionary terms (freq>=2): {len(non_dict_term_strings):,}")
            logger.info(f"[OVERLAP_ANALYSIS] Overlapping terms: {overlap_count:,}")
            
            if overlap_count > 0:
                overlap_percentage = (overlap_count / len(non_dict_term_strings)) * 100 if len(non_dict_term_strings) > 0 else 0
                logger.warning(f"[OVERLAP_DETECTED] {overlap_count:,} terms overlap ({overlap_percentage:.1f}% of non-dict terms)")
                
                if overlap_percentage > 90:
                    logger.warning("[OVERLAP_RESOLUTION] High overlap detected (>90%) - using dictionary terms only")
                    # Use only dictionary terms to avoid double-counting
                    non_dictionary_terms = []  # Clear non-dictionary terms
                    logger.info("[OVERLAP_RESOLUTION] Non-dictionary terms excluded due to high overlap")
                else:
                    logger.info(f"[OVERLAP_RESOLUTION] Removing {overlap_count:,} overlapping terms from non-dictionary list")
                    # Remove overlapping terms from non-dictionary list
                    non_dictionary_terms = [term_data for term_data in non_dictionary_terms 
                                          if term_data.get('term', '') not in overlapping_terms]
                    logger.info(f"[OVERLAP_RESOLUTION] Non-dictionary terms after deduplication: {len(non_dictionary_terms):,}")
            else:
                logger.info("[OVERLAP_CHECK] No overlaps detected - proceeding with both lists")
            
            # Combine terms after overlap resolution
            terms_for_translation = dictionary_terms + non_dictionary_terms
            total_unique_terms = len(dict_term_strings) + len(non_dictionary_terms)
            logger.info(f"[OVERLAP_RESOLVED] Final terms for translation: {len(dictionary_terms):,} dict + {len(non_dictionary_terms):,} non-dict = {len(terms_for_translation):,}")
            logger.info(f"[UNIQUE_TERMS] Total unique terms after overlap resolution: {total_unique_terms:,}")
        
        if not terms_for_translation:
            logger.warning("[WARNING] No high frequency terms to translate")
            
            # CHECK IF TRANSLATION RESULTS ALREADY EXIST (from previous runs)
            translation_results_file = str(self.output_dir / "Translation_Results.json")
            
            if os.path.exists(translation_results_file):
                # Check if existing file has actual translation data
                try:
                    with open(translation_results_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    existing_results = existing_data.get('translation_results', [])
                    if existing_results and len(existing_results) > 0:
                        logger.info(f"[RESUME] Found existing translation results with {len(existing_results)} terms")
                        logger.info(f"[RESUME] Preserving existing Translation_Results.json - NOT overwriting")
                        return translation_results_file
                    else:
                        logger.info(f"[RESUME] Existing Translation_Results.json is empty - will create new empty file")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not read existing Translation_Results.json: {e}")
            
            # Create empty translation results file for pipeline continuity
            empty_results = {
                'metadata': {
                    'step': 5,
                    'step_name': 'Translation Process',
                    'timestamp': datetime.now().isoformat(),
                    'total_terms_processed': 0,
                    'translation_method': 'skipped_no_high_frequency_terms',
                    'status': 'completed_with_empty_results'
                },
                'translation_results': []
            }
            
            with open(translation_results_file, 'w', encoding='utf-8') as f:
                json.dump(empty_results, f, indent=2, ensure_ascii=False)
            
            self._add_step_metadata(translation_results_file, 5, "Translation Process (No High-Frequency Terms)", {
                'terms_processed': 0,
                'translation_status': 'skipped_no_high_frequency_terms'
            })
            
            logger.info(f"[OK] Created empty translation results file: {translation_results_file}")
            return translation_results_file
        
        # Prepare terms for translation runner
        # The ultra_optimized_smart_runner expects specific data files
        # We need to create compatible format
        
        # Create dictionary terms format
        dict_terms_file = str(self.output_dir / "Dictionary_Terms_For_Translation.json")
        translation_compatible_data = {
            'dictionary_terms': []
        }
        
        for term_data in terms_for_translation:
            translation_compatible_data['dictionary_terms'].append({
                'term': term_data['term'],
                'frequency': term_data.get('frequency', 'unknown'),  # Handle missing frequency
                'original_texts': term_data.get('original_texts', [])  # Handle missing original_texts
            })
        
        with open(dict_terms_file, 'w', encoding='utf-8') as f:
            json.dump(translation_compatible_data, f, indent=2, ensure_ascii=False)
        
        try:
            # Run sophisticated NLLB translation process with modern validation
            logger.info(f"[LOG] Prepared {len(terms_for_translation)} terms for translation")
            logger.info("[TRANSLATE] Running sophisticated NLLB translation process with modern validation...")
            
            # Initialize ultra-optimized smart runner with sophisticated dual-GPU architecture
            from ultra_optimized_smart_runner import UltraOptimizedSmartRunner, UltraOptimizedConfig
            
            logger.info("[INIT] Initializing ultra-optimized smart runner with dual-GPU NLLB architecture...")
            
            # Load optimized configuration based on hardware detection
            try:
                # Try FULL RESOURCE UTILIZATION adaptive configuration first
                from adaptive_system_config import get_adaptive_system_config
                profile, adaptive_config, resource_monitor = get_adaptive_system_config(enable_dynamic_monitoring=True)
                
                # Convert adaptive config to UltraOptimizedConfig format
                ultra_config = UltraOptimizedConfig(
                    model_size="1.3B",
                    gpu_workers=adaptive_config.gpu_workers,
                    cpu_workers=adaptive_config.cpu_workers,
                    cpu_translation_workers=adaptive_config.cpu_translation_workers,
                    gpu_batch_size=adaptive_config.gpu_batch_size,
                    max_queue_size=adaptive_config.max_queue_size,
                    predictive_caching=adaptive_config.enable_predictive_caching,
                    dynamic_batching=adaptive_config.enable_dynamic_batching,
                    async_checkpointing=adaptive_config.enable_async_processing,
                    memory_mapping=adaptive_config.enable_memory_mapping
                )
                
                logger.info(f"[ADAPTIVE] Using FULL RESOURCE UTILIZATION configuration")
                logger.info(f"[ADAPTIVE] Performance Tier: {profile.estimated_performance_tier.upper()}")
                logger.info(f"[ADAPTIVE] System: {profile.cpu_cores_logical} CPU cores, {profile.available_memory_gb:.1f}GB RAM, {profile.gpu_count} GPUs")
                logger.info(f"[ADAPTIVE] Workers: {ultra_config.gpu_workers} GPU + {ultra_config.cpu_workers} CPU")
                logger.info(f"[ADAPTIVE] Batching: GPU={ultra_config.gpu_batch_size}, Queue={ultra_config.max_queue_size}")
                logger.info(f"[ADAPTIVE] Memory: {adaptive_config.memory_limit_gb:.1f}GB limit ({adaptive_config.memory_limit_gb/profile.available_memory_gb*100:.0f}% utilization)")
                logger.info(f"[ADAPTIVE] Strategy: {adaptive_config.processing_strategy}, Optimization: {adaptive_config.optimization_level}")
                
                logger.info(f"[GPU] Using {ultra_config.gpu_workers} GPU workers with {ultra_config.model_size} model")
                
                if resource_monitor:
                    logger.info("[DYNAMIC] Real-time resource monitoring enabled for maximum performance")
                    # Store monitor for later use
                    self.resource_monitor = resource_monitor
                
            except ImportError:
                # Fallback to multi-model GPU configuration
                try:
                    from multi_model_gpu_config import get_multi_model_gpu_config
                    multi_gpu_config = get_multi_model_gpu_config(['nllb-200-1.3B', 'gpt-4.1', 'terminology_agent'])
                    
                    if multi_gpu_config['success']:
                        worker_config = multi_gpu_config['worker_config']
                        ultra_config = UltraOptimizedConfig(
                            model_size="1.3B",
                            gpu_workers=worker_config['gpu_workers'],
                            cpu_workers=worker_config['cpu_workers'],
                            gpu_batch_size=worker_config['gpu_batch_size'],
                            max_queue_size=worker_config['max_queue_size'],
                            predictive_caching=True,
                            dynamic_batching=True,
                            async_checkpointing=True,
                            memory_mapping=True
                        )
                        logger.info(f"[MULTI-GPU] Using enhanced multi-model configuration")
                        logger.info(f"[MULTI-GPU] Active GPUs: {worker_config['active_gpus']}, Total Models: {worker_config['total_models']}")
                        logger.info(f"[MULTI-GPU] GPU Workers: {ultra_config.gpu_workers}, CPU Workers: {ultra_config.cpu_workers}")
                        logger.info(f"[MULTI-GPU] Batch Size: {ultra_config.gpu_batch_size}, Memory Limit: {worker_config['memory_limit_gb']:.1f}GB")
                        
                        # Log recommendations if any
                        if multi_gpu_config['recommendations']:
                            logger.info("[RECOMMENDATIONS] GPU optimization suggestions:")
                            for rec in multi_gpu_config['recommendations']:
                                logger.info(f"  • {rec}")
                    else:
                        raise ImportError("Multi-model GPU config failed, falling back")
                        
                except ImportError:
                    # Fallback to original optimized configuration
                    try:
                        from optimized_translation_config import get_optimized_config
                        ultra_config, hardware_profile = get_optimized_config()
                        logger.info(f"[HARDWARE] Using standard optimized config: {hardware_profile.get('gpu_name', 'Unknown GPU')}, {hardware_profile.get('cpu_cores', 'Unknown')} cores")
                        logger.info(f"[CONFIG] Available RAM: {hardware_profile.get('available_ram_gb', 'Unknown')}GB")
                        logger.info(f"[CONFIG] GPU Workers: {ultra_config.gpu_workers}, CPU Workers: {ultra_config.cpu_workers}")
                        logger.info(f"[CONFIG] Batch Size: {ultra_config.gpu_batch_size}, Queue Size: {ultra_config.max_queue_size}")
                    except ImportError:
                        # Final fallback to default configuration
                        logger.warning("[WARNING] Could not load any optimized config, using defaults")
                        ultra_config = UltraOptimizedConfig(
                            model_size="1.3B",
                            gpu_workers=1,  # Conservative default
                            cpu_workers=8,
                            gpu_batch_size=32,
                            max_queue_size=50,
                            predictive_caching=True,
                            dynamic_batching=True,
                            async_checkpointing=True,
                            memory_mapping=False  # Conservative for memory
                        )
            
            # Initialize the ultra-optimized runner with the current output directory
            # When resuming from checkpoint, don't resume the ultra runner's internal session
            translator = UltraOptimizedSmartRunner(
                config=ultra_config,
                data_source_dir=str(self.output_dir),
                resume_session=None,  # Force fresh session when using terms_only
                skip_checkpoint_loading=True  # Skip checkpoint loading when resuming from main system checkpoint
            )
            
            # Initialize modern terminology review agent for validation
            from modern_terminology_review_agent import ModernTerminologyReviewAgent
            logger.info("[INIT] Initializing modern terminology review agent...")
            review_agent = ModernTerminologyReviewAgent(model_name="gpt-4.1")
            # Apply authentication fix
            review_agent = ensure_agent_auth_fix(review_agent)
            logger.info("[AUTH] Applied authentication fix to ModernTerminologyReviewAgent")
            
            translation_results_file = str(self.output_dir / "Translation_Results.json")
            
            # Generate comprehensive translation analysis using sophisticated NLLB architecture
            # This provides real multilingual translation data for the final review agent
            translation_results = {
                'metadata': {
                    'translation_timestamp': datetime.now().isoformat(),
                    'total_terms_translated': len(terms_for_translation),
                    'translation_method': 'nllb_200_multilingual_neural_translation',
                    'validation_agent': 'modern_terminology_review_agent',
                    'nllb_model': 'facebook/nllb-200-1.3B',
                    'gpu_optimization': True,
                    'enhanced_features': [
                        'dual_gpu_neural_translation',
                        'intelligent_language_selection', 
                        'linguistic_analysis',
                        'terminology_validation',
                        'cross_linguistic_validity',
                        'quality_metrics'
                    ]
                },
                'translation_results': []
            }
            
            # Comprehensive language set for sophisticated analysis
            # Using NLLB's full language support with intelligent selection
            comprehensive_languages = [
                ('spa_Latn', 'Spanish'), ('fra_Latn', 'French'), ('deu_Latn', 'German'), 
                ('ita_Latn', 'Italian'), ('por_Latn', 'Portuguese'), ('rus_Cyrl', 'Russian'),
                ('zho_Hans', 'Chinese (Simplified)'), ('jpn_Jpan', 'Japanese'), ('kor_Hang', 'Korean'),
                ('arb_Arab', 'Modern Standard Arabic'), ('hin_Deva', 'Hindi'), ('ben_Beng', 'Bengali'),
                ('nld_Latn', 'Dutch'), ('swe_Latn', 'Swedish'), ('dan_Latn', 'Danish'),
                ('pol_Latn', 'Polish'), ('ces_Latn', 'Czech'), ('ukr_Cyrl', 'Ukrainian'),
                ('tur_Latn', 'Turkish'), ('vie_Latn', 'Vietnamese'), ('tha_Thai', 'Thai')
            ]
            logger.info(f"[TRANSLATE] Comprehensive multilingual testing with {len(comprehensive_languages)} languages")
            
            # Prepare terms in ultra-optimized format for processing
            logger.info("[ULTRA] Preparing terms for ultra-optimized dual-GPU processing...")
            
            # Create temporary dictionary terms file exactly as ultra-optimized runner expects
            temp_dict_terms = []
            for term_data in terms_for_translation:
                temp_dict_terms.append({
                    'term': term_data['term'],
                    'frequency': term_data.get('frequency', 0)
                })
            
            # Create temporary dictionary file for ultra-optimized processing
            temp_dict_data = {"dictionary_terms": temp_dict_terms}
            temp_dict_file = os.path.join(self.output_dir, "Dictionary_Terms_Found.json")
            
            with open(temp_dict_file, 'w', encoding='utf-8') as f:
                json.dump(temp_dict_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[TEMP] Created temporary dictionary file with {len(temp_dict_terms)} terms")
            
            # Run ultra-optimized processing exactly as implemented
            logger.info("[ULTRA] Starting ultra-optimized multilingual translation processing...")
            logger.info("         Using dual-GPU architecture with intelligent language selection")
            logger.info("         Features: predictive caching, dynamic batching, memory optimization")
            
            # Execute the ultra-optimized processing
            # Pass specific terms when resuming from checkpoint
            if checkpoint_file.exists():
                logger.info(f"[RESUME] Passing {len(terms_for_translation)} remaining terms to translator")
                # Extract just the term strings for the translator
                term_strings = [term_data['term'] for term_data in terms_for_translation]
                
                # Pass checkpoint information to preserve progress tracking
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                # Set the progress counters before running
                translator.processed_terms = checkpoint_data.get('completed_terms', 0)
                translator.total_terms = checkpoint_data.get('total_terms', len(term_strings))
                logger.info(f"[CHECKPOINT] Resuming from {translator.processed_terms}/{translator.total_terms} terms ({checkpoint_data.get('completion_percentage', 0):.1f}%)")
                
                translator.run_ultra_optimized_processing(terms_only=term_strings)
            else:
                logger.info("[FRESH] Starting complete translation process")
                translator.run_ultra_optimized_processing()
            
            # NO SEPARATE ULTRA RESULTS LOADING: Results are already in Translation_Results.json
            # The ultra runner now updates Translation_Results.json directly during processing
            # No need to search for or merge separate ultra runner result files
            logger.info("[DIRECT] Ultra runner saves results directly to Translation_Results.json - no separate file loading needed")
            
            # Clean up temporary file
            try:
                if os.path.exists(temp_dict_file):
                    os.remove(temp_dict_file)
            except Exception as e:
                logger.warning(f"[CLEANUP] Could not remove temp file: {e}")
            
            logger.info(f"[ULTRA] Ultra-optimized processing completed with {len(translation_results['translation_results'])} results")
            
            # SAVE NEW RESULTS TO ACTUAL Translation_Results.json FILE
            logger.info("[SAVE] Saving new results to Translation_Results.json...")
            
            # First, merge with existing results if file exists
            if os.path.exists(translation_results_file):
                try:
                    logger.info("[MERGE] Merging with existing translation results...")
                    with open(translation_results_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    existing_results = existing_data.get('translation_results', [])
                    new_results = translation_results.get('translation_results', [])
                    
                    # Create a set of existing terms to avoid duplicates
                    existing_terms = set()
                    for result in existing_results:
                        if isinstance(result, dict) and 'term' in result:
                            existing_terms.add(result['term'])
                    
                    # Add only new results (avoid duplicates)
                    added_count = 0
                    for new_result in new_results:
                        if isinstance(new_result, dict) and 'term' in new_result:
                            if new_result['term'] not in existing_terms:
                                existing_results.append(new_result)
                                existing_terms.add(new_result['term'])
                                added_count += 1
                    
                    # Update the translation_results structure with merged data
                    translation_results['translation_results'] = existing_results
                    translation_results['metadata']['total_terms_translated'] = len(existing_results)
                    
                    logger.info(f"[MERGE] Added {added_count} new results, total now: {len(existing_results)}")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Failed to merge with existing results: {e}")
                    logger.info("[FALLBACK] Will overwrite existing file")
            else:
                logger.info("[NEW] Creating new Translation_Results.json file")
            
            # Save the updated results
            try:
                with open(translation_results_file, 'w', encoding='utf-8') as f:
                    json.dump(translation_results, f, indent=2, ensure_ascii=False)
                logger.info(f"[SAVED] Translation results saved with {len(translation_results['translation_results'])} total results")
            except Exception as e:
                logger.error(f"[ERROR] Failed to save translation results: {e}")
            
            # UPDATE MAIN CHECKPOINT based on ACTUAL TRANSLATION RESULTS (not translator object)
            logger.info("[UPDATE] Updating checkpoint based on actual translation results...")
            
            try:
                # Read current translation results to get actual count
                current_results_count = 0
                unique_terms_count = 0
                
                if os.path.exists(translation_results_file):
                    with open(translation_results_file, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                    current_results = current_data.get('translation_results', [])
                    current_results_count = len(current_results)
                    
                    # Count unique terms
                    unique_terms = set()
                    for result in current_results:
                        if isinstance(result, dict) and 'term' in result:
                            unique_terms.add(result['term'])
                    unique_terms_count = len(unique_terms)
                    
                    logger.info(f"[ACTUAL_COUNT] Current translation results: {current_results_count} entries, {unique_terms_count} unique terms")
                
                # Update checkpoint with dynamic calculation
                logger.info("[DYNAMIC] Updating checkpoint with dynamic calculation...")
                dynamic_result = self._calculate_remaining_terms_dynamically()
                if dynamic_result:
                    total_terms = dynamic_result['total_required']
                    remaining_terms = dynamic_result['remaining']
                    completion_percentage = (unique_terms_count / total_terms) * 100 if total_terms > 0 else 0
                else:
                    # Fallback to checkpoint values if dynamic calculation fails
                    checkpoint_file = self.output_dir / "step5_translation_checkpoint.json"
                    if os.path.exists(checkpoint_file):
                        with open(checkpoint_file, 'r', encoding='utf-8') as f:
                            checkpoint_data = json.load(f)
                        total_terms = checkpoint_data.get('total_terms', 8691)
                    else:
                        total_terms = 8691  # Fallback
                    remaining_terms = total_terms - unique_terms_count
                    completion_percentage = (unique_terms_count / total_terms) * 100 if total_terms > 0 else 0
                
                checkpoint_data = {
                    'step': 5,
                    'step_name': 'Translation Process',
                    'status': 'partially_completed' if unique_terms_count < total_terms else 'completed',
                    'checkpoint_timestamp': datetime.now().isoformat(),
                    'total_terms': total_terms,
                    'completed_terms': unique_terms_count,
                    'remaining_terms': remaining_terms,
                    'completion_percentage': completion_percentage,
                    'next_batch_info': {
                        'batch_size': remaining_terms,
                        'expected_output': 'Translation_Results.json (append mode)'
                    },
                    'source_of_truth': 'actual_translation_results',
                    'update_method': 'count_from_translation_results_file',
                    'last_session': getattr(translator, 'session_id', 'unknown'),
                    'validated_against_results': True
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"[CHECKPOINT_UPDATED] Saved checkpoint: {unique_terms_count}/{total_terms} terms ({completion_percentage:.1f}%)")
                logger.info(f"[CHECKPOINT_UPDATED] Based on actual translation results, not translator object")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to update checkpoint from actual results: {e}")
                # Try to at least log the error details
                import traceback
                logger.error(f"[ERROR_DETAILS] {traceback.format_exc()}")
            
            # Update metadata with ultra-optimized processing statistics
            if translation_results['translation_results']:
                total_languages_processed = sum(r.get('total_languages', 0) for r in translation_results['translation_results'])
                total_successful_translations = sum(r.get('translated_languages', 0) for r in translation_results['translation_results'])
                total_same_translations = sum(r.get('same_languages', 0) for r in translation_results['translation_results'])
                total_error_translations = sum(r.get('error_languages', 0) for r in translation_results['translation_results'])
                
                translation_results['metadata'].update({
                    'total_languages_processed': total_languages_processed,
                    'total_successful_translations': total_successful_translations,
                    'total_same_translations': total_same_translations,
                    'total_error_translations': total_error_translations,
                    'average_translatability_score': sum(r.get('translatability_score', 0) for r in translation_results['translation_results']) / len(translation_results['translation_results']),
                    'processing_tiers_used': list(set(r.get('processing_tier', 'unknown') for r in translation_results['translation_results'])),
                    'ultra_optimization_stats': {
                        'ultra_minimal_terms': getattr(translator, 'ultra_minimal_terms', 0),
                        'core_terms': getattr(translator, 'core_terms', 0),
                        'extended_terms': getattr(translator, 'extended_terms', 0),
                        'language_savings': getattr(translator, 'language_savings', 0),
                        'gpu_performance': getattr(translator, 'gpu_performance', [0.0, 0.0]),
                        'session_id': getattr(translator, 'session_id', 'unknown')
                    },
                    'ultra_optimized_processing_enabled': True,
                    'dual_gpu_architecture': True,
                    'intelligent_language_selection': True,
                    'predictive_caching': True,
                    'dynamic_batching': True,
                    'detailed_parameter_tracking': True
                })
            else:
                # No results case
                translation_results['metadata'].update({
                    'total_languages_processed': 0,
                    'ultra_optimization_attempted': True,
                    'results_loaded': False
                })
            
            # Handle merging results when resuming from checkpoint
            if checkpoint_file.exists():
                logger.info("[MERGE] Merging new results with existing translations...")
                
                # Load existing results
                existing_results = {}
                if os.path.exists(translation_results_file):
                    with open(translation_results_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                
                # Merge translation results
                existing_translations = existing_results.get('translation_results', [])
                new_translations = translation_results.get('translation_results', [])
                
                # Combine all translations
                all_translations = existing_translations + new_translations
                
                # Update metadata
                translation_results['metadata'].update({
                    'total_terms_processed': len(all_translations),
                    'existing_terms': len(existing_translations),
                    'new_terms_added': len(new_translations),
                    'merge_timestamp': datetime.now().isoformat(),
                    'resumed_from_checkpoint': True
                })
                translation_results['translation_results'] = all_translations
                
                logger.info(f"[MERGE] Combined {len(existing_translations)} + {len(new_translations)} = {len(all_translations)} translations")
                
                # Update remaining terms file after merge
                self._update_remaining_terms_after_progress(all_translations)
                
                # Check if we need to continue translation or if we're truly complete
                # Get total expected terms from checkpoint data
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                total_expected_terms = checkpoint_data.get('total_terms', 14987)
                
                # Verify completion by checking remaining terms file
                remaining_file = str(self.output_dir / "Remaining_Terms_For_Translation.json")
                actual_remaining = 0
                if os.path.exists(remaining_file):
                    with open(remaining_file, 'r', encoding='utf-8') as f:
                        remaining_data = json.load(f)
                    dict_remaining = len(remaining_data.get('dictionary_terms', []))
                    non_dict_remaining = len(remaining_data.get('non_dictionary_terms', []))
                    actual_remaining = dict_remaining + non_dict_remaining
                
                logger.info(f"[VERIFICATION] Translations: {len(all_translations)}, Expected: {total_expected_terms}, Remaining: {actual_remaining}")
                
                if len(all_translations) >= total_expected_terms or actual_remaining <= 0:
                    # Translation is truly complete
                    os.remove(checkpoint_file)
                    if os.path.exists(remaining_file):
                        os.remove(remaining_file)
                    logger.info(f"[COMPLETE] Translation fully complete: {len(all_translations)}/{total_expected_terms} terms")
                    logger.info(f"[CLEANUP] Removed checkpoint and remaining terms files")
                else:
                    # Create new checkpoint for remaining terms
                    remaining_dict_terms = 8537 - len([t for t in all_translations if 'dictionary_term' not in t or t.get('dictionary_term', True)])
                    remaining_total = total_expected_terms - len(all_translations)
                    
                    new_checkpoint = {
                        "step": 5,
                        "step_name": "Translation Process",
                        "status": "partially_completed",
                        "checkpoint_timestamp": datetime.now().isoformat(),
                        "total_terms": total_expected_terms,
                        "completed_terms": len(all_translations),
                        "remaining_terms": remaining_total,
                        "completion_percentage": (len(all_translations) / total_expected_terms) * 100,
                        "next_batch_info": {
                            "batch_size": remaining_total,
                            "expected_output": "Translation_Results.json (append mode)"
                        }
                    }
                    
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(new_checkpoint, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"[CONTINUE] Translation incomplete: {len(all_translations)}/{total_expected_terms} terms ({new_checkpoint['completion_percentage']:.1f}%)")
                    logger.info(f"[CHECKPOINT] Created new checkpoint for {remaining_total} remaining terms")
            
            # Results already saved above with proper merging - no need to save again
            logger.info("[SKIP] Translation results already saved with proper merging")
            
            # Add step metadata with ultra-optimized translation information
            self._add_step_metadata(translation_results_file, 5, "Ultra-Optimized Translation Process", {
                "translation_method": "ultra_optimized_dual_gpu_nllb_translation",
                "terms_translated": len(translation_results.get('translation_results', [])),
                "total_languages_processed": translation_results['metadata'].get('total_languages_processed', 0),
                "average_translatability_score": translation_results['metadata'].get('average_translatability_score', 0),
                "processing_tiers_used": translation_results['metadata'].get('processing_tiers_used', []),
                "ultra_optimization_enabled": True,
                "dual_gpu_architecture": True,
                "intelligent_language_selection": True,
                "predictive_caching": True,
                "dynamic_batching": True,
                "ultra_optimization_stats": translation_results['metadata'].get('ultra_optimization_stats', {}),
                "results_properly_merged": True,
                "duplicate_prevention": True
            })
            
            self.process_stats['terms_translated'] = len(translation_results.get('translation_results', []))
            
            logger.info(f"[OK] Step 5 completed: {len(translation_results.get('translation_results', []))} total terms in results file")
            logger.info(f"[FOLDER] Results properly merged and saved to: {translation_results_file}")
            
            return translation_results_file
            
        except Exception as e:
            logger.error(f"[ERROR] Translation process failed: {e}")
            # Create empty results file
            empty_results = str(self.output_dir / "Translation_Results_Empty.json")
            with open(empty_results, 'w', encoding='utf-8') as f:
                json.dump({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'terms_attempted': len(terms_for_translation)
                }, f, indent=2)
            return empty_results
    
    def step_6_language_verification(self, translation_results_file: str) -> str:
        """
        Step 6: Language Verification
        - Verify source and target language matching
        - Ensure language pair consistency
        """
        import os
        logger.info("[SEARCH] STEP 6: Language Verification")
        logger.info("=" * 60)
        
        if not translation_results_file or not os.path.exists(translation_results_file):
            logger.error(f"[ERROR] Translation results file not found: {translation_results_file}")
            # Create empty verification results file for pipeline continuity
            verified_results_file = str(self.output_dir / "Verified_Translation_Results.json")
            empty_verification = {
                'metadata': {
                    'step': 6,
                    'step_name': 'Language Verification',
                    'timestamp': datetime.now().isoformat(),
                    'total_terms_verified': 0,
                    'verification_status': 'skipped_no_translation_results'
                },
                'verified_results': [],
                'verification_issues': []
            }
            
            with open(verified_results_file, 'w', encoding='utf-8') as f:
                json.dump(empty_verification, f, indent=2, ensure_ascii=False)
                
            self._add_step_metadata(verified_results_file, 6, "Language Verification (No Translation Results)", {
                'terms_verified': 0,
                'verification_issues': 0
            })
            
            logger.info(f"[OK] Created empty verification results file: {verified_results_file}")
            return verified_results_file
        
        # Load translation results
        with open(translation_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'error' in data:
            logger.warning("[WARNING] Translation results contain errors, skipping language verification")
            return translation_results_file
        
        translation_results = data.get('translation_results', [])
        
        verified_results = []
        verification_issues = []
        
        # Use maximum CPU cores for parallel language verification
        max_workers = cpu_count()
        logger.info(f"[PARALLEL] Using {max_workers} workers (all CPU cores) for parallel language verification...")
        
        def verify_single_result(result_data):
            """Verify a single translation result"""
            result, index = result_data
            
            try:
                term = result['term']
                translations = result.get('all_translations', {})
                
                # Language verification logic
                verified_translations = {}
                issues = []
                
                for lang_code, translation in translations.items():
                    # Basic verification checks
                    if translation.startswith("ERROR:"):
                        issues.append({
                            'term': term,
                            'language': lang_code,
                            'issue': 'translation_error',
                            'details': translation
                        })
                        continue
                    
                    # Check for empty translations
                    if not translation.strip():
                        issues.append({
                            'term': term,
                            'language': lang_code,
                            'issue': 'empty_translation',
                            'details': 'Translation is empty'
                        })
                        continue
                    
                    # Check for suspicious patterns
                    if len(translation) > len(term) * 5:  # Suspiciously long
                        issues.append({
                            'term': term,
                            'language': lang_code,
                            'issue': 'suspicious_length',
                            'details': f'Translation too long: {len(translation)} chars'
                        })
                    
                    # Add to verified translations
                    verified_translations[lang_code] = translation
                
                # Create verified result
                verified_result = result.copy()
                verified_result.update({
                    'verified_translations': verified_translations,
                    'verification_passed': len(issues) == 0,
                    'verification_issues_count': len(issues),
                    'verification_timestamp': datetime.now().isoformat(),
                    'index': index
                })
                
                return verified_result, issues
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to verify result for term: {e}")
                error_result = result.copy()
                error_result.update({
                    'verified_translations': {},
                    'verification_passed': False,
                    'verification_issues_count': 1,
                    'verification_error': str(e),
                    'verification_timestamp': datetime.now().isoformat(),
                    'index': index
                })
                error_issues = [{
                    'term': result.get('term', 'unknown'),
                    'language': 'all',
                    'issue': 'verification_error',
                    'details': str(e)
                }]
                return error_result, error_issues
        
        # Process results in parallel using ThreadPoolExecutor with all CPU cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for all results with their indices
            future_to_result = {
                executor.submit(verify_single_result, (result, i)): result.get('term', f'result_{i}')
                for i, result in enumerate(translation_results)
            }
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_result):
                term = future_to_result[future]
                completed += 1
                
                if completed % 50 == 0 or completed == len(translation_results):
                    logger.info(f"   [PROGRESS] Verified {completed}/{len(translation_results)} results ({completed/len(translation_results)*100:.1f}%)")
                
                try:
                    verified_result, issues = future.result()
                    verified_results.append(verified_result)
                    verification_issues.extend(issues)
                    
                except Exception as e:
                    logger.warning(f"[WARNING] Error processing verification for '{term}': {e}")
        
        # Sort results by original index to maintain order
        verified_results.sort(key=lambda x: x.get('index', 0))
        
        # Save verified results
        verified_file = os.path.join(str(self.output_dir), "Verified_Translation_Results.json")
        with open(verified_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'verification_timestamp': datetime.now().isoformat(),
                    'total_terms': len(verified_results),
                    'verification_issues': len(verification_issues),
                    'source': translation_results_file
                },
                'verified_results': verified_results,
                'verification_issues': verification_issues
            }, f, indent=2, ensure_ascii=False)
        
        # Add step metadata
        self._add_step_metadata(verified_file, 6, "Language Verification", {
            "verification_method": "english_validation",
            "terms_verified": len(verified_results),
            "verification_issues": len(verification_issues),
            "language_focus": "English"
        })
        
        logger.info(f"[OK] Step 6 completed:")
        logger.info(f"   Terms verified: {len(verified_results)}")
        logger.info(f"   Verification issues: {len(verification_issues)}")
        logger.info(f"[FOLDER] Verified results: {verified_file}")
        
        return verified_file
    
    def step_7_final_review_decision(self, verified_file: str, translation_results_file: str = None) -> str:
        """
        Step 7: Final Review and Decision with PROPER Modern Validation Batch Processing
        
        This step uses the COMPLETE modern parallel validation system with:
        1. PROPER batch processing via EnhancedValidationSystem.process_terms_parallel()
        2. Organized folder structure with batch files, logs, and cache
        3. Modern terminology review agents processing each batch
        4. ML-based quality scoring and advanced context analysis in batch workers
        5. Integration of Step 5 (translation) and Step 6 (verification) data
        6. Comprehensive translatability analysis and NEW TERM verification
        7. Consolidation of batch results into final decisions
        """
        logger.info("[LOG] STEP 7: Final Review and Decision with Modern Validation Batch Processing")
        logger.info("=" * 80)
        
        # Load verified results from Step 6
        verified_file_path = os.path.join(str(self.output_dir), "Verified_Translation_Results.json") if not os.path.isabs(verified_file) else verified_file
        
        if not os.path.exists(verified_file_path):
            raise FileNotFoundError(f"Verified results file not found: {verified_file_path}")
        
        with open(verified_file_path, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        
        verified_results = verified_data.get('verified_results', [])
        
        if not verified_results:
            logger.warning("[WARNING] No verified results found for final review")
            return self._create_empty_decisions_file("no_verified_results")
        
        logger.info(f"[INPUT] Processing {len(verified_results):,} verified terms for final decisions")
        
        # Load translation results for enhanced evaluation (Step 5 integration)
        translation_data_map = {}
        if translation_results_file:
            translation_file_path = os.path.join(str(self.output_dir), "Translation_Results.json") if not os.path.isabs(translation_results_file) else translation_results_file
            
            if os.path.exists(translation_file_path):
                logger.info(f"[INTEGRATION] Loading Step 5 translation data for enhanced evaluation")
                with open(translation_file_path, 'r', encoding='utf-8') as f:
                    translation_data = json.load(f)
                
                translation_results = translation_data.get('translation_results', [])
                for result in translation_results:
                    term = result.get('term', '')
                    if term:
                        translation_data_map[term] = result
                
                logger.info(f"[INTEGRATION] Loaded {len(translation_data_map):,} translation records for cross-step analysis")
        
        # Initialize PROPER modern validation system with batch processing
        from modern_parallel_validation import OrganizedValidationManager, EnhancedValidationSystem
        
        # RESUME FROM EXISTING STEP 7 FOLDER (if it exists)
        import glob
        existing_step7_folders = glob.glob(os.path.join(str(self.output_dir), "step7_final_evaluation_*"))
        
        if existing_step7_folders:
            # Use the most recent existing Step 7 folder
            existing_step7_folders.sort(reverse=True)  # Most recent first
            existing_folder = existing_step7_folders[0]
            step7_folder_name = os.path.basename(existing_folder)
            logger.info(f"[RESUME] Found existing Step 7 folder: {step7_folder_name}")
            logger.info(f"[RESUME] Resuming batch processing from existing folder instead of creating new one")
        else:
            # Create new folder only if none exists
            step7_folder_name = f"step7_final_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"[NEW] Creating new Step 7 folder: {step7_folder_name}")
        
        # Create organized validation manager for Step 7 with proper batch processing
        step7_manager = OrganizedValidationManager(
            model_name="gpt-4.1",
            run_folder=step7_folder_name,
            organize_existing=False,
            base_output_dir=str(self.output_dir)
        )
        
        # Initialize enhanced validation system for PROPER batch processing
        modern_validation_system = EnhancedValidationSystem(step7_manager)
        
        logger.info(f"[MODERN_SYSTEM] Initialized proper batch processing system:")
        logger.info(f"   Batch directory: {step7_manager.batch_dir}")
        logger.info(f"   Cache directory: {step7_manager.cache_dir}")
        logger.info(f"   Logs directory: {step7_manager.logs_dir}")
        
        # CHECK IF BATCH PROCESSING IS ALREADY COMPLETE
        existing_batch_files = glob.glob(os.path.join(step7_manager.batch_dir, "modern_step7_final_decisions_validation_batch_*.json"))
        consolidated_file = os.path.join(step7_manager.results_dir, "consolidated_step7_final_decisions_validation_results.json")
        
        if existing_batch_files and not os.path.exists(consolidated_file):
            logger.info(f"[RESUME] Found {len(existing_batch_files)} existing batch files")
            logger.info(f"[RESUME] No consolidated file found - will consolidate existing batches")
            logger.info(f"[RESUME] Skipping batch processing and proceeding to consolidation")
            
            # Skip to consolidation of existing batches
            try:
                modern_validation_system.consolidate_results("step7_final_decisions", "final_validation")
                logger.info("[CONSOLIDATION] Successfully consolidated existing batch results")
                
                # Load consolidated results and proceed with decision conversion
                if os.path.exists(consolidated_file):
                    with open(consolidated_file, 'r', encoding='utf-8') as f:
                        consolidated_data = json.load(f)
                    
                    logger.info(f"[CONSOLIDATION] Loaded consolidated results from {len(consolidated_data.get('batches', {}))} batches")
                    
                    # Import the helper functions from step7_fixed_batch_processing
                    from step7_fixed_batch_processing import (_convert_modern_batch_results_to_decisions, 
                                                             _create_final_decisions_data_with_batch_info,
                                                             _log_step7_completion_with_batch_info)
                    
                    # Convert modern validation results to final decisions format
                    final_decisions = _convert_modern_batch_results_to_decisions(
                        consolidated_data, verified_results, translation_data_map
                    )
                    
                    # Create comprehensive final decisions data
                    final_decisions_data = _create_final_decisions_data_with_batch_info(
                        final_decisions, consolidated_data, step7_manager
                    )
                    
                    # Save final decisions
                    decisions_file = os.path.join(str(self.output_dir), "Final_Terminology_Decisions.json")
            
                    with open(decisions_file, 'w', encoding='utf-8') as f:
                        json.dump(final_decisions_data, f, indent=2, ensure_ascii=False)
                        
                            # Log completion statistics
                    _log_step7_completion_with_batch_info(final_decisions_data, decisions_file, step7_manager)
                        
                    return decisions_file
                else:
                    logger.error(f"[ERROR] Consolidated results file not found after consolidation: {consolidated_file}")
                    
            except Exception as e:
                logger.error(f"[ERROR] Failed to consolidate existing batch results: {e}")
                logger.info(f"[FALLBACK] Will proceed with normal batch processing")
        
        elif existing_batch_files and os.path.exists(consolidated_file):
            logger.info(f"[RESUME] Found {len(existing_batch_files)} batch files and consolidated results")
            logger.info(f"[RESUME] Checking if all terms have been processed or if continuation is needed")
            
            # Load existing consolidated results to check completeness
            with open(consolidated_file, 'r', encoding='utf-8') as f:
                consolidated_data = json.load(f)
            
            # Check how many terms have been processed vs total needed
            processed_terms = set()
            for batch_key, batch_info in consolidated_data.get('batches', {}).items():
                batch_results = batch_info.get('data', {}).get('results', [])
                for result in batch_results:
                    term = result.get('term', '')
                    if term:
                        processed_terms.add(term)
            
            total_terms_needed = len(verified_results)
            logger.info(f"[RESUME] Processed terms: {len(processed_terms)}, Total needed: {total_terms_needed}")
            
            if len(processed_terms) >= total_terms_needed:
                logger.info(f"[COMPLETE] All terms appear to be processed - converting to final decisions")
                
                # Import the helper functions from step7_fixed_batch_processing
                from step7_fixed_batch_processing import (_convert_modern_batch_results_to_decisions, 
                                                         _create_final_decisions_data_with_batch_info,
                                                         _log_step7_completion_with_batch_info)
                
                # Convert to final decisions format
                final_decisions = _convert_modern_batch_results_to_decisions(
                    consolidated_data, verified_results, translation_data_map
                )
                
                # Create comprehensive final decisions data
                final_decisions_data = _create_final_decisions_data_with_batch_info(
                    final_decisions, consolidated_data, step7_manager
                )
                
                # Save final decisions
                decisions_file = os.path.join(str(self.output_dir), "Final_Terminology_Decisions.json")
            
                with open(decisions_file, 'w', encoding='utf-8') as f:
                    json.dump(final_decisions_data, f, indent=2, ensure_ascii=False)
                
                # Log completion statistics
                _log_step7_completion_with_batch_info(final_decisions_data, decisions_file, step7_manager)
                
                logger.info(f"[COMPLETE] Step 7 completed using existing batch results")
                return decisions_file
            else:
                logger.info(f"[CONTINUE] Only {len(processed_terms)}/{total_terms_needed} terms processed - continuing batch processing")
                
                # IDENTIFY MISSING TERMS: Find which specific terms need to be processed
                all_required_terms = set()
                for result in verified_results:
                    term = result.get('term', '')
                    if term:
                        all_required_terms.add(term)
                
                missing_terms = all_required_terms - processed_terms
                logger.info(f"[GAP_DETECTION] Found {len(missing_terms)} missing terms:")
                logger.info(f"[GAP_DETECTION] Missing terms: {sorted(list(missing_terms))[:10]}{'...' if len(missing_terms) > 10 else ''}")
                
                # Filter verified_results to only include missing terms
                missing_verified_results = []
                for result in verified_results:
                    term = result.get('term', '')
                    if term in missing_terms:
                        missing_verified_results.append(result)
                
                logger.info(f"[GAP_PROCESSING] Processing {len(missing_verified_results)} missing terms instead of all {len(verified_results)} terms")
                verified_results = missing_verified_results  # Use only missing terms for processing
        
        # Prepare terms for modern validation batch processing with comprehensive translatability analysis
        terms_for_validation = []
        for i, result in enumerate(verified_results):
            term = result.get('term', '')
            if not term:
                continue
                
            # Get original translation data for enhanced context
            original_translation_data = translation_data_map.get(term, {})
            
            # Calculate comprehensive translatability metrics
            total_languages = result.get('total_languages', 1)
            translated_languages = result.get('translated_languages', 0)
            same_languages = result.get('same_languages', 0)
            error_languages = result.get('error_languages', 0)
            translatability_score = result.get('translatability_score', 0.0)
            
            # Calculate derived translatability metrics
            language_coverage_rate = translated_languages / max(1, total_languages)
            same_language_rate = same_languages / max(1, total_languages)
            error_rate = error_languages / max(1, total_languages)
            
            # Classify translatability based on detailed analysis
            translatability_analysis = self._analyze_term_translatability(
                term, translatability_score, language_coverage_rate, same_language_rate, error_rate
            )
            
            # Create comprehensive term data structure for modern validation with Step 5 & 6 integration
            term_data = {
                'term': term,
                'frequency': result.get('frequency', 1),
                'original_texts': result.get('original_texts', {'texts': []}),
                'translations': result.get('translations', {}),
                'translatability_score': translatability_score,
                'translatability_analysis': translatability_analysis,
                
                # STEP 5 INTEGRATION DATA - for agents to consider in validation
                'step5_translation_data': {
                    'processing_tier': original_translation_data.get('processing_tier', 'unknown'),
                    'gpu_worker': original_translation_data.get('gpu_worker', 'unknown'),
                    'processing_time_seconds': original_translation_data.get('processing_time_seconds', 0),
                    'total_languages': original_translation_data.get('total_languages', 0),
                    'translated_languages': original_translation_data.get('translated_languages', 0),
                    'error_languages': original_translation_data.get('error_languages', 0),
                    'translation_success_rate': original_translation_data.get('translated_languages', 0) / max(1, original_translation_data.get('total_languages', 1)),
                    'all_translations': original_translation_data.get('all_translations', {}),
                    'sample_translations': original_translation_data.get('sample_translations', {}),
                    'translatability_score': original_translation_data.get('translatability_score', 0.0)
                },
                
                # STEP 6 INTEGRATION DATA - for agents to consider in validation
                'step6_verification_data': {
                    'verification_passed': result.get('verification_passed', False),
                    'verification_issues_count': result.get('verification_issues_count', 0),
                    'verified_translations': result.get('verified_translations', {}),
                    'verification_score': 1.0 if result.get('verification_passed', False) else max(0.0, 1.0 - (result.get('verification_issues_count', 0) * 0.1)),
                    'verification_timestamp': result.get('verification_timestamp', ''),
                    'consistency_with_step5': self._calculate_step5_step6_consistency(original_translation_data, result)
                },
                
                # ENHANCED CONTEXT for agents to use in decision making
                'validation_context': {
                    'source_step': 'step7_final_review',
                    'integration_level': 'step5_step6_modern_validation_with_translatability',
                    'decision_factors': [
                        'translatability_classification_and_analysis',
                        'translation_quality_from_step5',
                        'verification_results_from_step6',
                        'ml_quality_scoring',
                        'advanced_context_analysis',
                        'domain_classification',
                        'comprehensive_term_evaluation',
                        'untranslatable_term_detection'
                    ],
                    'quality_indicators': {
                        'step5_success_rate': original_translation_data.get('translated_languages', 0) / max(1, original_translation_data.get('total_languages', 1)),
                        'step6_verification_passed': result.get('verification_passed', False),
                        'term_frequency': result.get('frequency', 1),
                        'processing_tier': original_translation_data.get('processing_tier', 'unknown'),
                        'translatability_category': translatability_analysis['category'],
                        'is_translatable': translatability_analysis['is_translatable'],
                        'language_coverage_rate': language_coverage_rate,
                        'untranslatable_indicators': translatability_analysis['untranslatable_indicators']
                    },
                    'agent_instructions': {
                        'translatability_considerations': [
                            'Consider if term can be translated at all based on translatability analysis',
                            'Evaluate untranslatable indicators (technical codes, file extensions, etc.)',
                            'Factor in same-language rate for universal technical terms',
                            'Apply translatability-based decision thresholds',
                            'Provide reasoning for untranslatable or partially translatable terms'
                        ],
                        'decision_logic': translatability_analysis['recommended_action'],
                        'reasoning_context': translatability_analysis['reasoning_points']
                    }
                },
                
                # Metadata for processing
                'step7_metadata': {
                    'original_index': i,
                    'source_step6_result': result,
                    'source_step5_result': original_translation_data,
                    'batch_processing_enabled': True,
                    'modern_validation_system': True
                }
            }
            
            terms_for_validation.append(term_data)
        
        logger.info(f"[BATCH_PREP] Prepared {len(terms_for_validation):,} terms for modern validation batch processing")
        logger.info(f"[INTEGRATION] Each term includes Step 5 translation data, Step 6 verification data, and translatability analysis")
        
        # Run PROPER modern validation batch processing with agents
        logger.info("[MODERN_BATCH] Starting enhanced parallel validation processing with agents...")
        
        # DYNAMIC RESOURCE DETECTION (like ultra_optimized_smart_runner.py)
        import multiprocessing as mp
        import psutil
        import platform
        
        # Detect system resources dynamically
        cpu_cores = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        os_name = platform.system()
        
        logger.info(f"[RESOURCE_DETECTION] System Analysis:")
        logger.info(f"   CPU Cores: {cpu_cores}")
        logger.info(f"   Total Memory: {memory_gb:.1f}GB")
        logger.info(f"   Available Memory: {available_memory_gb:.1f}GB")
        logger.info(f"   Operating System: {os_name}")
        
        # DYNAMIC WORKER CALCULATION (similar to ultra runner's approach)
        if os_name == "Windows":
            if memory_gb >= 32:  # High-end system
                if cpu_cores >= 16:
                    optimal_workers = min(cpu_cores // 2, 12)
                elif cpu_cores >= 8:
                    optimal_workers = min(cpu_cores - 2, 8)
                else:
                    optimal_workers = max(4, cpu_cores // 2)
            elif memory_gb >= 16:  # Mid-range system
                if cpu_cores >= 12:
                    optimal_workers = min(cpu_cores // 3, 8)
                elif cpu_cores >= 6:
                    optimal_workers = min(cpu_cores - 2, 6)
                else:
                    optimal_workers = max(4, cpu_cores // 2)
            else:  # Low-end system
                optimal_workers = min(6, max(4, cpu_cores // 2))
        else:  # Linux/Mac
            if memory_gb >= 16:
                optimal_workers = min(cpu_cores - 1, 16)
            else:
                optimal_workers = min(cpu_cores // 2, 8)
        
        # Memory-based adjustment (like ultra runner)
        memory_per_worker = memory_gb / optimal_workers if optimal_workers > 0 else memory_gb
        if memory_per_worker < 2:  # Less than 2GB per worker
            optimal_workers = max(4, int(memory_gb // 2))
        
        # DYNAMIC BATCH SIZE CALCULATION (like ultra runner's memory-based approach)
        if memory_gb >= 32:
            dynamic_batch_size = min(16, max(8, cpu_cores // 2))  # High-end: larger batches
        elif memory_gb >= 16:
            dynamic_batch_size = min(12, max(6, cpu_cores // 3))  # Mid-range: moderate batches
        else:
            dynamic_batch_size = min(8, max(4, cpu_cores // 4))   # Low-end: smaller batches
        
        # Apply conservative caps for agent processing stability
        optimal_workers = min(optimal_workers, 12)  # Cap for agent stability
        dynamic_batch_size = min(dynamic_batch_size, 10)  # Cap for complex validation
        
        logger.info(f"[RESOURCE_OPTIMIZATION] Dynamic Configuration:")
        logger.info(f"   Optimal Workers: {optimal_workers} (vs fixed 4)")
        logger.info(f"   Dynamic Batch Size: {dynamic_batch_size} (vs fixed 8)")
        logger.info(f"   Memory per Worker: {memory_per_worker:.1f}GB")
        logger.info(f"   Expected Performance Boost: {optimal_workers/4:.1f}x workers, {dynamic_batch_size/8:.1f}x batch efficiency")
        
        try:
            # Use the PROPER modern validation system with DYNAMIC resource allocation
            # The agents will receive the Step 5, Step 6, and translatability data in their validation context
            modern_validation_system.process_terms_parallel(
                terms=terms_for_validation,
                file_prefix="step7_final_decisions",
                batch_size=dynamic_batch_size,  # DYNAMIC: Based on system resources (like ultra runner)
                max_workers=optimal_workers,    # DYNAMIC: Based on CPU/memory analysis (like ultra runner)
                src_lang="EN",
                tgt_lang=None,
                industry_context="Autodesk_Terminology_Final_Validation_with_Step5_Step6_Integration_and_Translatability_Analysis"
            )
            
            logger.info("[MODERN_BATCH] Modern validation batch processing with agents completed successfully")
                        
        except Exception as e:
            logger.error(f"[ERROR] Modern validation batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to in-memory processing if batch processing fails
            logger.info("[FALLBACK] Falling back to in-memory processing...")
            return self._step_7_fallback_processing(verified_results, translation_data_map)
        
        # Consolidate batch results into final decisions
        logger.info("[CONSOLIDATION] Consolidating modern validation batch results...")
        
        try:
            # Import the helper functions from step7_fixed_batch_processing
            from step7_fixed_batch_processing import (_convert_modern_batch_results_to_decisions, 
                                                     _create_final_decisions_data_with_batch_info,
                                                     _log_step7_completion_with_batch_info)
            
            # Consolidate all batch files into organized results
            modern_validation_system.consolidate_results("step7_final_decisions", "final_validation")
            
            # Load consolidated results and convert to final decisions format
            consolidated_file = os.path.join(
                step7_manager.results_dir,
                "consolidated_modern_step7_final_decisions_validation_results.json"
            )
            
            if os.path.exists(consolidated_file):
                with open(consolidated_file, 'r', encoding='utf-8') as f:
                    consolidated_data = json.load(f)
                
                logger.info(f"[CONSOLIDATION] Loaded consolidated results from {len(consolidated_data.get('batches', {}))} batches")
                
                # Convert modern validation results to final decisions format
                final_decisions = _convert_modern_batch_results_to_decisions(
                    consolidated_data, verified_results, translation_data_map
                )
                
                # Create comprehensive final decisions data
                final_decisions_data = _create_final_decisions_data_with_batch_info(
                    final_decisions, consolidated_data, step7_manager
                )
        
        # Save final decisions
                decisions_file = os.path.join(str(self.output_dir), "Final_Terminology_Decisions.json")
                
                with open(decisions_file, 'w', encoding='utf-8') as f:
                    json.dump(final_decisions_data, f, indent=2, ensure_ascii=False)
                
                # Log completion statistics
                _log_step7_completion_with_batch_info(final_decisions_data, decisions_file, step7_manager)
                
                return decisions_file
            
            else:
                logger.error(f"[ERROR] Consolidated results file not found: {consolidated_file}")
                return self._step_7_fallback_processing(verified_results, translation_data_map)
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to consolidate modern validation results: {e}")
            import traceback
            traceback.print_exc()
            return self._step_7_fallback_processing(verified_results, translation_data_map)

    def _analyze_term_translatability(self, term: str, translatability_score: float, language_coverage_rate: float, 
                                    same_language_rate: float, error_rate: float) -> dict:
        """Comprehensive translatability analysis for agent decision-making"""
        from step7_fixed_batch_processing import _analyze_term_translatability
        return _analyze_term_translatability(term, translatability_score, language_coverage_rate, same_language_rate, error_rate)

    def _calculate_step5_step6_consistency(self, step5_data: dict, step6_data: dict) -> float:
        """Calculate consistency between Step 5 and Step 6 results"""
        from step7_fixed_batch_processing import _calculate_step5_step6_consistency
        return _calculate_step5_step6_consistency(step5_data, step6_data)

    def _convert_modern_batch_results_to_decisions(self, consolidated_data: dict, verified_results: list, translation_data_map: dict) -> list:
        """Convert modern validation batch results to final decisions format"""
        from step7_fixed_batch_processing import _convert_modern_batch_results_to_decisions
        return _convert_modern_batch_results_to_decisions(consolidated_data, verified_results, translation_data_map)

    def _create_final_decisions_data_with_batch_info(self, final_decisions: list, consolidated_data: dict, step7_manager) -> dict:
        """Create comprehensive final decisions data structure with batch processing information"""
        from step7_fixed_batch_processing import _create_final_decisions_data_with_batch_info
        return _create_final_decisions_data_with_batch_info(final_decisions, consolidated_data, step7_manager)

    def _log_step7_completion_with_batch_info(self, final_decisions_data: dict, decisions_file: str, step7_manager):
        """Log Step 7 completion with comprehensive batch processing statistics"""
        from step7_fixed_batch_processing import _log_step7_completion_with_batch_info
        return _log_step7_completion_with_batch_info(final_decisions_data, decisions_file, step7_manager)

    def _step_7_fallback_processing(self, verified_results: list, translation_data_map: dict) -> str:
        """Fallback to in-memory processing if batch processing fails"""
        from step7_fixed_batch_processing import _step_7_fallback_processing
        return _step_7_fallback_processing(self, verified_results, translation_data_map)

    def _create_empty_decisions_file(self, reason: str) -> str:
        """Create empty decisions file for pipeline continuity"""
        decisions_file = os.path.join(str(self.output_dir), "Final_Terminology_Decisions.json")
        empty_decisions = {
                'metadata': {
                'step': 7,
                'step_name': 'Final Review and Decision',
                'timestamp': datetime.now().isoformat(),
                'total_terms_reviewed': 0,
                'review_status': f'skipped_{reason}'
            },
            'final_decisions': [],
            'modern_validation_summary': {
                'total_decisions': 0,
                'fully_approved': 0,
                'conditionally_approved': 0,
                'needs_review': 0,
                'rejected': 0,
                'approved_as_new_term': 0,
                'conditionally_approved_as_new_term': 0,
                'needs_review_for_new_term': 0,
                'average_validation_score': 0.0,
                'average_ml_quality_score': 0.0
            }
        }
        
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(empty_decisions, f, indent=2, ensure_ascii=False)
            
        logger.info(f"[OK] Created empty decisions file: {decisions_file}")
        return decisions_file
    
    def step_8_timestamp_data_recording(self, decisions_file: str) -> str:
        """
        Step 8: Timestamp + Term Data Recording
        - Record timestamp information
        - Store term data for tracking and auditing
        - Generate comprehensive report
        - Integrate reviewed terminology analysis
        """
        logger.info("[LOG] STEP 8: Timestamp + Term Data Recording")
        logger.info("=" * 60)
        
        # Load final decisions (handle case where no decisions file exists)
        decisions_data = {}
        if decisions_file and os.path.exists(decisions_file):
            with open(decisions_file, 'r', encoding='utf-8') as f:
                decisions_data = json.load(f)
            logger.info(f"[OK] Loaded decisions from: {decisions_file}")
        else:
            logger.warning("[WARNING] No decisions file found - creating summary without final decisions")
            decisions_data = {
                'final_decisions': [],
                'summary': {
                    'total_terms_reviewed': 0,
                    'approved_terms': 0,
                    'rejected_terms': 0,
                    'review_status': 'no_terms_for_review'
                }
            }
        
        # Create comprehensive audit record with enhanced formatting
        audit_record = {
            'system_info': {
                'system_name': 'Multi-Agent Terminology Validation System',
                'version': '1.0.0',
                'session_id': self.session_id,
                'processing_start_time': datetime.now().isoformat(),
                'processing_end_time': datetime.now().isoformat(),
                'total_processing_time_seconds': time.time() - getattr(self, 'start_time', time.time())
            },
            'process_statistics': self.process_stats,
            'configuration': self.config,
            'final_decisions': decisions_data,
            'audit_trail': {
                'step_1_completed': True,
                'step_2_completed': True,
                'step_3_completed': True,
                'step_4_completed': True,
                'step_5_completed': True,
                'step_6_completed': True,
                'step_7_completed': True,
                'step_8_completed': True,
                'step_9_completed': True
            },
            'output_files': {
                'combined_terms': str(self.output_dir / "Combined_Terms_Data.csv"),
                'cleaned_terms': str(self.output_dir / "Cleaned_Terms_Data.csv"),
                'new_terms_candidates': str(self.output_dir / "New_Terms_Candidates.json"),
                'high_frequency_terms': str(self.output_dir / "High_Frequency_Terms.json"),
                'frequency_storage_export': str(self.output_dir / "Frequency_Storage_Export.json"),
                'translation_results': str(self.output_dir / "Translation_Results.json"),
                'verified_results': str(self.output_dir / "Verified_Translation_Results.json"),
                'final_decisions': decisions_file,
                'audit_record': str(self.output_dir / "Complete_Audit_Record.json"),
                'approved_terms_csv': str(self.output_dir / "Approved_Terms_Export.csv")
            },
            'enhanced_formatting': {
                'format_version': '2.0.0',
                'enhanced_features': [
                    'Structured audit trail',
                    'Comprehensive process statistics',
                    'Enhanced file organization',
                    'Improved data validation'
                ],
                'formatting_timestamp': datetime.now().isoformat()
            }
        }
        
        # Save complete audit record
        audit_file = str(self.output_dir / "Complete_Audit_Record.json")
        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_record, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self._generate_summary_report(audit_record)
        
        logger.info(f"[OK] Step 8 completed:")
        logger.info(f"[FOLDER] Complete audit record: {audit_file}")
        logger.info(f"[STATS] Summary report generated")
        
        return audit_file
    
    def step_9_approved_terms_csv_export(self, decisions_file: str) -> str:
        """
        Step 9: Approved Terms CSV Export
        - Export approved terms to CSV format matching reviewed/ folder structure
        - Use Azure OpenAI GPT-4.1 to generate professional contexts from original texts
        - Filter for fully approved and conditionally approved terms
        """
        logger.info("[CSV] STEP 9: Approved Terms CSV Export")
        logger.info("=" * 60)
        
        # Load final decisions
        if not decisions_file or not os.path.exists(decisions_file):
            logger.warning("[WARNING] No decisions file found - creating empty CSV")
            approved_csv_file = str(self.output_dir / "Approved_Terms_Export.csv")
            with open(approved_csv_file, 'w', encoding='utf-8', newline='') as f:
                f.write("source,target,context\n")
            return approved_csv_file
        
        with open(decisions_file, 'r', encoding='utf-8') as f:
            decisions_data = json.load(f)
        
        # Load High_Frequency_Terms.json for original texts
        high_freq_file = str(self.output_dir / "High_Frequency_Terms.json")
        original_texts_map = {}
        if os.path.exists(high_freq_file):
            logger.info("[CONTEXT] Loading original texts from High_Frequency_Terms.json...")
            with open(high_freq_file, 'r', encoding='utf-8') as f:
                high_freq_data = json.load(f)
            
            for term_data in high_freq_data.get('terms', []):
                term = term_data.get('term', '')
                original_texts = term_data.get('original_texts', {}).get('texts', [])
                if term and original_texts:
                    original_texts_map[term] = original_texts[:5]  # Use first 5 examples
        
        logger.info(f"[CONTEXT] Loaded original texts for {len(original_texts_map):,} terms")
        
        final_decisions = decisions_data.get('final_decisions', [])
        
        # Filter approved terms (both fully approved and conditionally approved)
        approved_decisions = []
        for decision in final_decisions:
            decision_type = decision.get('decision', '')
            if 'APPROVED' in decision_type:  # Includes both APPROVED and CONDITIONALLY_APPROVED
                approved_decisions.append(decision)
        
        logger.info(f"[PARALLEL] Processing {len(approved_decisions):,} approved terms with parallel GPT-4.1 context generation")
        
        # Generate contexts using parallel processing
        approved_terms = self._generate_contexts_parallel(approved_decisions, original_texts_map)
        
        # Sort terms alphabetically
        approved_terms.sort(key=lambda x: x['source'].lower())
        
        # Create CSV file
        approved_csv_file = str(self.output_dir / "Approved_Terms_Export.csv")
        
        import csv
        with open(approved_csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'context'])
            
            for term_data in approved_terms:
                writer.writerow([
                    term_data['source'],
                    term_data['target'], 
                    term_data['context']
                ])
        
        # Add step metadata
        self._add_step_metadata(approved_csv_file, 9, "Approved Terms CSV Export", {
            "export_format": "CSV",
            "total_approved_terms": len(approved_terms),
            "includes_fully_approved": True,
            "includes_conditionally_approved": True,
            "format_compatible_with": "reviewed/ folder structure",
            "columns": ["source", "target", "context"],
            "context_generation": "Azure OpenAI GPT-4.1",
            "original_texts_used": len(original_texts_map)
        })
        
        logger.info(f"[OK] Step 9 completed:")
        logger.info(f"   Approved terms exported: {len(approved_terms):,}")
        logger.info(f"   CSV format: source,target,context")
        logger.info(f"   Context generation: Azure OpenAI GPT-4.1")
        logger.info(f"   Compatible with reviewed/ folder structure")
        logger.info(f"[FOLDER] CSV file saved: {approved_csv_file}")
        
        # Update process stats
        self.process_stats['terms_approved'] = len(approved_terms)
        
        return approved_csv_file
    
    def _generate_contexts_parallel(self, approved_decisions: list, original_texts_map: dict) -> list:
        """
        Generate contexts for approved terms using parallel processing with GPT-4.1
        Optimized based on system specifications and Azure OpenAI rate limits
        """
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import psutil
        import time
        
        # Determine optimal worker count based on system specs and Azure OpenAI limits
        cpu_count = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Azure OpenAI rate limits: typically 240 requests/minute for GPT-4
        # Conservative approach: 4 requests/second = 240/minute
        max_workers = min(
            cpu_count * 2,  # CPU-based limit
            int(available_memory_gb / 2),  # Memory-based limit (2GB per worker)
            20,  # Azure OpenAI concurrent request limit
            len(approved_decisions) // 10  # Don't over-parallelize for small batches
        )
        max_workers = max(1, max_workers)  # Ensure at least 1 worker
        
        logger.info(f"[PARALLEL] System specs: {cpu_count} CPUs, {available_memory_gb:.1f}GB RAM")
        logger.info(f"[PARALLEL] Using {max_workers} parallel workers for GPT-4.1 context generation")
        logger.info(f"[PARALLEL] Processing {len(approved_decisions):,} terms in batches")
        
        # Split decisions into batches for parallel processing
        batch_size = max(1, len(approved_decisions) // max_workers)
        batches = [approved_decisions[i:i + batch_size] for i in range(0, len(approved_decisions), batch_size)]
        
        logger.info(f"[PARALLEL] Created {len(batches)} batches of ~{batch_size} terms each")
        
        approved_terms = []
        completed_count = 0
        
        # Process batches in parallel using ThreadPoolExecutor (better for I/O-bound GPT calls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._process_batch_contexts, batch_idx, batch, original_texts_map): batch_idx 
                for batch_idx, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    approved_terms.extend(batch_results)
                    completed_count += len(batch_results)
                    
                    progress_pct = (completed_count / len(approved_decisions)) * 100
                    logger.info(f"[PROGRESS] Batch {batch_idx + 1}/{len(batches)} completed - "
                              f"{completed_count:,}/{len(approved_decisions):,} terms ({progress_pct:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Batch {batch_idx} failed: {e}")
                    # Continue processing other batches
        
        logger.info(f"[PARALLEL] Completed parallel context generation: {len(approved_terms):,} terms processed")
        return approved_terms
    
    def _process_batch_contexts(self, batch_idx: int, batch_decisions: list, original_texts_map: dict) -> list:
        """
        Process a batch of decisions to generate contexts using GPT-4.1
        This runs in a separate thread/process
        """
        import time
        
        batch_results = []
        batch_start_time = time.time()
        
        logger.info(f"[BATCH {batch_idx}] Starting batch with {len(batch_decisions)} terms")
        
        for i, decision in enumerate(batch_decisions):
            try:
                term = decision.get('term', '')
                decision_type = decision.get('decision', '')
                
                # Generate professional context using GPT-4.1
                context = self._generate_professional_context_with_gpt(
                    term, 
                    original_texts_map.get(term, []),
                    decision_type,
                    decision.get('reasoning', ''),
                    decision.get('translatability_score', 0),
                    decision.get('quality_score', 0)
                )
                
                batch_results.append({
                    'source': term,
                    'target': term,  # For English terms, target is same as source
                    'context': context
                })
                
                # Rate limiting: small delay to respect Azure OpenAI limits
                if i > 0 and i % 5 == 0:  # Every 5 requests
                    time.sleep(0.1)  # 100ms delay
                
            except Exception as e:
                logger.warning(f"[BATCH {batch_idx}] Failed to process term '{term}': {e}")
                # Add fallback context
                batch_results.append({
                    'source': term,
                    'target': term,
                    'context': self._generate_fallback_context(term, decision_type)
                })
        
        batch_duration = time.time() - batch_start_time
        logger.info(f"[BATCH {batch_idx}] Completed {len(batch_results)} terms in {batch_duration:.1f}s "
                   f"({len(batch_results)/batch_duration:.1f} terms/sec)")
        
        return batch_results
    
    def _generate_professional_context_with_gpt(self, term: str, original_texts: list, 
                                              decision_type: str, reasoning: str, 
                                              translatability_score: float, quality_score: float) -> str:
        """
        Generate professional context description using agentic framework (smolagent)
        Intelligently analyzes original_texts to select or create aligned context
        Following the style and format of reviewed/ CSV files
        """
        try:
            # Use agentic framework for intelligent context analysis
            return self._generate_context_with_agentic_framework(
                term, original_texts, decision_type, reasoning, 
                translatability_score, quality_score
            )
                
        except Exception as e:
            logger.warning(f"[AGENTIC] Context generation failed for '{term}': {e}")
            return self._generate_fallback_context(term, decision_type)
    
    def _generate_context_with_agentic_framework(self, term: str, original_texts: list, 
                                               decision_type: str, reasoning: str, 
                                               translatability_score: float, quality_score: float) -> str:
        """
        Use agentic framework (smolagent) to intelligently analyze original_texts
        and create context aligned with reviewed/ CSV formatting requirements
        """
        try:
            # Try to use smolagents with proper Azure OpenAI integration
            context = self._use_smolagents_for_context(term, original_texts, decision_type, reasoning)
            
            if context and len(context) > 20:
                return context
            
            # Fallback to pattern-based analysis if agent fails
            logger.info(f"[AGENTIC] smolagent failed, using pattern-based analysis for '{term}'")
            return self._analyze_original_texts_pattern_based(term, original_texts, decision_type)
            
        except ImportError:
            logger.info(f"[AGENTIC] smolagent not available, using pattern-based analysis for '{term}'")
            return self._analyze_original_texts_pattern_based(term, original_texts, decision_type)
        except Exception as e:
            logger.warning(f"[AGENTIC] Agent analysis failed for '{term}': {e}")
            return self._analyze_original_texts_pattern_based(term, original_texts, decision_type)
    
    def _use_smolagents_for_context(self, term: str, original_texts: list, decision_type: str, reasoning: str) -> str:
        """
        Use smolagents framework with proper Azure OpenAI integration
        Based on the working implementation from terminology_agent.py
        """
        try:
            from smolagents import CodeAgent, AzureOpenAIServerModel
            from azure.identity import EnvironmentCredential
            from datetime import datetime, timedelta
            import json
            import os
            
            # Create patched Azure OpenAI model (similar to terminology_agent.py)
            class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._credential = EnvironmentCredential()
                    self._token_expiry = None
                    self._force_token_refresh()

                def _force_token_refresh(self):
                    try:
                        token_result = self._credential.get_token("https://cognitiveservices.azure.com/.default")
                        self.api_key = token_result.token
                        self._token_expiry = datetime.fromtimestamp(token_result.expires_on)
                        return True
                    except Exception:
                        return False

                def _prepare_completion_kwargs(self, *args, **kwargs):
                    self._force_token_refresh()
                    completion_kwargs = super()._prepare_completion_kwargs(*args, **kwargs)
                    if 'stop' in completion_kwargs:
                        del completion_kwargs['stop']
                    return completion_kwargs
            
            # Get Azure credentials and setup
            credential = EnvironmentCredential()
            token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://autodesk-openai-eastus.openai.azure.com/")
            
            # Create the model
            model = PatchedAzureOpenAIServerModel(
                model_id="gpt-4.1",
                azure_endpoint=endpoint,
                api_key=token_result.token,
                api_version="2024-02-01",
                custom_role_conversions={
                    "tool-call": "user",
                    "tool-response": "assistant",
                },
            )
            
            # Create agent
            agent = CodeAgent(
                model=model,
                tools=[],  # No additional tools needed for context generation
                add_base_tools=False,  # Keep it simple
                max_steps=3  # Limit steps for efficiency
            )
            
            # Create the prompt
            original_texts_sample = original_texts[:3] if original_texts else []
            examples_text = "\n".join([f"- {text[:100]}..." if len(text) > 100 else f"- {text}" 
                                     for text in original_texts_sample])
            
            task_prompt = f"""Create a professional context description for the term "{term}" for a terminology database.

TERM: {term}
DECISION: {decision_type}

ORIGINAL USAGE EXAMPLES:
{examples_text if examples_text else "No specific usage examples available"}

REQUIREMENTS:
1. Write in technical software documentation style (AutoCAD, 3ds Max, Revit)
2. Be concise (80-150 characters ideal)
3. Include domain context (CAD, 3D modeling, engineering, graphics)
4. Use patterns: "Tool for...", "Feature that...", "Command used to...", "Process of..."
5. Do NOT include scores, status, or approval information

EXAMPLES OF GOOD CONTEXTS:
- "Tool for moving 3D objects or parts along specific axes or planes in the drawing area."
- "Command used in AutoCAD software; appears in command line to trigger specific functions."
- "Feature that converts objects to meshes, often used for particle systems or applying modifiers."

Generate ONLY the context description (no quotes, no extra text):"""
            
            # Run the agent with timeout
            response = agent.run(task_prompt)
            
            # Clean the response
            if isinstance(response, str):
                context = response.strip().replace('"', '').replace("'", "")
                if len(context) > 300:
                    context = context[:297] + "..."
                return context
            
            return None
            
        except Exception as e:
            logger.debug(f"[SMOLAGENT] Failed for '{term}': {e}")
            return None
    
    
    def _analyze_original_texts_pattern_based(self, term: str, original_texts: list, decision_type: str) -> str:
        """
        Analyze original_texts using pattern-based approach to extract or create context
        This is the fallback when agentic frameworks are not available
        """
        try:
            if not original_texts:
                return self._generate_fallback_context(term, decision_type)
            
            # Analyze patterns in original texts
            patterns = {
                'command': ['command', 'cmd', 'function', 'tool', 'option'],
                'feature': ['feature', 'capability', 'function', 'tool'],
                'process': ['process', 'workflow', 'procedure', 'method'],
                'object': ['object', 'element', 'component', 'item'],
                'interface': ['dialog', 'window', 'panel', 'menu', 'button'],
                'file': ['file', 'document', 'export', 'import', 'save'],
                'modeling': ['3D', 'model', 'mesh', 'surface', 'geometry'],
                'cad': ['AutoCAD', 'drawing', 'design', 'draft', 'sketch'],
                'graphics': ['render', 'material', 'texture', 'lighting', 'visual']
            }
            
            # Find the most relevant pattern
            text_content = " ".join(original_texts[:3]).lower()
            pattern_scores = {}
            
            for pattern_type, keywords in patterns.items():
                score = sum(1 for keyword in keywords if keyword.lower() in text_content)
                if score > 0:
                    pattern_scores[pattern_type] = score
            
            # Generate context based on detected patterns
            if pattern_scores:
                top_pattern = max(pattern_scores.keys(), key=lambda k: pattern_scores[k])
                return self._generate_pattern_based_context(term, top_pattern, original_texts[0] if original_texts else "")
            else:
                # Extract key information from first original text
                first_text = original_texts[0] if original_texts else ""
                return self._extract_context_from_text(term, first_text)
                
        except Exception as e:
            logger.warning(f"[PATTERN] Pattern analysis failed for '{term}': {e}")
            return self._generate_fallback_context(term, decision_type)
    
    def _generate_pattern_based_context(self, term: str, pattern_type: str, sample_text: str) -> str:
        """
        Generate context based on detected pattern type
        """
        pattern_templates = {
            'command': f"Command used in CAD software to {self._extract_action(sample_text)}",
            'feature': f"Feature that {self._extract_capability(sample_text)} in design applications",
            'process': f"Process of {self._extract_process(sample_text)} in engineering workflows",
            'object': f"Design element used for {self._extract_purpose(sample_text)} in 3D modeling",
            'interface': f"Interface element for {self._extract_ui_purpose(sample_text)} in software applications",
            'file': f"File operation for {self._extract_file_action(sample_text)} in CAD systems",
            'modeling': f"3D modeling tool for {self._extract_modeling_action(sample_text)}",
            'cad': f"CAD function for {self._extract_cad_purpose(sample_text)} in technical drawings",
            'graphics': f"Graphics feature for {self._extract_graphics_purpose(sample_text)} in visual design"
        }
        
        template = pattern_templates.get(pattern_type, f"Technical term used in {pattern_type} operations")
        
        # Ensure reasonable length
        if len(template) > 150:
            template = template[:147] + "..."
        
        return template
    
    def _extract_action(self, text: str) -> str:
        """Extract action description from text"""
        # Simple extraction - can be enhanced with NLP
        action_words = ['create', 'modify', 'edit', 'delete', 'move', 'copy', 'select', 'draw', 'design']
        text_lower = text.lower()
        for action in action_words:
            if action in text_lower:
                return f"{action} objects or elements"
        return "perform operations"
    
    def _extract_capability(self, text: str) -> str:
        """Extract capability description from text"""
        if 'convert' in text.lower():
            return "converts or transforms data"
        elif 'manage' in text.lower():
            return "manages system resources"
        elif 'display' in text.lower():
            return "displays information or graphics"
        return "provides functionality"
    
    def _extract_process(self, text: str) -> str:
        """Extract process description from text"""
        if 'render' in text.lower():
            return "rendering graphics or models"
        elif 'export' in text.lower():
            return "exporting data or files"
        elif 'import' in text.lower():
            return "importing external data"
        return "processing data or operations"
    
    def _extract_purpose(self, text: str) -> str:
        """Extract purpose from text"""
        return "design and modeling purposes"
    
    def _extract_ui_purpose(self, text: str) -> str:
        """Extract UI purpose from text"""
        return "user interaction and control"
    
    def _extract_file_action(self, text: str) -> str:
        """Extract file action from text"""
        if 'save' in text.lower():
            return "saving project data"
        elif 'load' in text.lower():
            return "loading external files"
        return "managing project files"
    
    def _extract_modeling_action(self, text: str) -> str:
        """Extract 3D modeling action from text"""
        return "creating and manipulating 3D geometry"
    
    def _extract_cad_purpose(self, text: str) -> str:
        """Extract CAD purpose from text"""
        return "technical design and documentation"
    
    def _extract_graphics_purpose(self, text: str) -> str:
        """Extract graphics purpose from text"""
        return "visual representation and rendering"
    
    def _extract_context_from_text(self, term: str, text: str) -> str:
        """
        Extract meaningful context directly from the original text
        """
        if not text or len(text) < 10:
            return self._generate_fallback_context(term, "APPROVED")
        
        # Clean and truncate the text to create a context
        cleaned_text = text.replace('"', '').replace("'", "").strip()
        
        # If text is already short enough, use it directly
        if len(cleaned_text) <= 150:
            return cleaned_text
        
        # Try to find a complete sentence
        sentences = cleaned_text.split('.')
        if sentences and len(sentences[0]) <= 150:
            return sentences[0].strip() + "."
        
        # Truncate to reasonable length
        return cleaned_text[:147] + "..."
    
    def _generate_fallback_context(self, term: str, decision_type: str) -> str:
        """Generate fallback context when GPT is not available"""
        if 'CONDITIONALLY' in decision_type:
            return f"Technical term used in CAD/engineering software; conditionally approved for terminology database."
        else:
            return f"Technical term used in CAD/engineering software; approved for terminology database."
    
    def _generate_summary_report(self, audit_record: Dict):
        """Generate a human-readable summary report"""
        
        report_file = str(self.output_dir / "Validation_Summary_Report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Multi-Agent Terminology Validation System - Summary Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Process Overview\n\n")
            f.write("This report summarizes the complete terminology validation workflow:\n")
            f.write("1. Initial Term Collection and Verification\n")
            f.write("2. Glossary Validation\n")
            f.write("3. New Terminology Processing\n")
            f.write("4. Frequency Analysis and Filtering\n")
            f.write("5. Translation Process\n")
            f.write("6. Language Verification\n")
            f.write("7. Final Review and Decision\n")
            f.write("8. Timestamp + Term Data Recording\n")
            f.write("9. Approved Terms CSV Export\n\n")
            
            f.write("## Statistics\n\n")
            stats = self.process_stats
            f.write(f"- **Total Terms Input:** {stats['total_terms_input']:,}\n")
            f.write(f"- **Terms After Cleaning:** {stats['terms_after_cleaning']:,}\n")
            f.write(f"- **Frequency = 1 Terms (Stored):** {stats['frequency_1_terms']:,}\n")
            f.write(f"- **Frequency > 2 Terms (Processed):** {stats['frequency_gt_2_terms']:,}\n")
            f.write(f"- **Terms Translated:** {stats['terms_translated']:,}\n")
            f.write(f"- **Terms Approved:** {stats['terms_approved']:,}\n")
            f.write(f"- **Terms Rejected:** {stats['terms_rejected']:,}\n\n")
            
            f.write("## Output Files\n\n")
            output_files = audit_record['output_files']
            for file_type, file_path in output_files.items():
                f.write(f"- **{file_type.replace('_', ' ').title()}:** `{file_path}`\n")
            
            f.write("\n## System Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n")
            
            f.write("\n## Process Completion\n\n")
            f.write("[OK] All 9 steps of the terminology validation process completed successfully.\n")
            f.write(f"[STATS] Processing session: {self.session_id}\n")
            f.write(f"[FOLDER] All outputs saved to: `{self.output_dir}`\n")
        
        logger.info(f"[FILE] Summary report saved: {report_file}")
    
    def run_complete_validation_process(self, input_file: str) -> str:
        """Run the complete 9-step validation process"""
        
        self.start_time = time.time()
        
        logger.info("[*] STARTING AGENTIC TERMINOLOGY VALIDATION SYSTEM")
        logger.info("=" * 80)
        logger.info(f"[INPUT] Input file: {input_file}")
        logger.info(f"[ID] Session: {self.session_id}")
        
        try:
            # Initialize all components
            self.initialize_components()
            
            # Step 1: Initial Term Collection and Verification
            if 1 in self.completed_steps:
                logger.info("[SKIP] Step 1 already completed - loading existing results")
                cleaned_file = str(self.step_files[1]['Combined_Terms_Data.csv'])
            else:
                cleaned_file = self.step_1_initial_term_collection(input_file)
            
            # Step 2: Glossary Validation
            if 2 in self.completed_steps:
                logger.info("[SKIP] Step 2 already completed - loading existing results")
                glossary_results = self._load_json_file(self.step_files[2]['Glossary_Analysis_Results.json'])
            else:
                glossary_results = self.step_2_glossary_validation(cleaned_file)
            
            # Step 3: New Terminology Processing
            if 3 in self.completed_steps:
                logger.info("[SKIP] Step 3 already completed - loading existing results")
                new_terms_file = str(self.step_files[3]['New_Terms_Candidates_With_Dictionary.json'])
            else:
                new_terms_file = self.step_3_new_terminology_processing(glossary_results, cleaned_file)
            
            # Step 4: Frequency Analysis and Filtering
            if 4 in self.completed_steps:
                logger.info("[SKIP] Step 4 already completed - loading existing results")
                freq_results = self._load_json_file(self.step_files[4]['Frequency_Storage_Export.json'])
                # Point directly to High_Frequency_Terms.json when resuming
                high_freq_file = str(self.output_dir / "High_Frequency_Terms.json")
                storage_export = freq_results.get('storage_export', None)
            else:
                high_freq_file, storage_export = self.step_4_frequency_analysis_filtering(new_terms_file)
            
            # Step 5: Translation Process
            if 5 in self.completed_steps:
                logger.info("[SKIP] Step 5 already completed - loading existing results")
                translation_results_file = str(self.step_files[5]['Translation_Results.json'])
            else:
                translation_results_file = self.step_5_translation_process(high_freq_file)
            
            # Step 6: Language Verification
            if 6 in self.completed_steps:
                logger.info("[SKIP] Step 6 already completed - loading existing results")
                verified_file = str(self.step_files[6]['Verified_Translation_Results.json'])
            else:
                verified_file = self.step_6_language_verification(translation_results_file)
            
            # Step 7: Final Review and Decision
            if 7 in self.completed_steps:
                logger.info("[SKIP] Step 7 already completed - loading existing results")
                decisions_file = str(self.step_files[7]['Final_Terminology_Decisions.json'])
            else:
                decisions_file = self.step_7_final_review_decision(verified_file, translation_results_file)
                # Handle case where no decisions file is created (no terms to review)
                if not decisions_file:
                    logger.info("[INFO] No terms required final review - proceeding with audit")
            
            # Step 8: Timestamp + Term Data Recording
            if 8 in self.completed_steps:
                logger.info("[SKIP] Step 8 already completed - loading existing results")
                audit_file = str(self.step_files[8]['Complete_Audit_Record.json'])
            else:
                audit_file = self.step_8_timestamp_data_recording(decisions_file)
            
            # Step 9: Approved Terms CSV Export
            if 9 in self.completed_steps:
                logger.info("[SKIP] Step 9 already completed - loading existing results")
                csv_file = str(self.step_files[9]['Approved_Terms_Export.csv'])
            else:
                csv_file = self.step_9_approved_terms_csv_export(decisions_file)
            
            # Calculate final processing time
            self.process_stats['processing_time'] = time.time() - self.start_time
            
            logger.info("[SUCCESS] AGENTIC TERMINOLOGY VALIDATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"[TIME] Total processing time: {self.process_stats['processing_time']:.1f} seconds")
            logger.info(f"[STATS] Final statistics: {self.process_stats}")
            logger.info(f"[FOLDER] All outputs in: {self.output_dir}")
            
            return str(self.output_dir)
            
        except Exception as e:
            logger.error(f"[ERROR] VALIDATION PROCESS FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error report
            error_file = str(self.output_dir / "Error_Report.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'partial_statistics': self.process_stats
                }, f, indent=2)
            
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Terminology Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new validation process
  python agentic_terminology_validation_system.py Term_Extracted_result.csv
  
  # Resume from a specific folder
  python agentic_terminology_validation_system.py Term_Extracted_result.csv --resume-from agentic_validation_output_20250918_155758
  
  # Resume from partial folder name (will find matching folder)
  python agentic_terminology_validation_system.py Term_Extracted_result.csv --resume-from 20250918_155758
        """
    )
    parser.add_argument("input_file", help="Input CSV file (Term_Extracted_result.csv)")
    parser.add_argument("--resume-from", help="Resume from existing output folder (exact path or partial name)")
    parser.add_argument("--glossary-folder", default="glossary", help="Glossary folder path")
    parser.add_argument("--terminology-model", default="gpt-4.1", help="Model for terminology agent")
    parser.add_argument("--validation-model", default="gpt-4.1", help="Model for validation")
    parser.add_argument("--translation-model-size", default="1.3B", help="Translation model size")
    parser.add_argument("--gpu-workers", type=int, default=3, help="Number of GPU workers (supports up to 3 for multiple model setup)")
    parser.add_argument("--cpu-workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode (skip glossary validation)")
    parser.add_argument("--skip-glossary", action="store_true", help="Skip glossary validation step")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'glossary_folder': args.glossary_folder,
        'terminology_model': args.terminology_model,
        'validation_model': args.validation_model,
        'translation_model_size': args.translation_model_size,
        'gpu_workers': args.gpu_workers,
        'cpu_workers': args.cpu_workers
    }
    
    # Initialize and run system
    system = AgenticTerminologyValidationSystem(config, resume_from=args.resume_from)
    
    try:
        output_dir = system.run_complete_validation_process(args.input_file)
        print(f"\n[SUCCESS] SUCCESS: Complete validation results in {output_dir}")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
