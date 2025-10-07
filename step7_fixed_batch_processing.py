#!/usr/bin/env python3
"""
Fixed Step 7 implementation with PROPER modern validation batch processing
This replaces the current Step 7 in agentic_terminology_validation_system.py
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def step_7_final_review_decision_fixed(self, verified_file: str, translation_results_file: str = None) -> str:
    """
    Step 7: Final Review and Decision with PROPER Modern Validation Batch Processing
    
    This step uses the COMPLETE modern parallel validation system with:
    1. PROPER batch processing via EnhancedValidationSystem.process_terms_parallel()
    2. Organized folder structure with batch files, logs, and cache
    3. Modern terminology review agents processing each batch
    4. ML-based quality scoring and advanced context analysis in batch workers
    5. Integration of Step 5 (translation) and Step 6 (verification) data
    6. Comprehensive validation workflow with error handling and resumption
    7. Consolidation of batch results into final decisions
    
    Args:
        verified_file: Path to Step 6 verification results
        translation_results_file: Optional path to Step 5 translation results for enhanced evaluation
        
    Returns:
        Path to final decisions file
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
    
    # Prepare terms for modern validation batch processing
    # Convert Step 6 verified results to format expected by modern validation system
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
        translatability_analysis = _analyze_term_translatability(
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
                'consistency_with_step5': _calculate_step5_step6_consistency(original_translation_data, result)
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
    logger.info(f"[INTEGRATION] Each term includes Step 5 translation data and Step 6 verification data")
    
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
        # The agents will receive the Step 5 and Step 6 data in their validation context
        modern_validation_system.process_terms_parallel(
            terms=terms_for_validation,
            file_prefix="step7_final_decisions",
            batch_size=dynamic_batch_size,  # DYNAMIC: Based on system resources (like ultra runner)
            max_workers=optimal_workers,    # DYNAMIC: Based on CPU/memory analysis (like ultra runner)
            src_lang="EN",
            tgt_lang=None,
            industry_context="Autodesk_Terminology_Final_Validation_with_Step5_Step6_Integration"
        )
        
        logger.info("[MODERN_BATCH] Modern validation batch processing with agents completed successfully")
        
    except Exception as e:
        logger.error(f"[ERROR] Modern validation batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to in-memory processing if batch processing fails
        logger.info("[FALLBACK] Falling back to in-memory processing...")
        return _step_7_fallback_processing(self, verified_results, translation_data_map)
    
    # Consolidate batch results into final decisions
    logger.info("[CONSOLIDATION] Consolidating modern validation batch results...")
    
    try:
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
            return _step_7_fallback_processing(self, verified_results, translation_data_map)
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to consolidate modern validation results: {e}")
        import traceback
        traceback.print_exc()
        return _step_7_fallback_processing(self, verified_results, translation_data_map)


def _calculate_step5_step6_consistency(step5_data: Dict, step6_data: Dict) -> float:
    """Calculate consistency between Step 5 and Step 6 results"""
    if not step5_data or not step6_data:
        return 0.0
    
    step5_translations = step5_data.get('all_translations', {})
    step6_translations = step6_data.get('verified_translations', {})
    
    if not step5_translations or not step6_translations:
        return 0.0
    
    # Calculate overlap in language codes
    step5_langs = set(step5_translations.keys())
    step6_langs = set(step6_translations.keys())
    
    if not step5_langs:
        return 0.0
    
    overlap = len(step5_langs & step6_langs)
    consistency = overlap / len(step5_langs)
    
    return consistency


def _convert_modern_batch_results_to_decisions(consolidated_data: Dict, verified_results: List, translation_data_map: Dict) -> List[Dict]:
    """Convert modern validation batch results to final decisions format with Step 5 & 6 integration"""
    
    final_decisions = []
    
    # Extract results from all batches
    batches = consolidated_data.get('batches', {})
    
    logger.info(f"[CONVERSION] Converting results from {len(batches)} batches to final decisions")
    
    for batch_key, batch_info in batches.items():
        batch_data = batch_info.get('data', {})
        batch_results = batch_data.get('results', [])
        
        logger.info(f"[CONVERSION] Processing batch {batch_key} with {len(batch_results)} results")
        
        for result in batch_results:
            term = result.get('term', '')
            if not term:
                continue
            
            # Get enhanced scores from modern validation agents
            enhanced_score = result.get('enhanced_score', 0.0)
            ml_features = result.get('ml_features', {})
            advanced_context = result.get('advanced_context_analysis', {})
            validation_result = result.get('validation_result', {}) if isinstance(result.get('validation_result'), dict) else {}
            
            # Get original Step 6 and Step 5 data
            original_step6_data = None
            original_step5_data = translation_data_map.get(term, {})
            
            for verified_result in verified_results:
                if verified_result.get('term') == term:
                    original_step6_data = verified_result
                    break
            
            if not original_step6_data:
                continue
            
            # Calculate comprehensive modern validation score using agent results + Step 5 & 6 data
            modern_score = _calculate_comprehensive_score_with_agent_results(
                result, enhanced_score, ml_features, advanced_context, validation_result,
                original_step5_data, original_step6_data
            )
            
            # Generate comprehensive decision reasons including agent analysis
            decision_reasons = _generate_comprehensive_reasons_with_agent_results(
                result, enhanced_score, ml_features, advanced_context, validation_result,
                original_step5_data, original_step6_data
            )
            
            # Make final decision based on comprehensive score and translatability
            translatability_analysis = result.get('translatability_analysis', {})
            final_decision, status = _make_final_decision(modern_score, translatability_analysis, term)
            
            # Get translatability_score from Step 5 data
            translatability_score = 0.0
            if original_step5_data:
                translated_langs = original_step5_data.get('translated_languages', 0)
                total_langs = original_step5_data.get('total_languages', 1)
                translatability_score = translated_langs / max(1, total_langs)
            
            # Calculate quality tier based on modern_score
            quality_tier = _determine_quality_tier(modern_score)
            
            # Generate comprehensive decision reasoning
            decision_reasoning = _generate_decision_reasoning(
                term, modern_score, translatability_score, final_decision, 
                translatability_analysis, decision_reasons, original_step5_data, original_step6_data
            )
            
            # Create comprehensive decision record with batch processing metadata
            decision_record = {
                'term': term,
                'decision': final_decision,
                'status': status,
                'final_score': modern_score,  # CRITICAL: Add final_score for Step 9
                'comprehensive_score': modern_score,  # FIX: Add comprehensive_score field
                'translatability_score': translatability_score,  # FIX: Add translatability_score field
                'quality_tier': quality_tier,  # FIX: Add quality_tier field
                'decision_reasoning': decision_reasoning,  # FIX: Add decision_reasoning field
                'modern_validation_score': modern_score,
                'ml_quality_score': enhanced_score,
                'decision_reasons': decision_reasons,
                'advanced_context_analysis': advanced_context,
                'agent_validation_result': validation_result,
                'modern_validation_metadata': {
                    'validation_system': 'modern_parallel_validation_v4.1_batch_processing_with_agents',
                    'ml_features': ml_features,
                    'batch_processing': True,
                    'organized_validation': True,
                    'agent_processing': True,
                    'step5_integration': bool(original_step5_data),
                    'step6_integration': bool(original_step6_data),
                    'processing_method': result.get('validation_method', 'enhanced_agent_batch_validation'),
                    'search_strategy': result.get('search_strategy', 'comprehensive'),
                    'term_characteristics': result.get('term_characteristics', {})
                },
                'step5_integration_data': {
                    'translation_success_rate': original_step5_data.get('translated_languages', 0) / max(1, original_step5_data.get('total_languages', 1)) if original_step5_data else 0,
                    'processing_tier': original_step5_data.get('processing_tier', 'unknown') if original_step5_data else 'unknown',
                    'gpu_worker': original_step5_data.get('gpu_worker', 'unknown') if original_step5_data else 'unknown',
                    'processing_time': original_step5_data.get('processing_time_seconds', 0) if original_step5_data else 0
                },
                'step6_integration_data': {
                    'verification_passed': original_step6_data.get('verification_passed', False),
                    'verification_issues_count': original_step6_data.get('verification_issues_count', 0),
                    'verified_translations_count': len(original_step6_data.get('verified_translations', {}))
                },
                'review_timestamp': datetime.now().isoformat(),
                'batch_info': {
                    'batch_id': result.get('batch_id', 'unknown'),
                    'worker_id': result.get('worker_id', 'unknown'),
                    'processing_attempt': result.get('processing_attempt', 1),
                    'batch_key': batch_key
                }
            }
            
            final_decisions.append(decision_record)
    
    # Sort by term for consistency
    final_decisions.sort(key=lambda x: x.get('term', ''))
    
    logger.info(f"[CONVERSION] Converted {len(final_decisions)} batch results to final decisions")
    
    return final_decisions


def _calculate_comprehensive_score_with_agent_results(result: Dict, enhanced_score: float, ml_features: Dict, 
                                                    advanced_context: Dict, validation_result: Dict,
                                                    step5_data: Dict, step6_data: Dict) -> float:
    """Calculate comprehensive modern validation score including agent validation results and translatability analysis"""
    
    # Get translatability analysis from result
    translatability_analysis = result.get('translatability_analysis', {})
    
    # ROBUST SCORING: Start with a reasonable base score even if agents fail
    # Base score from modern validation ML system (25% weight - reduced to make room for translatability)
    # CRITICAL FIX: Check for None, not > 0, because 0.0 is a valid score (after penalties)
    if enhanced_score is not None:
        base_score = enhanced_score * 0.25
    else:
        # Fallback: Use Step 5/6 data to estimate a reasonable base score
        base_score = 0.4  # Start with neutral score if ML system fails
    
    # Agent validation result score (20% weight)
    # CRITICAL FIX: Check if validation_result exists and has score, not if score > 0
    agent_score = 0.0  # Initialize to 0.0 for cases where agent fails
    if validation_result and 'score' in validation_result:
        agent_score = validation_result.get('score', 0.0)
        base_score += agent_score * 0.2
    # CRITICAL FIX: Removed fallback bonus - agent failure should not help score
    
    # Translatability Analysis Integration (15% weight - NEW)
    if translatability_analysis:
        translatability_category = translatability_analysis.get('category', 'UNKNOWN')
        language_coverage_rate = translatability_analysis.get('language_coverage_rate', 0.0)
        same_language_rate = translatability_analysis.get('same_language_rate', 0.0)
        untranslatable_indicators = translatability_analysis.get('untranslatable_indicators', [])
        
        # Translatability scoring with NEW TERM logic
        # CRITICAL FIX: Reduced bonuses from 0.12 to 0.08 to prevent over-rewarding
        if translatability_category == "UNTRANSLATABLE":
            has_technical_value = translatability_analysis.get('has_technical_value', False)
            if has_technical_value:
                if same_language_rate > 0.8:
                    # Universal technical terms - strong NEW TERM candidates
                    base_score += 0.08  # Reduced from 0.12
                else:
                    # Technical terms that should be NEW TERMS
                    base_score += 0.06  # Reduced from 0.08
            else:
                # Non-technical untranslatable terms get penalty
                base_score -= 0.05
        elif translatability_category == "PARTIALLY_TRANSLATABLE":
            if language_coverage_rate >= 0.3:
                base_score += 0.04  # Reduced from 0.05
            else:
                base_score -= 0.02
        else:  # FULLY_TRANSLATABLE
            base_score += 0.08  # Reduced from 0.12
        
        # Penalty for multiple untranslatable indicators
        indicator_penalty = min(0.05, len(untranslatable_indicators) * 0.01)
        base_score -= indicator_penalty
    
    # Step 5 Translation Quality Integration (25% weight) - BALANCED SCORING
    # CRITICAL FIX: Reduced bonuses to match 25% weight (max 0.15 instead of 0.33)
    if step5_data:
        translated_langs = step5_data.get('translated_languages', 0)
        total_langs = step5_data.get('total_languages', 1)
        translation_success_rate = translated_langs / max(1, total_langs)
        
        # Balanced translation quality scoring (max 0.15 = 25% weight)
        if translation_success_rate >= 0.95:
            base_score += 0.15  # Reduced from 0.25 - Excellent translation
        elif translation_success_rate >= 0.85:
            base_score += 0.12  # Reduced from 0.20 - Very good translation
        elif translation_success_rate >= 0.70:
            base_score += 0.09  # Reduced from 0.15 - Good translation
        elif translation_success_rate >= 0.50:
            base_score += 0.06  # Reduced from 0.10 - Acceptable translation
        elif translation_success_rate >= 0.30:
            base_score += 0.03  # Reduced from 0.05 - Partial translation
        
        # REMOVED: Processing tier bonus (+0.05) - was inflating scores
        # REMOVED: Basic translation bonus (+0.03) - was inflating scores
    else:
        # Penalty for missing Step 5 data
        base_score -= 0.05
    
    # Step 6 Verification Quality Integration (10% weight) - BALANCED SCORING
    # CRITICAL FIX: Reduced bonuses to match 10% weight (max 0.10 instead of 0.20)
    if step6_data:
        verification_passed = step6_data.get('verification_passed', False)
        verification_issues = step6_data.get('verification_issues_count', 0)
        verified_translations = step6_data.get('verified_translations', {})
        
        if verification_passed:
            base_score += 0.08  # Reduced from 0.15 - Passing verification
            
            # Additional bonus based on number of verified translations
            verified_count = len(verified_translations) if isinstance(verified_translations, dict) else 0
            if verified_count >= 10:
                base_score += 0.02  # Reduced from 0.05 - Extensive verification
            elif verified_count >= 5:
                base_score += 0.01  # Reduced from 0.03 - Good verification coverage
        else:
            # Stricter penalty system - verification failure should matter
            if verification_issues <= 2:
                base_score += 0.02  # Reduced from 0.05 - Minor issues
            elif verification_issues <= 5:
                base_score += 0.01  # Reduced from 0.02 - Moderate issues
            else:
                penalty = min(0.05, verification_issues * 0.01)
                base_score -= penalty
        
        # REMOVED: Bonus for having Step 6 data (+0.02) - was inflating scores
    else:
        # Penalty for missing Step 6 data
        base_score -= 0.03
    
    # Advanced Context Analysis Integration (10% weight) - IMPROVED SCORING
    if advanced_context:
        context_quality = advanced_context.get('context_quality', 0)
        semantic_richness = advanced_context.get('semantic_richness', 0)
        
        # More generous context scoring
        base_score += context_quality * 0.08
        base_score += semantic_richness * 0.07
        
        # Bonus for having context analysis at all
        base_score += 0.02
    else:
        # Small bonus even without context analysis if we have other good data
        if step5_data and step6_data:
            base_score += 0.03
    
    # DEBUGGING: Log the scoring breakdown for the first few terms
    term_name = result.get('term', 'unknown')
    if hasattr(_calculate_comprehensive_score_with_agent_results, 'debug_count'):
        _calculate_comprehensive_score_with_agent_results.debug_count += 1
    else:
        _calculate_comprehensive_score_with_agent_results.debug_count = 1
    
    if _calculate_comprehensive_score_with_agent_results.debug_count <= 5:
        print(f"🔍 SCORING DEBUG for '{term_name}':")
        print(f"   Enhanced Score: {enhanced_score}")
        print(f"   Agent Score: {agent_score}")
        print(f"   Step5 Data: {bool(step5_data)}")
        print(f"   Step6 Data: {bool(step6_data)}")
        print(f"   Final Score: {max(0.0, min(1.0, base_score))}")
    
    return max(0.0, min(1.0, base_score))


def _generate_comprehensive_reasons_with_agent_results(result: Dict, enhanced_score: float, ml_features: Dict,
                                                     advanced_context: Dict, validation_result: Dict,
                                                     step5_data: Dict, step6_data: Dict) -> List[str]:
    """Generate comprehensive decision reasons including agent validation results and translatability analysis"""
    
    reasons = []
    
    # Translatability Analysis Reasons (PRIORITY - shown first)
    translatability_analysis = result.get('translatability_analysis', {})
    if translatability_analysis:
        category = translatability_analysis.get('category', 'UNKNOWN')
        is_translatable = translatability_analysis.get('is_translatable', True)
        language_coverage_rate = translatability_analysis.get('language_coverage_rate', 0.0)
        same_language_rate = translatability_analysis.get('same_language_rate', 0.0)
        untranslatable_indicators = translatability_analysis.get('untranslatable_indicators', [])
        recommended_action = translatability_analysis.get('recommended_action', 'REVIEW')
        
        # Primary translatability assessment with NEW TERM logic
        if not is_translatable:
            has_technical_value = translatability_analysis.get('has_technical_value', False)
            if has_technical_value:
                reasons.append(f"🆕 NEW TERM CANDIDATE: {category} - Untranslatable but has technical value")
            else:
                reasons.append(f"⚠️ UNTRANSLATABLE TERM: {category} - Cannot be translated, no technical value")
        else:
            reasons.append(f"✅ TRANSLATABLE: {category} - Can be translated")
        
        reasons.append(f"Translation Coverage: {language_coverage_rate:.1%} - {'Excellent' if language_coverage_rate >= 0.9 else 'Good' if language_coverage_rate >= 0.7 else 'Poor'}")
        
        # Same language analysis
        if same_language_rate > 0.8:
            reasons.append(f"Universal Technical Term: {same_language_rate:.1%} unchanged across languages")
        elif same_language_rate > 0.3:
            reasons.append(f"Partially Universal: {same_language_rate:.1%} unchanged in some languages")
        
        # Untranslatable indicators
        if untranslatable_indicators:
            indicator_descriptions = {
                'contains_numbers': 'numeric chars', 'contains_special_chars': 'special chars',
                'very_short': '≤2 chars', 'very_long': '≥20 chars', 'all_uppercase': 'acronym',
                'file_extension': 'file ext', 'error_related': 'error term', 'web_related': 'web/URL'
            }
            indicators_text = ', '.join([indicator_descriptions.get(ind, ind) for ind in untranslatable_indicators[:3]])
            reasons.append(f"Technical Indicators: {indicators_text}")
            
            if len(untranslatable_indicators) >= 3:
                reasons.append("Multiple technical patterns suggest specialized/untranslatable term")
        
        # Recommended action
        reasons.append(f"Translatability Recommendation: {recommended_action}")
    
    # Agent validation reasons
    if validation_result:
        agent_score = validation_result.get('score', 0.0)
        agent_status = validation_result.get('status', 'unknown')
        reasons.append(f"Agent Validation: {agent_status} (score: {agent_score:.3f})")
        
        # Web research summary
        web_research = validation_result.get('web_research', {})
        if web_research:
            web_summary = web_research.get('summary', '')
            if web_summary:
                reasons.append(f"Agent Research: {web_summary[:100]}...")
    
    # Modern validation reasons
    reasons.append(f"Modern ML Quality Score: {enhanced_score:.3f}")
    
    if ml_features:
        domain_relevance = ml_features.get('domain_relevance', 0)
        if domain_relevance > 0:
            reasons.append(f"Modern - Domain Relevance: {domain_relevance:.2f}")
    
    # Step 5 integration reasons
    if step5_data:
        translated_langs = step5_data.get('translated_languages', 0)
        total_langs = step5_data.get('total_languages', 0)
        if total_langs > 0:
            success_rate = translated_langs / total_langs
            reasons.append(f"Step 5 - Translation Success: {success_rate:.1%} ({translated_langs}/{total_langs} languages)")
        
        processing_tier = step5_data.get('processing_tier', 'unknown')
        if processing_tier != 'unknown':
            reasons.append(f"Step 5 - Processing Tier: {processing_tier}")
        
        gpu_worker = step5_data.get('gpu_worker', 'unknown')
        if gpu_worker != 'unknown' and gpu_worker is not None:
            reasons.append(f"Step 5 - GPU Worker: {gpu_worker}")
    
    # Step 6 integration reasons
    if step6_data:
        verification_passed = step6_data.get('verification_passed', False)
        verification_issues = step6_data.get('verification_issues_count', 0)
        
        if verification_passed:
            reasons.append("Step 6 - Verification: PASSED")
        else:
            reasons.append(f"Step 6 - Verification: FAILED ({verification_issues} issues)")
        
        verified_count = len(step6_data.get('verified_translations', {}))
        reasons.append(f"Step 6 - Verified Translations: {verified_count} languages")
    
    # Advanced context analysis reasons
    if advanced_context:
        domain_scores = advanced_context.get('domain_classification', {})
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            reasons.append(f"Modern - Best Domain: {best_domain[0]} ({best_domain[1]:.2f})")
        
        usage_patterns = advanced_context.get('usage_patterns', [])
        if usage_patterns:
            reasons.append(f"Modern - Usage Patterns: {', '.join(usage_patterns)}")
    
    # Batch processing metadata
    batch_id = result.get('batch_id', 'unknown')
    processing_method = result.get('validation_method', 'agent_batch_validation')
    reasons.append(f"Modern - Agent Batch Processing: {processing_method} (batch {batch_id})")
    
    return reasons


def _make_final_decision(score: float, translatability_analysis: Dict = None, term: str = None) -> Tuple[str, str]:
    """
    Make final decision based on comprehensive score and translatability analysis.
    
    NEW: Applies STRICTER thresholds for single-word terms to ensure higher quality.
    Multi-word technical phrases use standard thresholds.
    """
    
    # Determine if this is a single-word term
    word_count = len(term.split()) if term else 2  # Default to multi-word if term not provided
    is_single_word = (word_count == 1)
    
    # Check if this is a NEW TERM candidate
    if translatability_analysis:
        recommended_action = translatability_analysis.get('recommended_action', '')
        if 'NEW_TERM' in recommended_action:
            if is_single_word:
                # Stricter thresholds for single-word new terms
                if score >= 0.7:  # +0.1 stricter
                    return "APPROVED_AS_NEW_TERM", "approved_as_new_term"
                elif score >= 0.5:  # +0.1 stricter
                    return "CONDITIONALLY_APPROVED_AS_NEW_TERM", "conditionally_approved_as_new_term"
                else:
                    return "NEEDS_REVIEW_FOR_NEW_TERM", "needs_review_for_new_term"
            else:
                # Standard thresholds for multi-word new terms
                if score >= 0.6:
                    return "APPROVED_AS_NEW_TERM", "approved_as_new_term"
                elif score >= 0.4:
                    return "CONDITIONALLY_APPROVED_AS_NEW_TERM", "conditionally_approved_as_new_term"
                else:
                    return "NEEDS_REVIEW_FOR_NEW_TERM", "needs_review_for_new_term"
    
    # Standard decision logic with STRICTER THRESHOLDS (Option E - Enhanced Quality)
    # Based on Option D + additional filtering for generic terms
    # Research: Technical documentation approval rate = 40-60% (high precision)
    # Target: ~1,200-1,350 approved terms (52-59%) - high quality terms only
    # Focus: Filter out generic single-word and generic two-word verb+noun combinations
    if is_single_word:
        # SINGLE-WORD THRESHOLDS (Stricter to filter generic terms like accept, add, create)
        # Raised by +0.05-0.08 from Option D to reduce generic single-word approvals
        if score >= 0.55:  # Exceptional single words (very rare, domain-specific)
            return "APPROVED", "approved"
        elif score >= 0.45:  # Quality single words (well-validated, technical)
            return "CONDITIONALLY_APPROVED", "conditionally_approved"
        elif score >= 0.38:  # Marginal single words (human review needed)
            return "NEEDS_REVIEW", "needs_review"
        else:
            return "REJECTED", "rejected"  # Generic/low-quality single words (< 0.38)
    else:
        # MULTI-WORD THRESHOLDS (Moderately stricter to filter generic two-word terms)
        # Raised by +0.03 from Option D to filter "add buttons", "create new", "click save"
        if score >= 0.48:  # Excellent multi-word terms (domain-specific, technical)
            return "APPROVED", "approved"
        elif score >= 0.38:  # Good multi-word terms (validated, contextual)
            return "CONDITIONALLY_APPROVED", "conditionally_approved"
        elif score >= 0.32:  # Marginal terms (manual review) - equivalent to TM 70-75%
            return "NEEDS_REVIEW", "needs_review"
        else:
            return "REJECTED", "rejected"  # Low quality terms (< 0.32)


def _count_actual_batch_files(batch_dir: str) -> int:
    """Count actual batch files in the batch directory"""
    try:
        import os
        import glob
        batch_pattern = os.path.join(str(batch_dir), "batch_*.json")
        batch_files = glob.glob(batch_pattern)
        return len(batch_files)
    except Exception as e:
        logger.warning(f"Could not count batch files: {e}")
        return 0


def _create_final_decisions_data_with_batch_info(final_decisions: List[Dict], consolidated_data: Dict, step7_manager) -> Dict:
    """Create comprehensive final decisions data structure with batch processing information"""
    
    # Calculate statistics including NEW TERM categories
    total_decisions = len(final_decisions)
    fully_approved = len([d for d in final_decisions if d['status'] == 'approved'])
    conditionally_approved = len([d for d in final_decisions if d['status'] == 'conditionally_approved'])
    needs_review = len([d for d in final_decisions if d['status'] == 'needs_review'])
    rejected = len([d for d in final_decisions if d['status'] == 'rejected'])
    
    # NEW TERM categories
    approved_as_new_term = len([d for d in final_decisions if d['status'] == 'approved_as_new_term'])
    conditionally_approved_as_new_term = len([d for d in final_decisions if d['status'] == 'conditionally_approved_as_new_term'])
    needs_review_for_new_term = len([d for d in final_decisions if d['status'] == 'needs_review_for_new_term'])
    
    # Calculate average scores
    avg_validation_score = sum(d.get('modern_validation_score', 0) for d in final_decisions) / max(1, total_decisions)
    avg_ml_score = sum(d.get('ml_quality_score', 0) for d in final_decisions) / max(1, total_decisions)
    
    return {
        'metadata': {
            'step': 7,
            'process_name': 'final_review_decision_modern_validation_agent_batch_processing',
            'timestamp': datetime.now().isoformat(),
            'total_terms_reviewed': total_decisions,
            'total_decisions_made': total_decisions,
            'modern_validation_system': 'enhanced_v4.1_agent_batch_processing',
            'batch_processing_enabled': True,
            'agent_processing_enabled': True,
            'organized_validation_structure': True,
            'integration_features': [
                'proper_agent_batch_processing_workflow',
                'organized_folder_structure_with_batch_files',
                'modern_terminology_review_agents',
                'step5_translation_data_integration',
                'step6_verification_data_integration', 
                'ml_based_quality_scoring',
                'advanced_context_analysis',
                'comprehensive_decision_reasoning',
                'enhanced_validation_caching',
                'robust_error_handling_and_resumption',
                'web_research_and_validation',
                'term_characteristics_analysis'
            ]
        },
        'batch_processing_summary': {
            'total_batches_processed': len(consolidated_data.get('batches', {})),
            'batch_directory': str(step7_manager.batch_dir),
            'cache_directory': str(step7_manager.cache_dir),
            'logs_directory': str(step7_manager.logs_dir),
            'consolidation_info': consolidated_data.get('consolidation_info', {}),
            'summary_statistics': consolidated_data.get('summary_statistics', {}),
            # Add actual batch count from filesystem as fallback
            'total_batches': _count_actual_batch_files(step7_manager.batch_dir),
            'terms_per_batch': round(total_decisions / max(1, len(consolidated_data.get('batches', {}))), 1),
            'total_processing_time_seconds': consolidated_data.get('consolidation_info', {}).get('total_duration', 0)
        },
        'modern_validation_summary': {
            'total_decisions': total_decisions,
            'fully_approved': fully_approved,
            'conditionally_approved': conditionally_approved,
            'needs_review': needs_review,
            'rejected': rejected,
            'approved_as_new_term': approved_as_new_term,
            'conditionally_approved_as_new_term': conditionally_approved_as_new_term,
            'needs_review_for_new_term': needs_review_for_new_term,
            'average_validation_score': round(avg_validation_score, 3),
            'average_ml_quality_score': round(avg_ml_score, 3),
            'decision_distribution': {
                'approved_rate': round(fully_approved / max(1, total_decisions) * 100, 1),
                'conditional_rate': round(conditionally_approved / max(1, total_decisions) * 100, 1),
                'review_rate': round(needs_review / max(1, total_decisions) * 100, 1),
                'rejection_rate': round(rejected / max(1, total_decisions) * 100, 1),
                'new_term_approved_rate': round(approved_as_new_term / max(1, total_decisions) * 100, 1),
                'new_term_conditional_rate': round(conditionally_approved_as_new_term / max(1, total_decisions) * 100, 1),
                'new_term_review_rate': round(needs_review_for_new_term / max(1, total_decisions) * 100, 1)
            }
        },
        'final_decisions': final_decisions
    }


def _log_step7_completion_with_batch_info(final_decisions_data: Dict, decisions_file: str, step7_manager):
    """Log Step 7 completion with comprehensive batch processing statistics"""
    
    summary = final_decisions_data.get('modern_validation_summary', {})
    batch_summary = final_decisions_data.get('batch_processing_summary', {})
    
    logger.info(f"[OK] Step 7 completed with PROPER modern validation agent batch processing:")
    logger.info(f"   Fully approved: {summary.get('fully_approved', 0)}")
    logger.info(f"   Conditionally approved: {summary.get('conditionally_approved', 0)}")
    logger.info(f"   Needs review: {summary.get('needs_review', 0)}")
    logger.info(f"   Rejected: {summary.get('rejected', 0)}")
    logger.info(f"   🆕 NEW TERMS - Approved: {summary.get('approved_as_new_term', 0)}")
    logger.info(f"   🆕 NEW TERMS - Conditional: {summary.get('conditionally_approved_as_new_term', 0)}")
    logger.info(f"   🆕 NEW TERMS - Review needed: {summary.get('needs_review_for_new_term', 0)}")
    logger.info(f"   Average validation score: {summary.get('average_validation_score', 0):.3f}")
    logger.info(f"   Average ML quality score: {summary.get('average_ml_quality_score', 0):.3f}")
    logger.info(f"   Batches processed: {batch_summary.get('total_batches_processed', 0)}")
    logger.info(f"   Batch files directory: {batch_summary.get('batch_directory', 'N/A')}")
    logger.info(f"[FOLDER] Modern validation decisions: {decisions_file}")


def _step_7_fallback_processing(self, verified_results: List, translation_data_map: Dict) -> str:
    """Fallback to in-memory processing if batch processing fails"""
    
    logger.warning("[FALLBACK] Using in-memory processing as fallback...")
    
    # This would contain the original in-memory processing logic as a fallback
    # For now, create a minimal fallback result
    logger.error("[ERROR] Fallback processing not fully implemented - creating minimal result")
    
    decisions_file = os.path.join(str(self.output_dir), "Final_Terminology_Decisions.json")
    fallback_decisions = {
        'metadata': {
            'step': 7,
            'step_name': 'Final Review and Decision (Fallback)',
            'timestamp': datetime.now().isoformat(),
            'total_terms_reviewed': len(verified_results),
            'review_status': 'fallback_processing_used',
            'note': 'Modern validation batch processing failed, fallback used'
        },
        'final_decisions': [
            {
                'term': result.get('term', ''),
                'decision': 'NEEDS_REVIEW',
                'status': 'needs_review',
                'modern_validation_score': 0.5,
                'ml_quality_score': 0.0,
                'decision_reasons': ['Fallback processing - requires manual review'],
                'review_timestamp': datetime.now().isoformat()
            }
            for result in verified_results[:100]  # Limit for fallback
        ],
        'modern_validation_summary': {
            'total_decisions': min(100, len(verified_results)),
            'fully_approved': 0,
            'conditionally_approved': 0,
            'needs_review': min(100, len(verified_results)),
            'rejected': 0,
            'average_validation_score': 0.5,
            'average_ml_quality_score': 0.0
        }
    }
    
    with open(decisions_file, 'w', encoding='utf-8') as f:
        json.dump(fallback_decisions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[FALLBACK] Created fallback decisions file: {decisions_file}")
    return decisions_file


def _analyze_term_translatability(term: str, translatability_score: float, language_coverage_rate: float, 
                                same_language_rate: float, error_rate: float) -> Dict:
    """
    Comprehensive translatability analysis for agent decision-making
    
    Returns detailed analysis including:
    - Translatability category (UNTRANSLATABLE, PARTIALLY_TRANSLATABLE, FULLY_TRANSLATABLE)
    - Untranslatable indicators (technical patterns, file extensions, etc.)
    - Recommended action for agents
    - Reasoning points for decision context
    """
    
    # Identify untranslatable term patterns
    term_lower = term.lower()
    untranslatable_indicators = []
    
    # Pattern detection
    if any(char.isdigit() for char in term):
        untranslatable_indicators.append("contains_numbers")
    if any(char in term for char in '.-_/\\@#$%^&*()+=[]{}|;:,<>?'):
        untranslatable_indicators.append("contains_special_chars")
    if len(term) <= 2:
        untranslatable_indicators.append("very_short")
    if len(term) >= 20:
        untranslatable_indicators.append("very_long")
    if term.isupper() and len(term) > 3:
        untranslatable_indicators.append("all_uppercase")
    if '.' in term:
        untranslatable_indicators.append("contains_dot")
    
    # File extension detection
    file_extensions = ['.exe', '.dll', '.log', '.txt', '.cfg', '.xml', '.json', '.dwg', '.dxf', '.dmg', 
                      '.3mf', '.ifc', '.step', '.iges', '.obj', '.fbx', '.max', '.blend']
    if any(ext in term_lower for ext in file_extensions):
        untranslatable_indicators.append("file_extension")
    
    # Error/technical term detection
    error_terms = ['err', 'error', 'fail', 'exception', 'debug', 'log', 'trace', 'stack']
    if any(error_term in term_lower for error_term in error_terms):
        untranslatable_indicators.append("error_related")
    
    # Common stopwords
    stopwords = ['and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had']
    if term_lower in stopwords:
        untranslatable_indicators.append("common_stopword")
    
    # Technical codes and identifiers
    if term_lower.startswith(('id_', 'uid_', 'guid_', 'uuid_')):
        untranslatable_indicators.append("identifier_prefix")
    
    # URL/web patterns
    if any(pattern in term_lower for pattern in ['http', 'www.', '.com', '.org', '.net', 'api_', 'url_']):
        untranslatable_indicators.append("web_related")
    
    # Version patterns
    if any(pattern in term_lower for pattern in ['v1.', 'v2.', 'version', '_v', '.0', '.1', '.2']):
        untranslatable_indicators.append("version_pattern")
    
    # Classify translatability
    if translatability_score == 0.0 or language_coverage_rate == 0.0:
        category = "UNTRANSLATABLE"
        reason = "zero_translations" if language_coverage_rate == 0.0 else "zero_score"
        is_translatable = False
    elif (translatability_score < 0.5 or language_coverage_rate < 0.5 or 
          len(untranslatable_indicators) >= 3):
        category = "PARTIALLY_TRANSLATABLE"
        reason = "low_score_coverage_or_technical_indicators"
        is_translatable = True
    else:
        category = "FULLY_TRANSLATABLE"
        reason = "good_score_and_coverage"
        is_translatable = True
    
    # Assess technical value for untranslatable terms (NEW TERM verification)
    has_technical_value = _assess_technical_value_for_new_term(term, untranslatable_indicators)
    
    # Generate recommended action for agents
    if category == "UNTRANSLATABLE":
        if same_language_rate > 0.8 and has_technical_value:
            recommended_action = "APPROVE_AS_NEW_TERM"  # Universal technical terms
            action_reason = "Universal technical term - should be approved as new term for terminology database"
        elif has_technical_value:
            recommended_action = "CONDITIONAL_APPROVAL_AS_NEW_TERM"  # Technical terms needing verification
            action_reason = "Technical term that cannot be translated - verify as new term candidate"
        elif any(indicator in untranslatable_indicators for indicator in ['common_stopword', 'error_related']):
            recommended_action = "REJECT"
            action_reason = "Invalid term - stopword or error term, not suitable for terminology database"
        else:
            recommended_action = "NEEDS_REVIEW_FOR_NEW_TERM"
            action_reason = "Untranslatable term - manual review needed to determine if it's a valid new term"
    elif category == "PARTIALLY_TRANSLATABLE":
        if language_coverage_rate >= 0.3 and translatability_score >= 0.3:
            recommended_action = "CONDITIONAL_APPROVAL"
            action_reason = "Partially translatable - acceptable for specialized terms"
        else:
            recommended_action = "NEEDS_REVIEW"
            action_reason = "Poor translatability - requires manual evaluation"
    else:  # FULLY_TRANSLATABLE
        if language_coverage_rate >= 0.8 and translatability_score >= 0.8:
            recommended_action = "APPROVE"
            action_reason = "Excellent translatability - strong approval candidate"
        elif language_coverage_rate >= 0.6 and translatability_score >= 0.6:
            recommended_action = "CONDITIONAL_APPROVAL"
            action_reason = "Good translatability - conditional approval"
        else:
            recommended_action = "NEEDS_REVIEW"
            action_reason = "Inconsistent translatability metrics - requires review"
    
    # Generate reasoning points for agents
    reasoning_points = []
    
    # Translatability category reasoning
    reasoning_points.append(f"Translatability Category: {category} (score: {translatability_score:.2f})")
    reasoning_points.append(f"Language Coverage: {language_coverage_rate:.1%} - {'Excellent' if language_coverage_rate >= 0.9 else 'Good' if language_coverage_rate >= 0.7 else 'Poor'}")
    
    # Same language analysis
    if same_language_rate > 0.0:
        if same_language_rate > 0.8:
            reasoning_points.append(f"Universal Term: {same_language_rate:.1%} unchanged - likely technical/universal term")
        elif same_language_rate > 0.5:
            reasoning_points.append(f"Partially Universal: {same_language_rate:.1%} unchanged - some universal usage")
        else:
            reasoning_points.append(f"Language-Specific: {same_language_rate:.1%} unchanged - good translation variation")
    
    # Error rate analysis
    if error_rate > 0.0:
        reasoning_points.append(f"Translation Errors: {error_rate:.1%} - {'High error rate' if error_rate > 0.1 else 'Low error rate'}")
    
    # Untranslatable indicators
    if untranslatable_indicators:
        indicator_descriptions = {
            'contains_numbers': 'contains numeric characters',
            'contains_special_chars': 'contains special characters',
            'very_short': 'very short term (≤2 chars)',
            'very_long': 'very long term (≥20 chars)',
            'all_uppercase': 'all uppercase (likely acronym)',
            'contains_dot': 'contains dots (file/version pattern)',
            'file_extension': 'file extension pattern',
            'error_related': 'error/debug related term',
            'common_stopword': 'common stopword',
            'identifier_prefix': 'identifier prefix pattern',
            'web_related': 'web/URL related pattern',
            'version_pattern': 'version number pattern'
        }
        
        indicator_text = ', '.join([indicator_descriptions.get(ind, ind) for ind in untranslatable_indicators])
        reasoning_points.append(f"Technical Indicators: {indicator_text}")
        
        if len(untranslatable_indicators) >= 3:
            reasoning_points.append("Multiple technical indicators suggest specialized/untranslatable term")
    
    # Quality assessment with NEW TERM logic
    if category == "UNTRANSLATABLE":
        if has_technical_value:
            if same_language_rate > 0.8:
                reasoning_points.append("Universal technical term - strong candidate for NEW TERM approval")
            else:
                reasoning_points.append("Technical term that cannot be translated - verify as NEW TERM candidate")
        else:
            reasoning_points.append("Term cannot be translated and lacks technical value - consider rejection")
    elif category == "PARTIALLY_TRANSLATABLE":
        reasoning_points.append("Term has limited translatability - evaluate context and necessity")
    else:
        reasoning_points.append("Term shows good translatability - strong candidate for approval")
    
    return {
        'category': category,
        'reason': reason,
        'is_translatable': is_translatable,
        'has_technical_value': has_technical_value,
        'translatability_score': translatability_score,
        'language_coverage_rate': language_coverage_rate,
        'same_language_rate': same_language_rate,
        'error_rate': error_rate,
        'untranslatable_indicators': untranslatable_indicators,
        'recommended_action': recommended_action,
        'action_reason': action_reason,
        'reasoning_points': reasoning_points,
        'new_term_assessment': {
            'is_new_term_candidate': category == "UNTRANSLATABLE" and has_technical_value,
            'new_term_confidence': "high" if same_language_rate > 0.8 and has_technical_value else "medium" if has_technical_value else "low",
            'technical_domain_relevance': "high" if any(ind in untranslatable_indicators for ind in ['file_extension', 'version_pattern', 'web_related']) else "medium"
        },
        'quality_assessment': {
            'translation_quality': "excellent" if language_coverage_rate >= 0.9 else "good" if language_coverage_rate >= 0.7 else "poor",
            'technical_complexity': "high" if len(untranslatable_indicators) >= 3 else "medium" if len(untranslatable_indicators) >= 1 else "low",
            'universality': "universal" if same_language_rate > 0.8 else "partial" if same_language_rate > 0.3 else "language_specific"
        }
    }


def _assess_technical_value_for_new_term(term: str, untranslatable_indicators: List[str]) -> bool:
    """
    Assess if an untranslatable term has technical value and should be considered as a NEW TERM
    
    Returns True if the term should be verified as a new term candidate
    """
    term_lower = term.lower()
    
    # HIGH VALUE INDICATORS (should be new terms)
    high_value_indicators = [
        'file_extension',      # .dwg, .dxf, .json, etc.
        'version_pattern',     # v1.0, version numbers
        'web_related',         # URLs, API endpoints
        'contains_numbers'     # Technical specifications like M24X750
    ]
    
    # MEDIUM VALUE INDICATORS (may be new terms)
    medium_value_indicators = [
        'contains_special_chars',  # Technical codes with special chars
        'all_uppercase',          # Acronyms like API, HTTP
        'contains_dot'            # Technical notation
    ]
    
    # NEGATIVE VALUE INDICATORS (should NOT be new terms)
    negative_indicators = [
        'common_stopword',    # and, or, the, etc.
        'error_related',      # error messages, debug terms
        'very_short'          # single/double characters
    ]
    
    # Check for negative indicators first
    if any(indicator in untranslatable_indicators for indicator in negative_indicators):
        return False
    
    # Check for high value indicators
    if any(indicator in untranslatable_indicators for indicator in high_value_indicators):
        return True
    
    # Check for medium value indicators
    if any(indicator in untranslatable_indicators for indicator in medium_value_indicators):
        # Additional validation for medium value terms
        
        # Acronyms should be meaningful (3+ characters)
        if 'all_uppercase' in untranslatable_indicators and len(term) >= 3:
            return True
        
        # Technical codes with special chars should be meaningful
        if 'contains_special_chars' in untranslatable_indicators and len(term) >= 3:
            # Check if it looks like a technical specification
            if any(char.isalnum() for char in term):
                return True
        
        # Terms with dots should be technical (not just random)
        if 'contains_dot' in untranslatable_indicators:
            # Check for technical patterns like version numbers, file paths, etc.
            if any(char.isdigit() for char in term) or len(term.split('.')) >= 2:
                return True
    
    # Domain-specific technical terms (Autodesk/CAD related)
    technical_domains = [
        'autodesk', 'autocad', 'maya', 'fusion', '3ds', 'max',
        'dwg', 'dxf', 'ifc', 'step', 'iges', 'obj', 'fbx',
        'api', 'sdk', 'gui', 'ui', 'ux', 'cad', 'cam', 'cae',
        'mesh', 'nurbs', 'spline', 'bezier', 'polygon',
        'render', 'shader', 'material', 'texture', 'lighting'
    ]
    
    if any(domain_term in term_lower for domain_term in technical_domains):
        return True
    
    # Check for technical patterns in the term itself
    # Product codes, model numbers, technical specifications
    if len(term) >= 4:  # Minimum length for meaningful technical terms
        # Pattern: Letters followed by numbers (like M24X750)
        import re
        if re.match(r'^[A-Za-z]+\d+', term) or re.match(r'^\d+[A-Za-z]+', term):
            return True
        
        # Pattern: Mixed alphanumeric with technical separators
        if re.match(r'^[A-Za-z0-9._-]+$', term) and any(char.isdigit() for char in term) and any(char.isalpha() for char in term):
            return True
    
    # Default: if no clear indicators, consider it low value
    return False


def _determine_quality_tier(score: float) -> str:
    """Determine quality tier based on comprehensive score"""
    if score >= 0.85:
        return "Exceptional"
    elif score >= 0.70:
        return "High Quality"
    elif score >= 0.55:
        return "Good Quality"
    elif score >= 0.40:
        return "Standard Quality"
    elif score >= 0.25:
        return "Below Standard"
    else:
        return "Poor Quality"


def _generate_decision_reasoning(term: str, comprehensive_score: float, translatability_score: float,
                                 decision: str, translatability_analysis: Dict, decision_reasons: List[str],
                                 step5_data: Dict, step6_data: Dict) -> str:
    """Generate comprehensive decision reasoning text"""
    
    # Determine if single or multi-word
    word_count = len(term.split())
    is_single_word = (word_count == 1)
    term_type = "single-word" if is_single_word else "multi-word"
    
    # Get threshold ranges for context
    if is_single_word:
        if decision == "APPROVED":
            threshold_info = "≥0.55 (exceptional single-word terms)"
        elif decision == "CONDITIONALLY_APPROVED":
            threshold_info = "0.45-0.54 (quality single-word terms)"
        elif decision == "NEEDS_REVIEW":
            threshold_info = "0.38-0.44 (marginal single-word terms)"
        else:
            threshold_info = "<0.38 (low-quality single-word terms)"
    else:
        if decision == "APPROVED":
            threshold_info = "≥0.48 (excellent multi-word terms)"
        elif decision == "CONDITIONALLY_APPROVED":
            threshold_info = "0.38-0.47 (good multi-word terms)"
        elif decision == "NEEDS_REVIEW":
            threshold_info = "0.32-0.37 (marginal multi-word terms)"
        else:
            threshold_info = "<0.32 (low-quality multi-word terms)"
    
    # Build reasoning text
    reasoning_parts = []
    
    # Score and threshold explanation
    reasoning_parts.append(
        f"Term '{term}' ({term_type}) scored {comprehensive_score:.3f}, "
        f"placing it in the {decision} category (threshold: {threshold_info})."
    )
    
    # Translatability analysis
    if translatability_score > 0:
        trans_category = "Excellent" if translatability_score >= 0.95 else \
                        "Very Good" if translatability_score >= 0.85 else \
                        "Good" if translatability_score >= 0.70 else \
                        "Acceptable" if translatability_score >= 0.50 else "Poor"
        reasoning_parts.append(
            f"Translatability: {trans_category} ({translatability_score:.1%} translation success rate across languages)."
        )
    
    # Step 5 and 6 integration
    if step5_data and step6_data:
        translated_langs = step5_data.get('translated_languages', 0)
        total_langs = step5_data.get('total_languages', 0)
        verified = "passed" if step6_data.get('verification_passed', False) else "flagged for review"
        reasoning_parts.append(
            f"Translation data: {translated_langs}/{total_langs} languages, verification {verified}."
        )
    
    # Translatability analysis insights
    if translatability_analysis:
        category = translatability_analysis.get('category', 'unknown')
        is_translatable = translatability_analysis.get('is_translatable', True)
        
        if not is_translatable:
            reasoning_parts.append(
                f"Translatability concern: classified as '{category}' (may require special handling)."
            )
        elif category in ['Highly Translatable', 'Well Translatable']:
            reasoning_parts.append(
                f"Strong translatability: classified as '{category}'."
            )
    
    # Agent validation insights (top 2 reasons)
    if decision_reasons and len(decision_reasons) > 0:
        top_reasons = decision_reasons[:2]
        reasoning_parts.append(
            f"Agent validation: {'; '.join(top_reasons)}."
        )
    
    # Quality tier
    quality_tier = _determine_quality_tier(comprehensive_score)
    reasoning_parts.append(
        f"Overall quality tier: {quality_tier}."
    )
    
    return " ".join(reasoning_parts)
