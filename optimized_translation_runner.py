#!/usr/bin/env python3
"""
Optimized Translation Runner with Resumable Processing
Uses single model instance with efficient checkpointing and progress tracking
"""

import os
import sys
import json
import time
import pickle
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from nllb_translation_tool import NLLBTranslationTool, TranslationResult
from translation_analyzer import analyze_translation_results


class OptimizedTranslationRunner:
    """
    Optimized translation runner with resumable processing and efficient memory usage
    """
    
    def __init__(self, device: str = "auto", model_size: str = "small", 
                 checkpoint_interval: int = 50):
        """
        Initialize the optimized runner
        
        Args:
            device: Device for translation model
            model_size: Model size to use
            checkpoint_interval: Save checkpoint every N terms
        """
        self.device = device
        self.model_size = model_size
        self.checkpoint_interval = checkpoint_interval
        self.source_lang = "eng_Latn"
        
        # Initialize single translation tool instance
        print("🔧 Initializing Optimized Translation Runner...")
        self.translator = NLLBTranslationTool(model_name=model_size, device=device, batch_size=8)
        
        # Get target languages
        all_languages = self.translator.get_available_languages()
        self.target_languages = [lang for lang in all_languages if lang != self.source_lang]
        
        print(f"✅ Optimized runner initialized")
        print(f"🎯 Target languages: {len(self.target_languages)}")
        
        # Session management
        self.session_id = self._generate_session_id()
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"optimized_{timestamp}"
    
    def _get_checkpoint_file(self, file_type: str) -> str:
        """Get checkpoint file path"""
        return os.path.join(self.checkpoint_dir, f"{self.session_id}_{file_type}_checkpoint.json")
    
    def _save_checkpoint(self, file_type: str, processed_index: int, results: List[Dict], 
                        total_terms: int, start_time: datetime):
        """Save processing checkpoint"""
        checkpoint_data = {
            "session_id": self.session_id,
            "file_type": file_type,
            "processed_index": processed_index,
            "total_terms": total_terms,
            "results_count": len(results),
            "start_time": start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "model_size": self.model_size,
            "device": self.device
        }
        
        checkpoint_file = self._get_checkpoint_file(file_type)
        results_file = checkpoint_file.replace('_checkpoint.json', '_results.json')
        
        try:
            # Save checkpoint info
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Save results separately to manage file size
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Checkpoint saved: {processed_index + 1}/{total_terms} terms")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, file_type: str) -> Optional[Dict]:
        """Load processing checkpoint if exists"""
        checkpoint_file = self._get_checkpoint_file(file_type)
        results_file = checkpoint_file.replace('_checkpoint.json', '_results.json')
        
        if os.path.exists(checkpoint_file) and os.path.exists(results_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                checkpoint_data['results'] = results
                print(f"📂 Checkpoint loaded: {len(results)} results")
                return checkpoint_data
                
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
        
        return None
    
    def find_resumable_sessions(self) -> List[Dict]:
        """Find all resumable sessions"""
        sessions = []
        checkpoint_files = list(Path(self.checkpoint_dir).glob("*_checkpoint.json"))
        
        for checkpoint_file in checkpoint_files:
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                sessions.append({
                    "session_id": checkpoint_data["session_id"],
                    "file_type": checkpoint_data["file_type"],
                    "progress": f"{checkpoint_data['processed_index'] + 1}/{checkpoint_data['total_terms']}",
                    "last_update": checkpoint_data["last_update"],
                    "checkpoint_file": str(checkpoint_file)
                })
            except Exception as e:
                print(f"❌ Failed to read checkpoint {checkpoint_file}: {e}")
        
        return sessions
    
    def translate_term_optimized(self, term_data: Dict) -> Dict:
        """
        Translate a single term efficiently
        
        Args:
            term_data: Term information dictionary
            
        Returns:
            Translation result dictionary
        """
        term = term_data.get('term', '')
        frequency = term_data.get('frequency', 0)
        
        if not term or not term.strip():
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
                "error": "Empty term"
            }
        
        start_time = time.time()
        
        try:
            # Translate to all target languages using the single model instance
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
                "processing_time_seconds": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
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
                "processing_time_seconds": processing_time,
                "error": str(e)
            }
    
    def process_terms_resumable(self, terms: List[Dict], output_file: str, 
                               file_type: str, resume_session: str = None) -> Dict:
        """
        Process terms with resumable checkpointing
        
        Args:
            terms: List of term dictionaries
            output_file: Output file path
            file_type: Type of terms being processed
            resume_session: Session ID to resume from
            
        Returns:
            Processing statistics
        """
        
        # Handle resuming
        start_index = 0
        existing_results = []
        start_time = datetime.now()
        
        if resume_session:
            self.session_id = resume_session
            checkpoint = self._load_checkpoint(file_type)
            if checkpoint:
                start_index = checkpoint['processed_index'] + 1
                existing_results = checkpoint['results']
                start_time = datetime.fromisoformat(checkpoint['start_time'])
                print(f"🔄 Resuming from term {start_index + 1}/{len(terms)}")
        
        # Process remaining terms
        terms_to_process = terms[start_index:]
        total_terms = len(terms)
        
        if not terms_to_process:
            print("✅ All terms already processed!")
            return {"status": "completed", "processed": len(existing_results)}
        
        print(f"🚀 Processing {len(terms_to_process)} terms (starting from {start_index + 1})")
        print(f"📁 Output file: {output_file}")
        print(f"💾 Checkpoints every {self.checkpoint_interval} terms")
        
        all_results = existing_results.copy()
        processed_count = len(existing_results)
        failed_count = 0
        
        # Process terms sequentially with progress tracking
        for i, term_data in enumerate(terms_to_process):
            current_index = start_index + i
            
            try:
                result = self.translate_term_optimized(term_data)
                all_results.append(result)
                
                if result.get('error'):
                    failed_count += 1
                else:
                    processed_count += 1
                
                # Progress reporting
                if (i + 1) % 10 == 0 or i + 1 == len(terms_to_process):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (processed_count - len(existing_results)) / elapsed if elapsed > 0 else 0
                    eta = (len(terms_to_process) - i - 1) / rate if rate > 0 else 0
                    
                    print(f"📊 Progress: {current_index + 1}/{total_terms} ({((current_index + 1)/total_terms)*100:.1f}%) | "
                          f"Rate: {rate:.3f} terms/sec | ETA: {eta/3600:.1f} hours")
                
                # Save checkpoint
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(file_type, current_index, all_results, total_terms, start_time)
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Failed to process term {current_index + 1}: {e}")
        
        # Final checkpoint and results
        self._save_checkpoint(file_type, len(terms) - 1, all_results, total_terms, start_time)
        self._save_final_results(all_results, output_file, file_type, start_time)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Processing completed!")
        print(f"📊 Processed: {processed_count}/{total_terms} terms")
        print(f"❌ Failed: {failed_count} terms")
        print(f"⏱️  Total time: {processing_time/3600:.1f} hours")
        print(f"🚀 Rate: {processed_count/processing_time:.3f} terms/second")
        
        return {
            "status": "completed",
            "processed": processed_count,
            "failed": failed_count,
            "processing_time": processing_time,
            "session_id": self.session_id
        }
    
    def _save_final_results(self, results: List[Dict], output_file: str, 
                           file_type: str, start_time: datetime):
        """Save final results with processing info"""
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        output_data = {
            "processing_info": {
                "status": "completed",
                "session_id": self.session_id,
                "file_type": file_type,
                "total_terms": len(results),
                "processing_time_seconds": processing_time,
                "terms_per_second": len(results) / processing_time if processing_time > 0 else 0,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "source_language": self.source_lang,
                "target_languages_count": len(self.target_languages),
                "device_used": self.device,
                "model_size": self.model_size,
                "checkpoint_interval": self.checkpoint_interval
            },
            "results": results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Final results saved to: {output_file}")
            
            # Also save summary
            summary_file = output_file.replace('.json', '_summary.json')
            summary_data = {
                "processing_info": output_data["processing_info"],
                "result_count": len(results),
                "sample_results": results[:5] if results else []
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Failed to save final results: {e}")


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
    parser = argparse.ArgumentParser(description="Optimized NLLB Translation Analysis")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use (default: auto)")
    parser.add_argument("--model-size", choices=["small", "medium", "1.3B", "3.3B"], default="small",
                       help="Model size (default: small)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                       help="Save checkpoint every N terms (default: 50)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Session ID to resume from")
    parser.add_argument("--list-sessions", action="store_true",
                       help="List available sessions")
    parser.add_argument("--max-terms", type=int, default=None,
                       help="Maximum terms to process per file")
    parser.add_argument("--skip-processing", action="store_true",
                       help="Skip processing, only analyze existing results")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = OptimizedTranslationRunner(
        device=args.device,
        model_size=args.model_size,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # List sessions if requested
    if args.list_sessions:
        sessions = runner.find_resumable_sessions()
        if sessions:
            print("📋 AVAILABLE SESSIONS:")
            for session in sessions:
                print(f"  • {session['session_id']} ({session['file_type']}) - {session['progress']}")
        else:
            print("📋 No resumable sessions found")
        return
    
    if args.skip_processing:
        print("⏭️  Skipping processing, analyzing existing results...")
        # Find and analyze existing results
        result_files = []
        if os.path.exists("translation_results"):
            for file_name in os.listdir("translation_results"):
                if (file_name.endswith('.json') and 
                    'translation_results' in file_name and 
                    'summary' not in file_name):
                    result_files.append(os.path.join("translation_results", file_name))
        
        if result_files:
            for result_file in result_files:
                print(f"📊 Analyzing: {os.path.basename(result_file)}")
                analyze_translation_results(result_file, "analysis_reports")
        else:
            print("❌ No result files found")
        return
    
    # Process files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dictionary terms
    dict_file = "Fast_Dictionary_Terms_20250903_123659.json"
    if os.path.exists(dict_file):
        print(f"\n📚 PROCESSING DICTIONARY TERMS")
        print("=" * 50)
        
        dictionary_terms = load_term_candidates(dict_file)
        
        if args.max_terms:
            dictionary_terms = dictionary_terms[:args.max_terms]
            print(f"🔢 Limited to {args.max_terms} terms")
        
        dict_output = f"translation_results/optimized_dictionary_results_{timestamp}.json"
        
        runner.process_terms_resumable(
            dictionary_terms,
            dict_output,
            'dictionary',
            args.resume
        )
    
    # Non-dictionary terms
    non_dict_file = "Fast_Non_Dictionary_Terms_20250903_123659.json"
    if os.path.exists(non_dict_file):
        print(f"\n📖 PROCESSING NON-DICTIONARY TERMS")
        print("=" * 50)
        
        non_dict_terms = load_term_candidates(non_dict_file)
        
        if args.max_terms:
            non_dict_terms = non_dict_terms[:args.max_terms]
            print(f"🔢 Limited to {args.max_terms} terms")
        
        non_dict_output = f"translation_results/optimized_non_dictionary_results_{timestamp}.json"
        
        runner.process_terms_resumable(
            non_dict_terms,
            non_dict_output,
            'non_dictionary',
            args.resume
        )
    
    print(f"\n🎉 PROCESSING COMPLETED!")
    print(f"📋 Session ID: {runner.session_id}")
    print(f"💾 Use --resume {runner.session_id} to resume if interrupted")


if __name__ == "__main__":
    main()

