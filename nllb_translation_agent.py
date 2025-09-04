#!/usr/bin/env python3
"""
NLLB Translation Agent using smolagents
Integrates nllb_translation_tool.py as a custom smolagents tool for multilingual translation
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import smolagents components
from smolagents import CodeAgent, Tool, HfApiModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import the NLLB translation tool
from nllb_translation_tool import NLLBTranslationTool, TranslationResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLLBTranslationSmolTool(Tool):
    """
    Smolagents tool wrapper for the NLLBTranslationTool class
    Provides multilingual translation functionality to agents using NLLB-200
    """
    
    name = "nllb_translator"
    description = """
    A comprehensive multilingual translation tool powered by Facebook's NLLB-200 model.
    
    This tool can:
    - Translate text between any of the 200 supported languages
    - Perform batch translation for efficiency
    - Translate text to all available languages at once
    - Detect if translations remain the same as the original
    - Handle technical terminology and proper nouns
    - Provide GPU-accelerated translation for fast processing
    
    The tool uses the NLLB-200-3.3B model which supports 200 languages including:
    - Major languages: English, Spanish, French, German, Chinese, Japanese, etc.
    - Regional languages: Various Arabic dialects, African languages, etc.
    - Technical and minority languages
    
    Perfect for analyzing term translatability across multiple languages.
    """
    
    inputs = {
        "action": {
            "type": "string",
            "description": "Action to perform: 'translate', 'translate_batch', 'translate_to_all', 'get_languages', 'get_language_info'"
        },
        "text": {
            "type": "string",
            "description": "Text to translate (required for translation actions)",
            "nullable": True
        },
        "texts": {
            "type": "string", 
            "description": "JSON string of list of texts for batch translation",
            "nullable": True
        },
        "src_lang": {
            "type": "string",
            "description": "Source language code in NLLB format (e.g., 'eng_Latn' for English)",
            "nullable": True
        },
        "tgt_lang": {
            "type": "string", 
            "description": "Target language code in NLLB format (e.g., 'spa_Latn' for Spanish)",
            "nullable": True
        },
        "lang_code": {
            "type": "string",
            "description": "Language code to get information about",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, device: str = "auto", batch_size: int = 8):
        """Initialize the NLLB translation tool"""
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.nllb_tool = None
        self._initialize_tool()
    
    def _initialize_tool(self):
        """Initialize the underlying NLLB translation tool"""
        try:
            print(f"🔧 Initializing NLLB translation tool...")
            self.nllb_tool = NLLBTranslationTool(
                device=self.device,
                batch_size=self.batch_size
            )
            print(f"✅ NLLB translation tool initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize NLLB tool: {e}")
            self.nllb_tool = None
    
    def forward(self, action: str, text: str = "", texts: str = "", src_lang: str = "", 
                tgt_lang: str = "", lang_code: str = "") -> str:
        """
        Execute the requested translation operation
        
        Args:
            action: The operation to perform
            text: Single text to translate
            texts: JSON string of list of texts for batch translation
            src_lang: Source language code (NLLB format)
            tgt_lang: Target language code (NLLB format)
            lang_code: Language code for information lookup
            
        Returns:
            JSON string with results or error message
        """
        if self.nllb_tool is None:
            return json.dumps({
                "error": "NLLB translation tool not initialized. Check GPU/CUDA availability.",
                "device": self.device
            })
        
        try:
            if action == "get_languages":
                languages = self.nllb_tool.get_available_languages()
                # Group languages by script/family for better organization
                language_info = {}
                for lang_code in languages:
                    lang_name = self.nllb_tool.get_language_info(lang_code)
                    script = lang_code.split('_')[-1] if '_' in lang_code else 'Unknown'
                    if script not in language_info:
                        language_info[script] = []
                    language_info[script].append({
                        "code": lang_code,
                        "name": lang_name
                    })
                
                return json.dumps({
                    "action": "get_languages",
                    "total_languages": len(languages),
                    "languages_by_script": language_info,
                    "all_language_codes": languages,
                    "description": "All 200 languages supported by NLLB-200"
                })
            
            elif action == "get_language_info":
                if not lang_code:
                    return json.dumps({"error": "lang_code parameter is required for get_language_info"})
                
                lang_name = self.nllb_tool.get_language_info(lang_code)
                is_valid = self.nllb_tool.is_valid_language(lang_code)
                
                return json.dumps({
                    "action": "get_language_info",
                    "language_code": lang_code,
                    "language_name": lang_name,
                    "is_valid": is_valid,
                    "description": f"Information about language code {lang_code}"
                })
            
            elif action == "translate":
                if not text:
                    return json.dumps({"error": "text parameter is required for translate action"})
                if not src_lang or not tgt_lang:
                    return json.dumps({"error": "Both src_lang and tgt_lang are required for translate action"})
                
                result = self.nllb_tool.translate_text(text, src_lang, tgt_lang)
                
                return json.dumps({
                    "action": "translate",
                    "original_text": result.original_text,
                    "translated_text": result.translated_text,
                    "source_language": result.source_lang,
                    "target_language": result.target_lang,
                    "is_same_as_original": result.is_same,
                    "confidence_score": result.confidence_score,
                    "error": result.error,
                    "language_pair": f"{src_lang} -> {tgt_lang}",
                    "description": "Single text translation result"
                })
            
            elif action == "translate_batch":
                if not texts:
                    return json.dumps({"error": "texts parameter is required for translate_batch action"})
                if not src_lang or not tgt_lang:
                    return json.dumps({"error": "Both src_lang and tgt_lang are required for translate_batch action"})
                
                try:
                    text_list = json.loads(texts)
                    if not isinstance(text_list, list):
                        return json.dumps({"error": "texts parameter must be a JSON list of strings"})
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON in texts parameter: {str(e)}"})
                
                results = self.nllb_tool.translate_batch(text_list, src_lang, tgt_lang)
                
                # Prepare results summary
                translation_results = []
                same_count = 0
                error_count = 0
                
                for i, result in enumerate(results):
                    translation_results.append({
                        "index": i,
                        "original": result.original_text,
                        "translated": result.translated_text,
                        "is_same": result.is_same,
                        "error": result.error
                    })
                    
                    if result.is_same:
                        same_count += 1
                    if result.error:
                        error_count += 1
                
                return json.dumps({
                    "action": "translate_batch",
                    "language_pair": f"{src_lang} -> {tgt_lang}",
                    "total_texts": len(text_list),
                    "same_as_original": same_count,
                    "successfully_translated": len(text_list) - same_count - error_count,
                    "errors": error_count,
                    "translation_results": translation_results,
                    "description": "Batch translation results with statistics"
                })
            
            elif action == "translate_to_all":
                if not text:
                    return json.dumps({"error": "text parameter is required for translate_to_all action"})
                
                src_lang = src_lang or "eng_Latn"  # Default to English
                
                print(f"🌍 Starting translation to all languages for: '{text[:50]}...'")
                results = self.nllb_tool.translate_to_all_languages(text, src_lang)
                
                # Analyze results
                same_count = 0
                translated_count = 0
                error_count = 0
                same_languages = []
                translated_languages = []
                error_languages = []
                
                for lang_code, result in results.items():
                    if result.error:
                        error_count += 1
                        error_languages.append({
                            "code": lang_code,
                            "name": self.nllb_tool.get_language_info(lang_code),
                            "error": result.error
                        })
                    elif result.is_same:
                        same_count += 1
                        same_languages.append({
                            "code": lang_code,
                            "name": self.nllb_tool.get_language_info(lang_code),
                            "translation": result.translated_text
                        })
                    else:
                        translated_count += 1
                        translated_languages.append({
                            "code": lang_code,
                            "name": self.nllb_tool.get_language_info(lang_code),
                            "translation": result.translated_text
                        })
                
                return json.dumps({
                    "action": "translate_to_all",
                    "original_text": text,
                    "source_language": src_lang,
                    "total_target_languages": len(results),
                    "statistics": {
                        "same_as_original": same_count,
                        "successfully_translated": translated_count,
                        "errors": error_count
                    },
                    "same_languages": same_languages[:10],  # Limit for readability
                    "translated_languages": translated_languages[:10],  # Limit for readability
                    "error_languages": error_languages[:10],  # Limit for readability
                    "full_results_available": True,
                    "description": "Translation to all 200 languages with analysis"
                })
            
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["translate", "translate_batch", "translate_to_all", "get_languages", "get_language_info"]
                })
                
        except Exception as e:
            return json.dumps({
                "error": f"Error executing {action}: {str(e)}",
                "action": action
            })


class NLLBTranslationAgent:
    """
    NLLB Translation Agent powered by smolagents
    Specializes in multilingual translation analysis using NLLB-200
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", batch_size: int = 8):
        """
        Initialize the translation agent
        
        Args:
            model_name: HuggingFace model for the agent's reasoning (not translation)
            device: Device for NLLB translation model ("auto", "cuda", "cpu")
            batch_size: Batch size for translation operations
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.agent = None
        
        self._setup_model()
        self._create_agent()
    
    def _setup_model(self):
        """Set up the HuggingFace model for agent reasoning"""
        print(f"🔧 Setting up agent model: {self.model_name}...")
        
        try:
            # Use HuggingFace API model for agent reasoning
            # Try different model options for compatibility
            model_options = [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "gpt2",
                "distilgpt2"
            ]
            
            for model_id in model_options:
                try:
                    self.model = HfApiModel(model_id=model_id)
                    print(f"✅ Agent model initialized: {model_id}")
                    break
                except Exception as model_error:
                    print(f"⚠️  Failed to load {model_id}: {model_error}")
                    continue
            else:
                raise Exception("All model options failed")
            
        except Exception as e:
            print(f"❌ Failed to setup agent model: {e}")
            raise
    
    def _create_agent(self):
        """Create the smolagents CodeAgent with NLLB translation tool"""
        print("🤖 Creating NLLB translation agent...")
        
        try:
            # Create the NLLB translation tool
            nllb_tool = NLLBTranslationSmolTool(device=self.device, batch_size=self.batch_size)
            
            # Create the agent with the NLLB tool
            self.agent = CodeAgent(
                model=self.model,
                tools=[nllb_tool],
                add_base_tools=True,
                max_steps=20  # Allow more steps for complex translation operations
            )
            
            print("✅ NLLB translation agent created successfully")
            print(f"🎮 Using device: {self.device}")
            print(f"🔢 Batch size: {self.batch_size}")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            raise
    
    def run_translation_task(self, prompt: str) -> str:
        """
        Run a translation-related task using the agent
        
        Args:
            prompt: Natural language prompt describing the translation task
            
        Returns:
            Agent response
        """
        print(f"🚀 Running translation task...")
        print(f"⏱️  Started at {time.strftime('%H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            response = self.agent.run(prompt)
            
            duration = time.time() - start_time
            print(f"⏱️  Completed in {duration:.1f} seconds")
            print(f"✅ Task completed successfully")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Task failed after {duration:.1f} seconds: {e}")
            raise
    
    def analyze_term_translatability(self, term: str, source_lang: str = "eng_Latn") -> str:
        """Analyze how well a term translates across languages"""
        prompt = f"""
        Please analyze the translatability of the term "{term}" using the nllb_translator tool.
        
        Steps to follow:
        1. Use 'translate_to_all' action to translate the term to all 200 languages
        2. Analyze the results to determine:
           - How many languages keep the term the same (untranslatable/borrowed)
           - How many languages successfully translate it
           - Any error cases
        3. Provide insights about why certain languages might keep it the same
        4. Categorize the term based on translatability patterns
        
        Source language: {source_lang}
        Term to analyze: "{term}"
        
        Present the analysis with clear statistics and insights.
        """
        
        return self.run_translation_task(prompt)
    
    def compare_terms_translatability(self, terms: List[str], source_lang: str = "eng_Latn") -> str:
        """Compare translatability of multiple terms"""
        prompt = f"""
        Please compare the translatability of these terms using the nllb_translator tool:
        {json.dumps(terms, indent=2)}
        
        For each term:
        1. Use 'translate_to_all' to get translations across all languages
        2. Count how many languages keep it the same vs translate it
        3. Compare patterns between terms
        
        Then provide:
        - A ranking of terms from most to least translatable
        - Common patterns in untranslatable terms
        - Language families that tend to borrow vs translate
        - Insights about term characteristics that affect translatability
        
        Source language: {source_lang}
        
        Present as a comprehensive comparative analysis.
        """
        
        return self.run_translation_task(prompt)
    
    def get_language_overview(self) -> str:
        """Get overview of available languages"""
        prompt = """
        Please use the nllb_translator tool to provide a comprehensive overview of available languages.
        
        Use the 'get_languages' action to:
        1. Show the total number of supported languages
        2. Break down languages by script/writing system
        3. Highlight major language families represented
        4. Provide examples of interesting or unique languages included
        
        Present the information in an organized, informative way.
        """
        
        return self.run_translation_task(prompt)


def demo_nllb_agent():
    """Demonstrate the NLLB translation agent capabilities"""
    
    print("🌟 NLLB TRANSLATION AGENT DEMO")
    print("=" * 60)
    
    try:
        # Create agent
        agent = NLLBTranslationAgent(device="auto", batch_size=4)
        
        print("\n1️⃣ GETTING LANGUAGE OVERVIEW")
        print("-" * 40)
        overview = agent.get_language_overview()
        print(overview)
        
        print("\n2️⃣ ANALYZING SINGLE TERM TRANSLATABILITY")
        print("-" * 40)
        term_analysis = agent.analyze_term_translatability("computer")
        print(term_analysis)
        
        print("\n3️⃣ COMPARING MULTIPLE TERMS")
        print("-" * 40)
        terms = ["software", "algorithm", "database", "internet"]
        comparison = agent.compare_terms_translatability(terms)
        print(comparison)
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_nllb_agent()
