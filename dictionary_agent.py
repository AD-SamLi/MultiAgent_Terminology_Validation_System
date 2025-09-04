#!/usr/bin/env python3
"""
Dictionary Agent using smolagents with Azure OpenAI GPT-5
Integrates PyMultiDictionary as a custom smolagents tool
Provides meanings, synonyms, antonyms, and translations in 20+ languages
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Union
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv

# Import smolagents components
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel

# Import PyMultiDictionary
try:
    from PyMultiDictionary import MultiDictionary, DICT_EDUCALINGO, DICT_MW, DICT_SYNONYMCOM, DICT_THESAURUS
except ImportError:
    print("❌ PyMultiDictionary not installed. Install with: pip install PyMultiDictionary")
    sys.exit(1)


class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
    """
    Patched version of AzureOpenAIServerModel that removes unsupported parameters
    for GPT-5 and other reasoning models.
    
    GPT-5 and O-series models don't support the 'stop' parameter that smolagents
    tries to use by default. This wrapper filters it out.
    """
    def _prepare_completion_kwargs(self, *args, **kwargs):
        completion_kwargs = super()._prepare_completion_kwargs(*args, **kwargs)
        # Remove unsupported params for reasoning models
        if 'stop' in completion_kwargs:
            del completion_kwargs['stop']
        model_id_str = getattr(self, "model_id", "") or ""
        if isinstance(model_id_str, str) and model_id_str.startswith("gpt-5"):
            # Remove temperature which is restricted to default for reasoning models
            if "temperature" in completion_kwargs:
                del completion_kwargs["temperature"]
            # Convert max_tokens -> max_completion_tokens if present
            if "max_tokens" in completion_kwargs:
                completion_kwargs["max_completion_tokens"] = completion_kwargs.pop("max_tokens")
            # Set a conservative default if not provided
            completion_kwargs.setdefault("max_completion_tokens", 128)
        return completion_kwargs


def run_with_retries(agent, prompt: str, max_retries: int = 2):
    """Run agent with retry logic for transient server errors and content filter issues."""
    for attempt_index in range(1, max_retries + 1):
        try:
            return agent.run(prompt)
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            is_server_error = (
                "server had an error" in error_text.lower()
                or " 500 " in f" {error_text} "
                or "500 internal server error" in error_text.lower()
            )
            is_content_filter = (
                "content_filter" in error_text.lower()
                or "responsibleaipolicyviolation" in error_text.lower()
                or "jailbreak" in error_text.lower()
            )
            
            if (is_server_error or is_content_filter) and attempt_index < max_retries:
                backoff_seconds = attempt_index * 3
                if is_content_filter:
                    print(f"⚠️ Content filter triggered (attempt {attempt_index}/{max_retries})")
                    print("   Will retry with same prompt - sometimes filters are inconsistent")
                else:
                    print(f"⚠️ Server error (attempt {attempt_index}/{max_retries}): {exc}")
                print(f"⏳ Retrying in {backoff_seconds}s...")
                time.sleep(backoff_seconds)
                continue
            raise


class DictionarySmolTool(Tool):
    """
    Smolagents tool wrapper for PyMultiDictionary
    Provides comprehensive dictionary functionality to agents
    """
    
    name = "multilingual_dictionary"
    description = """
    A comprehensive multilingual dictionary tool powered by PyMultiDictionary.
    
    This tool can:
    - Get word meanings in 20+ languages using multiple dictionary sources
    - Find synonyms and antonyms for words
    - Translate words between languages
    - Support multiple dictionary sources (Educalingo, Merriam-Webster, Synonym.com, Thesaurus)
    - Handle 20 supported languages: Bengali, German, English, Spanish, French, Hindi, 
      Italian, Japanese, Javanese, Korean, Marathi, Malay, Polish, Portuguese, 
      Romanian, Russian, Tamil, Turkish, Ukrainian, Chinese
    
    The tool provides intelligent word analysis and cross-language functionality
    for comprehensive linguistic support.
    """
    
    inputs = {
        "action": {
            "type": "string", 
            "description": "Action to perform: 'meaning', 'synonym', 'antonym', 'translate', 'get_supported_languages', 'get_supported_dictionaries'"
        },
        "word": {
            "type": "string", 
            "description": "Word to analyze (required for meaning, synonym, antonym, translate)",
            "nullable": True
        },
        "language": {
            "type": "string", 
            "description": "Source language code (e.g., 'en', 'es', 'fr', 'de', 'zh') - required for most actions",
            "nullable": True
        },
        "target_language": {
            "type": "string", 
            "description": "Target language for translation (optional - if not provided, translates to all supported languages)",
            "nullable": True
        },
        "dictionary_source": {
            "type": "string", 
            "description": "Dictionary source: 'educalingo' (default), 'merriam_webster', 'synonym_com', 'thesaurus'",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    # Supported languages mapping
    SUPPORTED_LANGUAGES = {
        'bn': 'Bengali', 'de': 'German', 'en': 'English', 'es': 'Spanish',
        'fr': 'French', 'hi': 'Hindi', 'it': 'Italian', 'ja': 'Japanese',
        'jv': 'Javanese', 'ko': 'Korean', 'mr': 'Marathi', 'ms': 'Malay',
        'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
        'ta': 'Tamil', 'tr': 'Turkish', 'uk': 'Ukrainian', 'zh': 'Chinese'
    }
    
    # Dictionary sources mapping
    DICTIONARY_SOURCES = {
        'educalingo': DICT_EDUCALINGO,
        'merriam_webster': DICT_MW,
        'synonym_com': DICT_SYNONYMCOM,
        'thesaurus': DICT_THESAURUS
    }
    
    def __init__(self):
        """Initialize the dictionary tool"""
        super().__init__()
        self.dictionary = None
        self._initialize_dictionary()
    
    def _initialize_dictionary(self):
        """Initialize the PyMultiDictionary instance"""
        try:
            self.dictionary = MultiDictionary()
            print("✅ PyMultiDictionary initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize PyMultiDictionary: {e}")
            self.dictionary = None
    
    def forward(self, action: str, word: str = "", language: str = "", 
                target_language: str = "", dictionary_source: str = "educalingo") -> str:
        """
        Execute the requested dictionary operation
        
        Args:
            action: The operation to perform
            word: Word to analyze
            language: Source language code  
            target_language: Target language for translation
            dictionary_source: Dictionary source to use
            
        Returns:
            JSON string with results or error message
        """
        if self.dictionary is None:
            return json.dumps({
                "error": "PyMultiDictionary not initialized. Please install PyMultiDictionary.",
                "install_command": "pip install PyMultiDictionary"
            })
        
        try:
            if action == "get_supported_languages":
                return json.dumps({
                    "action": "get_supported_languages",
                    "supported_languages": self.SUPPORTED_LANGUAGES,
                    "total_languages": len(self.SUPPORTED_LANGUAGES),
                    "description": "All supported language codes and their full names"
                })
            
            elif action == "get_supported_dictionaries":
                return json.dumps({
                    "action": "get_supported_dictionaries",
                    "supported_dictionaries": {
                        "educalingo": "Meaning, synonym, translation for all languages",
                        "merriam_webster": "Meanings (English only) - Merriam-Webster",
                        "synonym_com": "Synonyms and Antonyms (English only)",
                        "thesaurus": "Synonyms (English only)"
                    },
                    "default": "educalingo",
                    "description": "Available dictionary sources with their capabilities"
                })
            
            elif action == "meaning":
                if not word:
                    return json.dumps({"error": "Word parameter is required for meaning action"})
                if not language:
                    return json.dumps({"error": "Language parameter is required for meaning action"})
                
                # Validate language
                if language not in self.SUPPORTED_LANGUAGES:
                    return json.dumps({
                        "error": f"Unsupported language: {language}",
                        "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                    })
                
                # Get dictionary source
                dict_source = self.DICTIONARY_SOURCES.get(dictionary_source, DICT_EDUCALINGO)
                
                try:
                    meaning_result = self.dictionary.meaning(language, word, dictionary=dict_source)
                    
                    return json.dumps({
                        "action": "meaning",
                        "word": word,
                        "language": language,
                        "language_name": self.SUPPORTED_LANGUAGES[language],
                        "dictionary_source": dictionary_source,
                        "meaning": meaning_result,
                        "description": f"Meaning of '{word}' in {self.SUPPORTED_LANGUAGES[language]}"
                    })
                    
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to get meaning for '{word}': {str(e)}",
                        "word": word,
                        "language": language
                    })
            
            elif action == "synonym":
                if not word:
                    return json.dumps({"error": "Word parameter is required for synonym action"})
                if not language:
                    return json.dumps({"error": "Language parameter is required for synonym action"})
                
                # Validate language
                if language not in self.SUPPORTED_LANGUAGES:
                    return json.dumps({
                        "error": f"Unsupported language: {language}",
                        "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                    })
                
                try:
                    synonyms = self.dictionary.synonym(language, word)
                    
                    return json.dumps({
                        "action": "synonym",
                        "word": word,
                        "language": language,
                        "language_name": self.SUPPORTED_LANGUAGES[language],
                        "synonyms": synonyms if synonyms else [],
                        "synonym_count": len(synonyms) if synonyms else 0,
                        "description": f"Synonyms for '{word}' in {self.SUPPORTED_LANGUAGES[language]}"
                    })
                    
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to get synonyms for '{word}': {str(e)}",
                        "word": word,
                        "language": language
                    })
            
            elif action == "antonym":
                if not word:
                    return json.dumps({"error": "Word parameter is required for antonym action"})
                if not language:
                    return json.dumps({"error": "Language parameter is required for antonym action"})
                
                # Note: Antonyms are primarily supported for English
                if language != "en":
                    return json.dumps({
                        "warning": "Antonyms are primarily supported for English language",
                        "word": word,
                        "language": language,
                        "antonyms": []
                    })
                
                try:
                    antonyms = self.dictionary.antonym(language, word)
                    
                    return json.dumps({
                        "action": "antonym",
                        "word": word,
                        "language": language,
                        "language_name": self.SUPPORTED_LANGUAGES[language],
                        "antonyms": antonyms if antonyms else [],
                        "antonym_count": len(antonyms) if antonyms else 0,
                        "description": f"Antonyms for '{word}' in {self.SUPPORTED_LANGUAGES[language]}"
                    })
                    
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to get antonyms for '{word}': {str(e)}",
                        "word": word,
                        "language": language
                    })
            
            elif action == "translate":
                if not word:
                    return json.dumps({"error": "Word parameter is required for translate action"})
                if not language:
                    return json.dumps({"error": "Source language parameter is required for translate action"})
                
                # Validate source language
                if language not in self.SUPPORTED_LANGUAGES:
                    return json.dumps({
                        "error": f"Unsupported source language: {language}",
                        "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                    })
                
                try:
                    if target_language:
                        # Validate target language
                        if target_language not in self.SUPPORTED_LANGUAGES:
                            return json.dumps({
                                "error": f"Unsupported target language: {target_language}",
                                "supported_languages": list(self.SUPPORTED_LANGUAGES.keys())
                            })
                        
                        # Translate to specific language
                        translation = self.dictionary.translate(language, word, to=target_language)
                        
                        return json.dumps({
                            "action": "translate",
                            "word": word,
                            "source_language": language,
                            "source_language_name": self.SUPPORTED_LANGUAGES[language],
                            "target_language": target_language,
                            "target_language_name": self.SUPPORTED_LANGUAGES[target_language],
                            "translation": translation,
                            "description": f"Translation of '{word}' from {self.SUPPORTED_LANGUAGES[language]} to {self.SUPPORTED_LANGUAGES[target_language]}"
                        })
                    else:
                        # Translate to all supported languages
                        translations = self.dictionary.translate(language, word)
                        
                        return json.dumps({
                            "action": "translate",
                            "word": word,
                            "source_language": language,
                            "source_language_name": self.SUPPORTED_LANGUAGES[language],
                            "translations": translations if translations else {},
                            "translation_count": len(translations) if translations else 0,
                            "description": f"Translations of '{word}' from {self.SUPPORTED_LANGUAGES[language]} to all supported languages"
                        })
                        
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to translate '{word}': {str(e)}",
                        "word": word,
                        "source_language": language,
                        "target_language": target_language
                    })
            
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["meaning", "synonym", "antonym", "translate", "get_supported_languages", "get_supported_dictionaries"]
                })
                
        except Exception as e:
            return json.dumps({
                "error": f"Error executing {action}: {str(e)}",
                "action": action
            })


class DictionaryAgent:
    """
    Dictionary Agent powered by smolagents and Azure OpenAI GPT-5
    Provides comprehensive multilingual dictionary functionality
    """
    
    def __init__(self, model_name: str = "gpt-5"):
        """
        Initialize the dictionary agent
        
        Args:
            model_name: Azure OpenAI model to use ("gpt-5" or "gpt-4.1")
        """
        self.model_name = model_name
        self.model = None
        self.agent = None
        
        # Model configurations
        self.model_configs = {
            "gpt-5": {"model_id": "gpt-5", "api_version": "2025-04-01-preview"},
            "gpt-4.1": {"model_id": "gpt-4.1", "api_version": "2024-08-01-preview"}
        }
        
        self._setup_model()
        self._create_agent()
    
    def _setup_model(self):
        """Set up the Azure OpenAI model"""
        print(f"🔧 Setting up Azure OpenAI {self.model_name}...")
        
        # Load environment variables
        load_dotenv()
        
        try:
            # Get Azure AD token
            print("🔐 Getting Azure AD token...")
            credential = EnvironmentCredential()
            token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("✅ Token acquired successfully")
            
            # Get model configuration
            config = self.model_configs[self.model_name]
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            # Create patched model for GPT-5 compatibility
            self.model = PatchedAzureOpenAIServerModel(
                model_id=config["model_id"],
                azure_endpoint=endpoint,
                api_key=token_result.token,
                api_version=config["api_version"],
                custom_role_conversions={
                    "tool-call": "user",
                    "tool-response": "assistant",
                },
            )
            
            print(f"✅ Model {self.model_name} initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup model: {e}")
            raise
    
    def _create_agent(self):
        """Create the smolagents CodeAgent with dictionary tool"""
        print("🤖 Creating dictionary agent...")
        
        try:
            # Create the dictionary tool
            dictionary_tool = DictionarySmolTool()
            
            # Create the agent with the dictionary tool
            self.agent = CodeAgent(
                model=self.model,
                tools=[dictionary_tool],
                add_base_tools=True,  # Include base tools like web search
                max_steps=15,  # Allow more steps for complex dictionary operations
                verbose=True
            )
            
            print("✅ Dictionary agent created successfully")
            print("📖 Multilingual dictionary functionality ready")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            raise
    
    def run_dictionary_task(self, prompt: str) -> str:
        """
        Run a dictionary-related task using the agent
        
        Args:
            prompt: Natural language prompt describing the dictionary task
            
        Returns:
            Agent response
        """
        print(f"🚀 Running dictionary task: {prompt[:100]}...")
        print(f"⏱️  Started at {time.strftime('%H:%M:%S')}")
        
        if self.model_name == "gpt-5":
            print("ℹ️  Using GPT-5 - reasoning may take 30-60+ seconds...")
        
        start_time = time.time()
        
        try:
            # Run the agent with retry logic
            response = run_with_retries(self.agent, prompt, max_retries=3)
            
            duration = time.time() - start_time
            print(f"⏱️  Completed in {duration:.1f} seconds")
            print(f"✅ Task completed successfully")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Task failed after {duration:.1f} seconds: {e}")
            raise
    
    def get_word_meaning(self, word: str, language: str = "en", dictionary_source: str = "educalingo") -> str:
        """Get comprehensive meaning of a word"""
        prompt = f"""
        Please use the multilingual_dictionary tool to get the meaning of the word "{word}" in {language}.
        
        Use the dictionary source "{dictionary_source}" and provide:
        1. The complete meaning/definition
        2. Word types (noun, verb, adjective, etc.) if available
        3. Any additional context or examples
        
        Present the information clearly and comprehensively.
        """
        
        return self.run_dictionary_task(prompt)
    
    def find_synonyms_antonyms(self, word: str, language: str = "en") -> str:
        """Find synonyms and antonyms for a word"""
        prompt = f"""
        Please use the multilingual_dictionary tool to find both synonyms and antonyms for the word "{word}" in {language}.
        
        Steps:
        1. First get synonyms using the 'synonym' action
        2. Then get antonyms using the 'antonym' action (note: primarily works for English)
        3. Present both lists clearly
        4. If antonyms aren't available for the language, explain why
        
        Organize the results in a clear, structured format.
        """
        
        return self.run_dictionary_task(prompt)
    
    def translate_word(self, word: str, source_lang: str, target_lang: str = None) -> str:
        """Translate a word between languages"""
        if target_lang:
            prompt = f"""
            Please use the multilingual_dictionary tool to translate the word "{word}" from {source_lang} to {target_lang}.
            
            Provide:
            1. The direct translation
            2. Language names for both source and target
            3. Any additional context about the translation
            
            Present the translation clearly.
            """
        else:
            prompt = f"""
            Please use the multilingual_dictionary tool to translate the word "{word}" from {source_lang} to all available languages.
            
            Provide:
            1. Translations to all supported languages
            2. Count of successful translations
            3. Organize by language families or regions if helpful
            
            Present the translations in an organized, easy-to-read format.
            """
        
        return self.run_dictionary_task(prompt)
    
    def comprehensive_word_analysis(self, word: str, language: str = "en") -> str:
        """Perform comprehensive analysis of a word"""
        prompt = f"""
        Please perform a comprehensive analysis of the word "{word}" in {language} using the multilingual_dictionary tool.
        
        Include:
        1. Complete meaning and definition
        2. Synonyms (if available)
        3. Antonyms (if available, primarily for English)
        4. Translations to key languages (at least 5 major languages)
        5. Word classification and usage notes
        
        Use multiple actions from the dictionary tool and present a complete linguistic profile of the word.
        """
        
        return self.run_dictionary_task(prompt)


def demo_dictionary_agent():
    """Demonstrate the dictionary agent capabilities"""
    
    print("📖 MULTILINGUAL DICTIONARY AGENT DEMO")
    print("=" * 60)
    
    try:
        # Create agent with GPT-5 (change to "gpt-4.1" for faster testing)
        agent = DictionaryAgent(model_name="gpt-5")
        
        print("\n1️⃣ GETTING SUPPORTED LANGUAGES AND DICTIONARIES")
        print("-" * 50)
        overview = agent.run_dictionary_task("""
        Please use the multilingual_dictionary tool to show me:
        1. All supported languages (get_supported_languages)
        2. All available dictionary sources (get_supported_dictionaries)
        
        Present this information in an organized way to help me understand the capabilities.
        """)
        print(overview)
        
        print("\n2️⃣ WORD MEANING ANALYSIS")
        print("-" * 40)
        meaning = agent.get_word_meaning("excellent", "en", "merriam_webster")
        print(meaning)
        
        print("\n3️⃣ SYNONYMS AND ANTONYMS")
        print("-" * 40)
        syn_ant = agent.find_synonyms_antonyms("beautiful", "en")
        print(syn_ant)
        
        print("\n4️⃣ WORD TRANSLATION")
        print("-" * 40)
        translation = agent.translate_word("hello", "en", "es")
        print(translation)
        
        print("\n5️⃣ COMPREHENSIVE WORD ANALYSIS")
        print("-" * 40)
        analysis = agent.comprehensive_word_analysis("innovation", "en")
        print(analysis)
        
        print("\n6️⃣ MULTILINGUAL TRANSLATION")
        print("-" * 40)
        multi_translation = agent.translate_word("friend", "en")
        print(multi_translation)
        
        print("\n7️⃣ CUSTOM DICTIONARY QUERY")
        print("-" * 40)
        custom_query = """
        I'm learning Spanish and need help with the word "hermoso". Can you:
        1. Give me the meaning in Spanish
        2. Find Spanish synonyms
        3. Translate it to English
        4. Show me related words in both languages
        
        Use the multilingual dictionary tools to provide comprehensive language learning support.
        """
        custom_response = agent.run_dictionary_task(custom_query)
        print(custom_response)
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_dictionary_agent()

