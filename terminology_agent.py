#!/usr/bin/env python3
"""
Terminology Agent using smolagents with Azure OpenAI GPT-5
Integrates terminology_tool.py as a custom smolagents tool
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv

# Import smolagents components
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel

# Import the terminology tool
from terminology_tool import TerminologyTool


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


class TerminologySmolTool(Tool):
    """
    Smolagents tool wrapper for the TerminologyTool class
    Provides terminology management functionality to agents
    """
    
    name = "terminology_manager"
    description = """
    A comprehensive terminology management tool for multilingual content.
    
    This tool can:
    - Load and manage translation glossaries from CSV files
    - Find terminology terms used in text for specific language pairs
    - Replace/translate terms in text based on glossary entries
    - Handle DNT (Do Not Translate) terms that should remain unchanged
    - Support multiple language pairs and formats
    
    The tool loads glossaries from a specified folder structure and provides
    intelligent term matching and replacement for consistent translations.
    """
    
    inputs = {
        "action": {
            "type": "string", 
            "description": "Action to perform: 'get_languages', 'get_language_pairs', 'find_used_terms', 'replace_terms', 'get_glossary_info'"
        },
        "text": {
            "type": "string", 
            "description": "Text to analyze or process (required for 'find_used_terms' and 'replace_terms')",
            "nullable": True
        },
        "src_lang": {
            "type": "string", 
            "description": "Source language code (e.g., 'EN', 'en', 'ENG') - required for 'find_used_terms' and 'replace_terms'",
            "nullable": True
        },
        "tgt_lang": {
            "type": "string", 
            "description": "Target language code (e.g., 'CS', 'cs', 'CSY') - required for 'find_used_terms' and 'replace_terms'",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, glossary_folder: str):
        """Initialize the terminology tool with glossary folder path"""
        super().__init__()
        self.glossary_folder = glossary_folder
        self.terminology_tool = None
        self._initialize_tool()
    
    def _initialize_tool(self):
        """Initialize the underlying TerminologyTool"""
        try:
            self.terminology_tool = TerminologyTool(self.glossary_folder)
            print(f"✅ Terminology tool initialized with glossary folder: {self.glossary_folder}")
        except Exception as e:
            print(f"❌ Failed to initialize terminology tool: {e}")
            # Create a minimal tool that will report the error
            self.terminology_tool = None
    
    def forward(self, action: str, text: str = "", src_lang: str = "", tgt_lang: str = "") -> str:
        """
        Execute the requested terminology operation
        
        Args:
            action: The operation to perform
            text: Text to process (for find_used_terms, replace_terms)
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            JSON string with results or error message
        """
        if self.terminology_tool is None:
            return json.dumps({
                "error": "Terminology tool not initialized. Check glossary folder path.",
                "glossary_folder": self.glossary_folder
            })
        
        try:
            if action == "get_languages":
                languages = self.terminology_tool.get_available_languages()
                return json.dumps({
                    "action": "get_languages",
                    "available_languages": languages,
                    "count": len(languages),
                    "description": "List of all available language codes in the glossaries"
                })
            
            elif action == "get_language_pairs":
                pairs = self.terminology_tool.get_available_language_pairs()
                return json.dumps({
                    "action": "get_language_pairs", 
                    "language_pairs": pairs,
                    "count": len(pairs),
                    "description": "List of available source-target language pair combinations"
                })
            
            elif action == "get_glossary_info":
                # Get comprehensive glossary information
                languages = self.terminology_tool.get_available_languages()
                pairs = self.terminology_tool.get_available_language_pairs()
                
                # Get term counts for each language pair
                glossary_stats = {}
                for src, tgt in pairs:
                    terms = self.terminology_tool.get_relevant_terms(src, tgt)
                    glossary_stats[f"{src}-{tgt}"] = len(terms)
                
                return json.dumps({
                    "action": "get_glossary_info",
                    "summary": {
                        "total_languages": len(languages),
                        "total_language_pairs": len(pairs),
                        "glossary_folder": self.glossary_folder
                    },
                    "available_languages": languages,
                    "language_pairs": pairs,
                    "glossary_term_counts": glossary_stats,
                    "description": "Complete overview of available terminology resources"
                })
            
            elif action == "find_used_terms":
                if not text:
                    return json.dumps({"error": "Text parameter is required for find_used_terms action"})
                if not src_lang or not tgt_lang:
                    return json.dumps({"error": "Both src_lang and tgt_lang are required for find_used_terms action"})
                
                used_terms = self.terminology_tool.get_used_terms(text, src_lang, tgt_lang)
                
                # Get the glossary to show what the terms would translate to
                glossary = self.terminology_tool.get_relevant_terms(src_lang, tgt_lang)
                term_details = {}
                
                for term in used_terms:
                    if term.startswith("DNT:"):
                        # DNT term
                        clean_term = term.replace("DNT:", "")
                        term_details[term] = {
                            "type": "DNT",
                            "translation": clean_term,
                            "note": "Do Not Translate - keep as original"
                        }
                    elif term in glossary:
                        term_details[term] = {
                            "type": "glossary",
                            "translation": glossary[term],
                            "note": "Found in glossary"
                        }
                    else:
                        term_details[term] = {
                            "type": "unknown",
                            "translation": "N/A",
                            "note": "Term found but no translation available"
                        }
                
                return json.dumps({
                    "action": "find_used_terms",
                    "language_pair": f"{src_lang}-{tgt_lang}",
                    "text_analyzed": text[:100] + "..." if len(text) > 100 else text,
                    "found_terms": used_terms,
                    "term_count": len(used_terms),
                    "term_details": term_details,
                    "description": "Terms from the glossary found in the provided text"
                })
            
            elif action == "replace_terms":
                if not text:
                    return json.dumps({"error": "Text parameter is required for replace_terms action"})
                if not src_lang or not tgt_lang:
                    return json.dumps({"error": "Both src_lang and tgt_lang are required for replace_terms action"})
                
                # First get the terms that will be affected
                used_terms = self.terminology_tool.get_used_terms(text, src_lang, tgt_lang)
                
                # Perform the replacement
                translated_text = self.terminology_tool.replace_terms(text, src_lang, tgt_lang)
                
                # Count actual changes
                changes_made = text != translated_text
                
                return json.dumps({
                    "action": "replace_terms",
                    "language_pair": f"{src_lang}-{tgt_lang}",
                    "original_text": text,
                    "translated_text": translated_text,
                    "changes_made": changes_made,
                    "terms_processed": used_terms,
                    "description": "Text with terminology replacements applied based on glossary"
                })
            
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["get_languages", "get_language_pairs", "find_used_terms", "replace_terms", "get_glossary_info"]
                })
                
        except Exception as e:
            return json.dumps({
                "error": f"Error executing {action}: {str(e)}",
                "action": action
            })


class TerminologyAgent:
    """
    Terminology Agent powered by smolagents and Azure OpenAI GPT-5
    """
    
    def __init__(self, glossary_folder: str, model_name: str = "gpt-5"):
        """
        Initialize the terminology agent
        
        Args:
            glossary_folder: Path to the folder containing glossary CSV files
            model_name: Azure OpenAI model to use ("gpt-5" or "gpt-4.1")
        """
        self.glossary_folder = glossary_folder
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
        """Create the smolagents CodeAgent with terminology tool"""
        print("🤖 Creating terminology agent...")
        
        try:
            # Create the terminology tool
            terminology_tool = TerminologySmolTool(self.glossary_folder)
            
            # Create the agent with the terminology tool
            self.agent = CodeAgent(
                model=self.model,
                tools=[terminology_tool],
                add_base_tools=True,  # Include base tools like web search
                max_steps=15,  # Allow more steps for complex terminology operations
                verbose=True
            )
            
            print("✅ Terminology agent created successfully")
            print(f"📁 Using glossary folder: {self.glossary_folder}")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            raise
    
    def run_terminology_task(self, prompt: str) -> str:
        """
        Run a terminology-related task using the agent
        
        Args:
            prompt: Natural language prompt describing the terminology task
            
        Returns:
            Agent response
        """
        print(f"🚀 Running terminology task: {prompt[:100]}...")
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
    
    def get_glossary_overview(self) -> str:
        """Get an overview of available glossaries"""
        prompt = """
        Please use the terminology_manager tool to get comprehensive information about the available glossaries.
        Use the 'get_glossary_info' action to provide a complete overview including:
        - Available languages
        - Language pairs
        - Term counts for each glossary
        - Summary statistics
        
        Present the information in a clear, organized format.
        """
        
        return self.run_terminology_task(prompt)
    
    def analyze_text_terminology(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Analyze text for terminology terms"""
        prompt = f"""
        Please analyze the following text for terminology terms using the terminology_manager tool:
        
        Text: "{text}"
        Source Language: {src_lang}
        Target Language: {tgt_lang}
        
        Use the 'find_used_terms' action to:
        1. Find all glossary terms present in the text
        2. Show what each term would translate to
        3. Identify any DNT (Do Not Translate) terms
        4. Provide a summary of the terminology analysis
        
        Present the results clearly, showing both the terms found and their translations.
        """
        
        return self.run_terminology_task(prompt)
    
    def translate_with_terminology(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text applying terminology consistency"""
        prompt = f"""
        Please translate the following text while applying consistent terminology using the terminology_manager tool:
        
        Text: "{text}"
        Source Language: {src_lang}
        Target Language: {tgt_lang}
        
        Steps to follow:
        1. First use 'find_used_terms' to analyze what terminology is present
        2. Then use 'replace_terms' to apply the glossary translations
        3. Show both the original text and the terminology-consistent version
        4. Highlight what changes were made and why
        
        Provide a clear before/after comparison and explain the terminology decisions.
        """
        
        return self.run_terminology_task(prompt)


def demo_terminology_agent():
    """Demonstrate the terminology agent capabilities"""
    
    print("🌟 TERMINOLOGY AGENT DEMO")
    print("=" * 60)
    
    # Initialize the agent
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"📁 Using glossary folder: {glossary_folder}")
    
    try:
        # Create agent with GPT-5 (change to "gpt-4.1" for faster testing)
        agent = TerminologyAgent(glossary_folder, model_name="gpt-5")
        
        print("\n1️⃣ GETTING GLOSSARY OVERVIEW")
        print("-" * 40)
        overview = agent.get_glossary_overview()
        print(overview)
        
        print("\n2️⃣ ANALYZING TEXT TERMINOLOGY")
        print("-" * 40)
        sample_text = "The AutoCAD file contains multiple layers with different vertex orientations. Please compile and import the template."
        analysis = agent.analyze_text_terminology(sample_text, "EN", "CS")
        print(analysis)
        
        print("\n3️⃣ TRANSLATING WITH TERMINOLOGY")
        print("-" * 40)
        translation = agent.translate_with_terminology(sample_text, "EN", "CS")
        print(translation)
        
        print("\n4️⃣ CUSTOM TERMINOLOGY QUERY")
        print("-" * 40)
        custom_query = """
        I need help with terminology consistency for technical documentation.
        Can you:
        1. Show me what language pairs are available
        2. Check if the term 'vertex' appears in any glossaries
        3. Show me what languages have the most terminology coverage
        
        Use the terminology tools to gather this information.
        """
        custom_response = agent.run_terminology_task(custom_query)
        print(custom_response)
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_terminology_agent()
