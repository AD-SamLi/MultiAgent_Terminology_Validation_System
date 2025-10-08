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
from datetime import datetime, timedelta
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv

# Import smolagents components
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel

# Import the terminology tool
from src.tools.terminology_tool import TerminologyTool


class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
    """
    Patched version of AzureOpenAIServerModel that removes unsupported parameters
    for GPT-5 and other reasoning models, and includes token refresh capabilities.
    
    GPT-5 and O-series models don't support the 'stop' parameter that smolagents
    tries to use by default. This wrapper filters it out and handles token refresh.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._credential = EnvironmentCredential()
        self._token_expiry = None
        # Immediately acquire a fresh token on initialization
        print("[CONFIG] [TerminologyAgent] Initializing with fresh token...")
        self._force_token_refresh()

    def _refresh_token_if_needed(self):
        """Check if token needs refresh and refresh if necessary"""
        try:
            # Always refresh if we don't have a token or expiry
            if not hasattr(self, '_token_expiry') or self._token_expiry is None or not hasattr(self, 'api_key'):
                print("[CONFIG] [TerminologyAgent] No token found, acquiring new token...")
                return self._force_token_refresh()

            # Check if token expires in the next 10 minutes (increased buffer)
            if datetime.now() + timedelta(minutes=10) >= self._token_expiry:
                print("[CONFIG] [TerminologyAgent] Token expiring soon, refreshing...")
                return self._force_token_refresh()
            
            return True
        except Exception as e:
            print(f"[ERROR] [TerminologyAgent] Token check failed: {e}")
            return self._force_token_refresh()

    def _force_token_refresh(self):
        """Force token refresh regardless of current state"""
        try:
            print("[CONFIG] [TerminologyAgent] Forcing token refresh...")
            token_result = self._credential.get_token("https://cognitiveservices.azure.com/.default")
            self.api_key = token_result.token
            self._token_expiry = datetime.fromtimestamp(token_result.expires_on)
            print(f"[OK] [TerminologyAgent] Token refreshed successfully, expires at {self._token_expiry.strftime('%H:%M:%S')}")
            return True
        except Exception as e:
            print(f"[ERROR] [TerminologyAgent] Token refresh failed: {e}")
            return False


    def _prepare_completion_kwargs(self, *args, **kwargs):
        # Check and refresh token if needed
        self._refresh_token_if_needed()
        
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


def is_token_expired(error_msg: str) -> bool:
    """Check if error is due to expired token"""
    token_indicators = [
        "401", "unauthorized", "access token", "expired", "invalid audience",
        "token is missing", "authentication failed", "credential"
    ]
    return any(indicator in error_msg.lower() for indicator in token_indicators)

def is_server_error(error_msg: str) -> bool:
    """Check if error is a server error"""
    server_indicators = [
        "500", "502", "503", "504", "internal server error", 
        "bad gateway", "service unavailable", "gateway timeout"
    ]
    return any(indicator in error_msg.lower() for indicator in server_indicators)

def is_content_filter_error(error_msg: str) -> bool:
    """Check if error is due to content filtering"""
    filter_indicators = [
        "content_filter", "responsibleaipolicyviolation", "jailbreak"
    ]
    return any(indicator in error_msg.lower() for indicator in filter_indicators)

def run_with_retries(agent, prompt: str, max_retries: int = 3):
    """Run agent with intelligent retry logic with cleanup on final failure"""
    for attempt in range(max_retries):
        try:
            # ALWAYS force token refresh before each attempt for maximum reliability
            print(f"[CONFIG] [TerminologyAgent] Attempt {attempt + 1}/{max_retries}: Ensuring fresh token...")
            if hasattr(agent, 'model') and hasattr(agent.model, '_force_token_refresh'):
                agent.model._force_token_refresh()
            elif hasattr(agent, 'refresh_model_token'):
                agent.refresh_model_token()
            
            return agent.run(prompt)
            
        except Exception as exc:
            error_msg = str(exc)
            print(f"[WARNING] [TerminologyAgent] Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt == max_retries - 1:
                print(f"[ERROR] [TerminologyAgent] All {max_retries} attempts failed. Returning clean failure result.")
                # Return a clean failure result instead of raising exception
                return f"Analysis failed after {max_retries} attempts due to persistent authentication issues. Term requires manual review."
            
            # Determine retry delay based on error type (matching modern agent)
            if is_content_filter_error(error_msg):
                delay = 1.0  # Shorter delay for content filter errors
                print(f"[WARNING] Content filter triggered (attempt {attempt + 1}/{max_retries})")
                print("   Will retry with same prompt - sometimes filters are inconsistent")
            elif is_token_expired(error_msg):
                delay = 4.0  # Longer delay for token issues
                print(f"[WARNING] Authentication error (attempt {attempt + 1}/{max_retries}): {exc}")
                print("   Token expired, forcing immediate refresh...")
                # Force immediate token refresh
                if hasattr(agent, 'refresh_model_token'):
                    try:
                        agent.refresh_model_token()
                        print("   [OK] Token refreshed successfully via refresh_model_token")
                        continue  # Skip the sleep for successful token refresh
                    except Exception as refresh_error:
                        print(f"   [ERROR] Token refresh failed: {refresh_error}")
                elif hasattr(agent, 'model') and hasattr(agent.model, '_force_token_refresh'):
                    if agent.model._force_token_refresh():
                        print("   [OK] Token refreshed successfully via _force_token_refresh")
                        continue  # Skip the sleep for successful token refresh
                    else:
                        print("   [ERROR] Token refresh failed")
                else:
                    # Fallback: force refresh on next call
                    if hasattr(agent, 'model') and hasattr(agent.model, '_token_expiry'):
                        agent.model._token_expiry = datetime.now()
                        print("   ⏰ Forced token expiry for next call")
            elif is_server_error(error_msg):
                delay = 2.0 * (2 ** attempt)  # Exponential backoff for server errors
                print(f"[WARNING] Server error (attempt {attempt + 1}/{max_retries}): {exc}")
            else:
                delay = 2.0 * (1.5 ** attempt)  # Standard backoff
                print(f"[WARNING] General error (attempt {attempt + 1}/{max_retries}): {exc}")
            
            print(f"[CONFIG] Retrying in {delay:.1f} seconds...")
            time.sleep(delay)


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
            print(f"[OK] Terminology tool initialized with glossary folder: {self.glossary_folder}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize terminology tool: {e}")
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
        print(f"[SETUP] Setting up Azure OpenAI {self.model_name}...")
        
        # Load environment variables
        load_dotenv()
        
        try:
            # Get Azure AD token
            print("[AUTH] Getting Azure AD token...")
            credential = EnvironmentCredential()
            token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("[OK] Token acquired successfully")
            
            # Get model configuration
            config = self.model_configs[self.model_name]
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            # Create patched model for GPT-5 compatibility
            self.model = PatchedAzureOpenAIServerModel(
                model_id=config["model_id"],
                azure_endpoint=endpoint,
                api_key=token_result.token,
                api_version=config["api_version"],
                temperature=0.0,  # Deterministic for consistency
                custom_role_conversions={
                    "tool-call": "user",
                    "tool-response": "assistant",
                },
            )
            
            # Set token expiry time for refresh mechanism
            self.model._token_expiry = datetime.fromtimestamp(token_result.expires_on)
            
            print(f"[OK] Model {self.model_name} initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup model: {e}")
            raise
    
    def refresh_model_token(self):
        """Recreate the model with a fresh token (matching modern_terminology_review_agent.py)"""
        try:
            print("[CONFIG] [TerminologyAgent] Recreating model with fresh token...")
            
            # Get fresh token
            credential = EnvironmentCredential()
            token_result = credential.get_token("https://cognitiveservices.azure.com/.default")
            print("[OK] [TerminologyAgent] Fresh token acquired")
            
            # Get model configuration
            config = self.model_configs[self.model_name]
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            # Create completely new model instance with fresh token
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
            
            # Set token expiry time for refresh mechanism
            self.model._token_expiry = datetime.fromtimestamp(token_result.expires_on)
            print(f"[OK] [TerminologyAgent] Model recreated with fresh token, expires at {self.model._token_expiry.strftime('%H:%M:%S')}")
            
            # Update the agent with the new model
            if hasattr(self, 'agent') and self.agent:
                self.agent.model = self.model
                print("[OK] [TerminologyAgent] Agent updated with new model")
                
        except Exception as e:
            print(f"[ERROR] [TerminologyAgent] Failed to refresh model token: {e}")
            raise
    
    def _create_agent(self):
        """Create the smolagents CodeAgent with terminology tool"""
        print("[AI] Creating terminology agent...")
        
        try:
            # Create the terminology tool
            terminology_tool = TerminologySmolTool(self.glossary_folder)
            
            # Create the agent with the terminology tool
            self.agent = CodeAgent(
                model=self.model,
                tools=[terminology_tool],
                add_base_tools=True,  # Include base tools like web search
                max_steps=15  # Allow more steps for complex terminology operations
            )
            
            print("[OK] Terminology agent created successfully")
            print(f"[FOLDER] Using glossary folder: {self.glossary_folder}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create agent: {e}")
            raise
    
    def run_terminology_task(self, prompt: str) -> str:
        """
        Run a terminology-related task using the agent
        
        Args:
            prompt: Natural language prompt describing the terminology task
            
        Returns:
            Agent response
        """
        print(f"[START] Running terminology task: {prompt[:100]}...")
        print(f"[TIME]  Started at {time.strftime('%H:%M:%S')}")
        
        if self.model_name == "gpt-5":
            print("ℹ️  Using GPT-5 - reasoning may take 30-60+ seconds...")
        
        start_time = time.time()
        
        try:
            # Run the agent with retry logic
            response = run_with_retries(self.agent, prompt, max_retries=3)
            
            duration = time.time() - start_time
            print(f"[TIME]  Completed in {duration:.1f} seconds")
            print(f"[OK] Task completed successfully")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"[ERROR] Task failed after {duration:.1f} seconds: {e}")
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
    
    print("[*] TERMINOLOGY AGENT DEMO")
    print("=" * 60)
    
    # Initialize the agent
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"[FOLDER] Using glossary folder: {glossary_folder}")
    
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
        
        print("\n[SUCCESS] DEMO COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_terminology_agent()
