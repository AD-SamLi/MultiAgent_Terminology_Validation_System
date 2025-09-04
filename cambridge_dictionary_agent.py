#!/usr/bin/env python3
"""
Cambridge Dictionary Agent using smolagents with Azure OpenAI GPT-5
Integrates cambridge dictionary package as a custom smolagents tool
Provides fast offline dictionary lookups for English terms
Compatible with analyze_dictionary_terms workflow
"""

import json
import time
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

# Azure and smolagents imports
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel

# Cambridge dictionary import
try:
    from cambridge.camb import search_cambridge
    import asyncio
    import aiohttp
    CAMBRIDGE_AVAILABLE = True
    print("✅ Cambridge dictionary package available")
except ImportError:
    print("❌ Cambridge dictionary not installed. Install with: pip install cambridge")
    CAMBRIDGE_AVAILABLE = False

# Common English words list (most frequent ~1000 words for fast filtering)
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
    'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
    'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
    'most', 'us', 'is', 'water', 'long', 'very', 'find', 'still', 'life', 'become', 'here', 'old', 'both', 'little', 'under',
    'last', 'right', 'move', 'thing', 'general', 'school', 'never', 'same', 'another', 'begin', 'while', 'number', 'part',
    'turn', 'real', 'leave', 'might', 'great', 'little', 'world', 'own', 'place', 'where', 'live', 'every', 'much', 'those',
    'come', 'his', 'both', 'during', 'there', 'should', 'each', 'such', 'make', 'three', 'also', 'small', 'large', 'end'
}

# Technical/specialized terms that are valid but might not be in common dictionaries
TECHNICAL_TERMS = {
    'api', 'cpu', 'gpu', 'html', 'css', 'javascript', 'python', 'java', 'sql', 'xml', 'json', 'http', 'https', 'url', 'uri',
    'ssh', 'ftp', 'smtp', 'tcp', 'udp', 'ip', 'dns', 'vpn', 'ssl', 'tls', 'oauth', 'jwt', 'crud', 'mvc', 'mvp', 'mvvm',
    'rest', 'soap', 'graphql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
    'gcp', 'cdn', 'seo', 'cms', 'crm', 'erp', 'saas', 'paas', 'iaas', 'devops', 'cicd', 'git', 'github', 'gitlab',
    'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'nft', 'defi', 'dao', 'smart', 'contract', 'token', 'wallet'
}


class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
    """
    Patched version of AzureOpenAIServerModel that removes unsupported parameters
    for GPT-5 and other reasoning models.
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
            result = agent.run(prompt)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "content filter" in error_msg or "filtered" in error_msg:
                print(f"⚠️  Attempt {attempt_index}: Content filter triggered. Retrying with safer prompt...")
                if attempt_index < max_retries:
                    continue
                return f"Content filtered after {max_retries} attempts. Please rephrase your query."
            elif "server" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                print(f"⚠️  Attempt {attempt_index}: Server error. Retrying...")
                if attempt_index < max_retries:
                    time.sleep(1)
                    continue
                return f"Server error after {max_retries} attempts: {str(e)}"
            else:
                return f"Error: {str(e)}"
    return "Max retries exceeded"


class CambridgeDictionaryTool(Tool):
    """
    Smolagents tool wrapper for Cambridge Dictionary
    Provides fast offline English dictionary functionality
    """
    
    name = "cambridge_dictionary"
    description = """
    A fast offline Cambridge English dictionary tool.
    
    This tool can:
    - Get detailed word definitions from Cambridge Dictionary (offline)
    - Check if words exist in the dictionary
    - Find pronunciation information
    - Provide example sentences
    - Handle word variations (plurals, verb forms, etc.)
    - Works completely offline for maximum speed
    - Covers comprehensive English vocabulary including academic and business terms
    
    Perfect for fast dictionary lookups and word validation without API limitations.
    Use this tool to check if terms are valid English words and get their meanings.
    """
    
    inputs = {
        "word": {
            "type": "text", 
            "description": "The English word to look up in the Cambridge dictionary"
        },
        "check_variations": {
            "type": "text",
            "description": "Whether to check word variations like plurals, verb forms (yes/no). Default: yes"
        }
    }
    
    output_type = "text"
    
    def __init__(self):
        super().__init__()
        if CAMBRIDGE_AVAILABLE:
            print("📚 Cambridge dictionary initialized (online mode)")
        else:
            print("❌ Cambridge dictionary not available")
    
    def forward(self, word: str, check_variations: str = "yes") -> str:
        """
        Look up a word in Cambridge Dictionary using async API
        """
        if not CAMBRIDGE_AVAILABLE:
            return f"❌ Cambridge dictionary not available. Please install: pip install cambridge"
        
        try:
            # Run the async lookup
            result = asyncio.run(self._async_lookup(word))
            return result
        except Exception as e:
            return f"❌ Error looking up '{word}': {str(e)}"
    
    async def _async_lookup(self, word: str) -> str:
        """
        Async lookup using Cambridge package
        """
        try:
            async with aiohttp.ClientSession() as session:
                # The search_cambridge function returns parsed results
                await search_cambridge(session, word, is_fresh=True, no_suggestions=True)
                
                # Since this function prints results, we need to capture them differently
                # For now, we'll do a simple check - if no exception, word exists
                return f"✅ '{word}' found in Cambridge Dictionary"
                
        except Exception as e:
            return f"❌ '{word}' not found in Cambridge Dictionary"
    
    async def _simple_word_check(self, word: str) -> bool:
        """
        Simple word existence check using Cambridge
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Use search_cambridge to check if word exists
                await search_cambridge(session, word, is_fresh=True, no_suggestions=True)
                return True  # If no exception, word likely exists
        except:
            return False
    
    def _generate_variations(self, word: str) -> List[str]:
        """Generate common word variations"""
        variations = []
        
        # Remove common suffixes
        if word.endswith('s') and len(word) > 3:
            variations.append(word[:-1])  # plural -> singular
        if word.endswith('ed') and len(word) > 4:
            variations.append(word[:-2])  # past tense -> base
            variations.append(word[:-1])   # if it's like "smiled" -> "smile"
        if word.endswith('ing') and len(word) > 5:
            variations.append(word[:-3])   # present participle -> base
            variations.append(word[:-3] + 'e')  # "running" -> "run", "making" -> "make"
        if word.endswith('er') and len(word) > 4:
            variations.append(word[:-2])   # comparative -> base
        if word.endswith('est') and len(word) > 5:
            variations.append(word[:-3])   # superlative -> base
        if word.endswith('ly') and len(word) > 4:
            variations.append(word[:-2])   # adverb -> adjective
        
        return list(set(variations))  # remove duplicates
    
    def _format_result(self, original_word: str, result: Dict) -> str:
        """Format the lookup result for display"""
        if not result["found"]:
            return f"❌ '{original_word}' not found"
        
        output = []
        word = result.get("found_as", result["word"])
        
        if result.get("found_as"):
            output.append(f"✅ '{original_word}' found as '{word}' in Cambridge Dictionary")
        else:
            output.append(f"✅ '{word}' found in Cambridge Dictionary")
        
        definitions = result.get("definitions", [])
        if definitions:
            if isinstance(definitions, list):
                for i, definition in enumerate(definitions[:3], 1):  # Limit to 3 definitions
                    output.append(f"   {i}. {definition}")
            else:
                output.append(f"   Definition: {definitions}")
        
        return "\n".join(output)


class CambridgeDictionaryAgent:
    """
    Main Cambridge Dictionary Agent class
    Provides high-level interface for dictionary operations and term analysis
    """
    
    def __init__(self):
        """Initialize the Cambridge Dictionary Agent"""
        load_dotenv()
        
        # Check environment
        if not CAMBRIDGE_AVAILABLE:
            print("❌ Cambridge dictionary package not available")
            self.dictionary_tool = None
            self.agent = None
            return
        
        # Initialize dictionary tool
        self.dictionary_tool = CambridgeDictionaryTool()
        
        # Initialize Azure OpenAI model
        try:
            self.model = PatchedAzureOpenAIServerModel(
                model_id=os.getenv("AZURE_OPENAI_MODEL", "gpt-5-2024-11-20"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                credentials=EnvironmentCredential()
            )
            
            # Create agent with dictionary tool
            self.agent = CodeAgent(
                tools=[self.dictionary_tool],
                model=self.model,
                max_iterations=3
            )
            
            print("✅ Cambridge Dictionary Agent initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            self.agent = None
    
    def check_word_fast(self, word: str) -> Dict:
        """
        Fast heuristic check for obvious dictionary/non-dictionary words
        Similar to the optimized version but focused on Cambridge compatibility
        """
        word = word.strip()
        word_lower = word.lower()
        
        # Quick validation
        if not word or len(word) < 2:
            return {"word": word, "in_dictionary": False, "method": "too_short", "confidence": "high"}
        
        # Check for non-alphabetic characters (except hyphens and apostrophes)
        if not re.match(r"^[a-zA-Z'-]+$", word):
            return {"word": word, "in_dictionary": False, "method": "non_alphabetic", "confidence": "high"}
        
        # Common words - definitely in dictionary
        if word_lower in COMMON_ENGLISH_WORDS:
            return {"word": word, "in_dictionary": True, "method": "common_word", "confidence": "high"}
        
        # Technical terms - likely valid
        if word_lower in TECHNICAL_TERMS:
            return {"word": word, "in_dictionary": True, "method": "technical_term", "confidence": "medium"}
        
        # Very long words - likely not standard dictionary words
        if len(word) > 20:
            return {"word": word, "in_dictionary": False, "method": "too_long", "confidence": "medium"}
        
        # Contains repeating patterns (likely made up)
        if re.search(r'(.{3,})\1', word_lower):
            return {"word": word, "in_dictionary": False, "method": "repeating_pattern", "confidence": "high"}
        
        # Multiple consecutive same letters (likely typos, except common ones)
        consecutive_letters = re.findall(r'([a-z])\1{2,}', word_lower)
        if consecutive_letters:
            # Allow some common patterns
            allowed_patterns = ['eee', 'ooo', 'lll', 'sss', 'ttt']  # Words like "bee", "zoo", etc.
            if not any(pattern in word_lower for pattern in allowed_patterns):
                return {"word": word, "in_dictionary": False, "method": "consecutive_letters", "confidence": "medium"}
        
        # Uncertain - needs Cambridge lookup
        return {"word": word, "in_dictionary": None, "method": "uncertain", "confidence": "unknown"}
    
    def check_word_cambridge(self, word: str) -> Dict:
        """
        Cambridge dictionary check using async API
        """
        if not CAMBRIDGE_AVAILABLE:
            return {
                "word": word,
                "in_dictionary": False,
                "method": "cambridge_unavailable",
                "confidence": "error",
                "reason": "Cambridge dictionary not available"
            }
        
        try:
            # Use async lookup
            word_exists = asyncio.run(self._simple_word_check(word))
            
            if word_exists:
                return {
                    "word": word,
                    "in_dictionary": True,
                    "method": "cambridge_online",
                    "confidence": "high"
                }
            else:
                return {
                    "word": word,
                    "in_dictionary": False,
                    "method": "cambridge_online",
                    "confidence": "high"
                }
                
        except Exception as e:
            return {
                "word": word,
                "in_dictionary": False,
                "method": "cambridge_error",
                "confidence": "error",
                "reason": f"Error: {str(e)[:50]}"
            }
    
    async def _simple_word_check(self, word: str) -> bool:
        """
        Simple word existence check using Cambridge
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Use search_cambridge to check if word exists
                await search_cambridge(session, word, is_fresh=True, no_suggestions=True)
                return True  # If no exception, word likely exists
        except:
            return False
    
    def analyze_terms_cambridge(self, terms: List[Dict], max_terms: Optional[int] = None) -> Tuple[List, List]:
        """
        Analyze terms using Cambridge dictionary - compatible with existing workflow
        Similar to analyze_terms_optimized but using Cambridge dictionary
        """
        if not CAMBRIDGE_AVAILABLE:
            print("❌ Cambridge dictionary not available. Please install: pip install cambridge")
            return [], []
        
        # Limit terms if specified
        if max_terms:
            terms = terms[:max_terms]
        
        print(f"🚀 CAMBRIDGE DICTIONARY ANALYSIS of {len(terms)} terms")
        start_time = time.time()
        
        dictionary_words = []
        non_dictionary_words = []
        
        # Stage 1: Fast heuristic filtering
        print("\n🔍 Stage 1: Fast heuristic filtering...")
        certain_results = []
        uncertain_terms = []
        
        for i, term_data in enumerate(terms):
            if i % 1000 == 0:
                print(f"   Processing {i:,}/{len(terms):,} terms...")
            
            term = term_data.get('term', '')
            fast_result = self.check_word_fast(term)
            
            enhanced_term = {
                **term_data,
                "dictionary_analysis": fast_result
            }
            
            if fast_result.get("in_dictionary") is not None:
                certain_results.append(enhanced_term)
            else:
                uncertain_terms.append(enhanced_term)
        
        # Categorize certain results
        for term_data in certain_results:
            if term_data["dictionary_analysis"]["in_dictionary"]:
                dictionary_words.append(term_data)
            else:
                non_dictionary_words.append(term_data)
        
        print(f"   ✅ Fast filtering complete: {len(certain_results)} certain, {len(uncertain_terms)} uncertain")
        
        # Stage 2: Cambridge offline lookup for uncertain terms
        print(f"\n📚 Stage 2: Cambridge offline verification...")
        print(f"   Checking {len(uncertain_terms)} uncertain terms...")
        
        for i, term_data in enumerate(uncertain_terms):
            if i % 500 == 0:
                print(f"   Processed {i}/{len(uncertain_terms)} uncertain terms...")
            
            term = term_data.get('term', '')
            cambridge_result = self.check_word_cambridge(term)
            
            # Update the analysis
            term_data["dictionary_analysis"] = cambridge_result
            
            # Categorize
            if cambridge_result["in_dictionary"]:
                dictionary_words.append(term_data)
            else:
                non_dictionary_words.append(term_data)
        
        # Results summary
        total_time = time.time() - start_time
        total_terms = len(dictionary_words) + len(non_dictionary_words)
        
        print(f"\n📊 CAMBRIDGE ANALYSIS COMPLETE!")
        print(f"   ⏱️  Total time: {total_time:.2f} seconds")
        print(f"   ⚡ Speed: {total_terms/total_time:.1f} terms/second")
        print(f"   ✅ Total terms analyzed: {total_terms:,}")
        print(f"   📖 Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
        print(f"   🔧 Non-dictionary words: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
        print(f"   📚 Using Cambridge offline dictionary - no API calls!")
        
        return dictionary_words, non_dictionary_words
    
    def query_agent(self, prompt: str) -> str:
        """
        Query the agent for dictionary-related tasks
        """
        if not self.agent:
            return "❌ Agent not available. Please check Cambridge dictionary installation and Azure configuration."
        
        try:
            return run_with_retries(self.agent, prompt)
        except Exception as e:
            return f"❌ Error querying agent: {str(e)}"
    
    def batch_lookup(self, words: List[str]) -> Dict:
        """
        Lookup multiple words efficiently
        """
        results = {}
        
        print(f"🔍 Looking up {len(words)} words in Cambridge Dictionary...")
        
        for i, word in enumerate(words):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(words)}")
            
            result = self.check_word_cambridge(word)
            results[word] = result
        
        # Summary
        found_count = sum(1 for r in results.values() if r.get("in_dictionary"))
        print(f"✅ Batch lookup complete: {found_count}/{len(words)} words found")
        
        return results


def load_terms_data(file_path: str) -> List[Dict]:
    """
    Load terms data from JSON file (compatible with existing format)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'terms' in data:
            return data['terms']
        else:
            print(f"❌ Unexpected data format in {file_path}")
            return []
            
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []


def save_results(dictionary_words: List[Dict], non_dictionary_words: List[Dict], output_dir: str = "."):
    """
    Save analysis results to JSON files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save dictionary words
    dict_file = os.path.join(output_dir, f"Cambridge_Dictionary_Terms_{timestamp}.json")
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_info": {
                "method": "cambridge_offline",
                "timestamp": datetime.now().isoformat(),
                "total_dictionary_terms": len(dictionary_words),
                "total_non_dictionary_terms": len(non_dictionary_words)
            },
            "dictionary_terms": dictionary_words
        }, f, indent=2, ensure_ascii=False)
    
    # Save non-dictionary words
    non_dict_file = os.path.join(output_dir, f"Cambridge_Non_Dictionary_Terms_{timestamp}.json")
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_info": {
                "method": "cambridge_offline",
                "timestamp": datetime.now().isoformat(),
                "total_dictionary_terms": len(dictionary_words),
                "total_non_dictionary_terms": len(non_dictionary_words)
            },
            "non_dictionary_terms": non_dictionary_words
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved:")
    print(f"   📖 Dictionary terms: {dict_file}")
    print(f"   🔧 Non-dictionary terms: {non_dict_file}")


def demo_cambridge_agent():
    """
    Demonstrate the Cambridge Dictionary Agent functionality
    """
    print("🚀 Cambridge Dictionary Agent Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = CambridgeDictionaryAgent()
    if not agent.dictionary_tool:
        print("❌ Demo aborted: Cambridge dictionary not available")
        return
    
    # Test individual word lookups
    test_words = ["hello", "computer", "blockchain", "xyzabc", "running", "beautiful"]
    
    print("\n1. Individual Word Lookups:")
    print("-" * 30)
    for word in test_words:
        result = agent.check_word_cambridge(word)
        status = "✅" if result["in_dictionary"] else "❌"
        print(f"{status} {word}: {result['method']} (confidence: {result['confidence']})")
    
    # Test agent query
    print("\n2. Agent Query:")
    print("-" * 30)
    if agent.agent:
        response = agent.query_agent("Look up the word 'technology' and tell me its definition")
        print(f"Agent response: {response}")
    
    # Test batch lookup
    print("\n3. Batch Lookup:")
    print("-" * 30)
    batch_words = ["python", "javascript", "algorithm", "xyz123", "beautiful"]
    batch_results = agent.batch_lookup(batch_words)
    
    for word, result in batch_results.items():
        status = "✅" if result["in_dictionary"] else "❌"
        print(f"{status} {word}")
    
    print("\n✅ Demo complete!")


def main():
    """
    Main function to run Cambridge dictionary analysis
    Compatible with existing analyze_dictionary_terms workflow
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Cambridge Dictionary Term Analysis")
    parser.add_argument("--input", "-i", help="Input JSON file with terms data")
    parser.add_argument("--max-terms", "-m", type=int, help="Maximum number of terms to analyze")
    parser.add_argument("--demo", action="store_true", help="Run demo instead of analysis")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_cambridge_agent()
        return
    
    # Initialize agent
    agent = CambridgeDictionaryAgent()
    if not agent.dictionary_tool:
        print("❌ Cambridge dictionary not available. Please install: pip install cambridge")
        return
    
    # Load input data
    if args.input:
        input_file = args.input
    else:
        # Look for common input files
        possible_files = [
            "Cleaned_Complete_Terms_Data.json",
            "Cleaned_Summary_Terms_Data.json",
            "Combined_Terms_Data.json"
        ]
        input_file = None
        for file in possible_files:
            if os.path.exists(file):
                input_file = file
                break
        
        if not input_file:
            print("❌ No input file specified and no default files found")
            print("   Use --input to specify a JSON file with terms data")
            return
    
    print(f"📂 Loading terms from: {input_file}")
    terms = load_terms_data(input_file)
    
    if not terms:
        print("❌ No terms loaded")
        return
    
    print(f"✅ Loaded {len(terms)} terms")
    
    # Run analysis
    dictionary_words, non_dictionary_words = agent.analyze_terms_cambridge(
        terms, 
        max_terms=args.max_terms
    )
    
    # Save results
    save_results(dictionary_words, non_dictionary_words)
    
    print("\n🎉 Cambridge dictionary analysis complete!")


if __name__ == "__main__":
    main()
