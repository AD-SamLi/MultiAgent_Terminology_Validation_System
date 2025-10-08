#!/usr/bin/env python3
"""
Fast Dictionary Agent using local word lists and NLTK
Provides ultra-fast offline dictionary lookups for English terms
Much faster than PyMultiDictionary API calls
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

# NLTK for comprehensive word lists
try:
    import nltk
    from nltk.corpus import words, wordnet
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    print("[OK] NLTK available for dictionary lookups")
except ImportError:
    print("[ERROR] NLTK not installed. Install with: pip install nltk")
    NLTK_AVAILABLE = False

# Load word lists
ENGLISH_WORDS = set()
LEMMATIZER = None

def initialize_word_lists():
    """Initialize comprehensive English word lists"""
    global ENGLISH_WORDS, LEMMATIZER
    
    if not NLTK_AVAILABLE:
        return False
    
    try:
        # Download required NLTK data
        nltk.download('words', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Load comprehensive word list
        word_list = words.words()
        ENGLISH_WORDS = set(word.lower() for word in word_list)
        
        # Initialize lemmatizer
        LEMMATIZER = WordNetLemmatizer()
        
        print(f"[OK] Loaded {len(ENGLISH_WORDS):,} English words from NLTK")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize word lists: {e}")
        return False

# Common English words list (most frequent ~2000 words for fast filtering)
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
    'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
    'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
    'most', 'us', 'is', 'water', 'long', 'very', 'find', 'still', 'life', 'become', 'here', 'old', 'both', 'little', 'under',
    'last', 'right', 'move', 'thing', 'general', 'school', 'never', 'same', 'another', 'begin', 'while', 'number', 'part',
    'turn', 'real', 'leave', 'might', 'great', 'world', 'own', 'place', 'where', 'live', 'every', 'much', 'those',
    'come', 'both', 'during', 'should', 'each', 'such', 'three', 'small', 'large', 'end', 'put', 'home', 'read', 'hand',
    'port', 'large', 'spell', 'add', 'even', 'land', 'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask',
    'men', 'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'picture', 'try', 'us', 'again', 'animal', 'point',
    'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand', 'own', 'page', 'should', 'country',
    'found', 'answer', 'school', 'grow', 'study', 'still', 'learn', 'plant', 'cover', 'food', 'sun', 'four', 'between',
    'state', 'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree', 'cross', 'farm', 'hard', 'start', 'might',
    'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'don', 'while', 'press', 'close', 'night', 'real', 'life',
    'few', 'north', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk', 'example', 'ease',
    'paper', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until', 'mile', 'river', 'car',
    'feet', 'care', 'second', 'book', 'carry', 'took', 'science', 'eat', 'room', 'friend', 'began', 'idea', 'fish',
    'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch', 'color', 'face', 'wood', 'main', 'enough',
    'plain', 'girl', 'usual', 'young', 'ready', 'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon',
    'body', 'dog', 'family', 'direct', 'pose', 'leave', 'song', 'measure', 'door', 'product', 'black', 'short', 'numeral',
    'class', 'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem',
    'piece', 'told', 'knew', 'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best', 'hour', 'better', 'during',
    'hundred', 'five', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing',
    'listen', 'six', 'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war', 'lay',
    'against', 'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'map', 'rain', 'rule',
    'govern', 'pull', 'cold', 'notice', 'voice', 'unit', 'power', 'town', 'fine', 'certain', 'fly', 'fall', 'lead', 'cry',
    'dark', 'machine', 'note', 'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field', 'rest', 'correct', 'able', 'pound',
    'done', 'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick', 'develop',
    'ocean', 'warm', 'free', 'minute', 'strong', 'special', 'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street',
    'inch', 'multiply', 'nothing', 'course', 'stay', 'wheel', 'full', 'force', 'blue', 'object', 'decide', 'surface',
    'deep', 'moon', 'island', 'foot', 'system', 'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible', 'plane',
    'stead', 'dry', 'wonder', 'laugh', 'thousands', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss',
    'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint', 'language', 'among'
}

# Technical/specialized terms that are valid
TECHNICAL_TERMS = {
    'api', 'cpu', 'gpu', 'html', 'css', 'javascript', 'python', 'java', 'sql', 'xml', 'json', 'http', 'https', 'url', 'uri',
    'ssh', 'ftp', 'smtp', 'tcp', 'udp', 'ip', 'dns', 'vpn', 'ssl', 'tls', 'oauth', 'jwt', 'crud', 'mvc', 'mvp', 'mvvm',
    'rest', 'soap', 'graphql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
    'gcp', 'cdn', 'seo', 'cms', 'crm', 'erp', 'saas', 'paas', 'iaas', 'devops', 'cicd', 'git', 'github', 'gitlab',
    'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'nft', 'defi', 'dao', 'smart', 'contract', 'token', 'wallet',
    'ai', 'ml', 'nlp', 'opencv', 'tensorflow', 'pytorch', 'sklearn', 'numpy', 'pandas', 'matplotlib', 'jupyter',
    'api', 'sdk', 'ide', 'gui', 'cli', 'ui', 'ux', 'css', 'html', 'svg', 'png', 'jpg', 'pdf', 'zip', 'tar', 'gz'
}


class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
    """
    Patched version for GPT-5 compatibility
    """
    def _prepare_completion_kwargs(self, *args, **kwargs):
        completion_kwargs = super()._prepare_completion_kwargs(*args, **kwargs)
        if 'stop' in completion_kwargs:
            del completion_kwargs['stop']
        model_id_str = getattr(self, "model_id", "") or ""
        if isinstance(model_id_str, str) and model_id_str.startswith("gpt-5"):
            if "temperature" in completion_kwargs:
                del completion_kwargs["temperature"]
            if "max_tokens" in completion_kwargs:
                completion_kwargs["max_completion_tokens"] = completion_kwargs.pop("max_tokens")
            completion_kwargs.setdefault("max_completion_tokens", 128)
        return completion_kwargs


def run_with_retries(agent, prompt: str, max_retries: int = 2):
    """Run agent with retry logic"""
    for attempt_index in range(1, max_retries + 1):
        try:
            result = agent.run(prompt)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "content filter" in error_msg or "filtered" in error_msg:
                print(f"[WARNING]  Attempt {attempt_index}: Content filter triggered. Retrying...")
                if attempt_index < max_retries:
                    continue
                return f"Content filtered after {max_retries} attempts."
            elif "server" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                print(f"[WARNING]  Attempt {attempt_index}: Server error. Retrying...")
                if attempt_index < max_retries:
                    time.sleep(1)
                    continue
                return f"Server error after {max_retries} attempts: {str(e)}"
            else:
                return f"Error: {str(e)}"
    return "Max retries exceeded"


class FastDictionaryTool(Tool):
    """
    Fast offline dictionary tool using NLTK word lists
    Much faster than API-based approaches
    """
    
    name = "fast_dictionary"
    description = """
    A ultra-fast offline English dictionary tool using NLTK word corpus.
    
    This tool can:
    - Check if words exist in comprehensive English dictionary (200,000+ words)
    - Handle word variations (plurals, verb forms, etc.) using lemmatization
    - Provide instant lookups without API limitations
    - Works completely offline for maximum speed
    - Covers comprehensive English vocabulary including technical terms
    
    Perfect for fast dictionary validation and word checking.
    Uses NLTK's words corpus and WordNet for comprehensive coverage.
    """
    
    inputs = {
        "word": {
            "type": "string", 
            "description": "The English word to check in the dictionary"
        },
        "check_variations": {
            "type": "string",
            "description": "Whether to check word variations using lemmatization (yes/no). Default: yes",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.initialized = initialize_word_lists()
        if self.initialized:
            print("[DICT] Fast dictionary initialized with NLTK word corpus")
        else:
            print("[ERROR] Fast dictionary initialization failed")
    
    def forward(self, word: str, check_variations: str = "yes") -> str:
        """
        Look up a word in the fast dictionary
        """
        if not self.initialized:
            return f"[ERROR] Dictionary not available. Please install NLTK: pip install nltk"
        
        try:
            word_clean = word.lower().strip()
            check_vars = check_variations.lower() in ['yes', 'true', '1']
            
            # Direct lookup
            if word_clean in ENGLISH_WORDS:
                return f"[OK] '{word}' found in dictionary"
            
            # Check variations if requested
            if check_vars and LEMMATIZER:
                variations = self._get_word_variations(word_clean)
                for variation in variations:
                    if variation in ENGLISH_WORDS:
                        return f"[OK] '{word}' found as '{variation}' in dictionary"
            
            return f"[ERROR] '{word}' not found in dictionary"
            
        except Exception as e:
            return f"[ERROR] Error looking up '{word}': {str(e)}"
    
    def _get_word_variations(self, word: str) -> List[str]:
        """Get word variations using lemmatization"""
        if not LEMMATIZER:
            return []
        
        variations = []
        
        # Try different POS tags
        pos_tags = ['n', 'v', 'a', 'r']  # noun, verb, adjective, adverb
        for pos in pos_tags:
            try:
                lemma = LEMMATIZER.lemmatize(word, pos)
                if lemma != word:
                    variations.append(lemma)
            except:
                pass
        
        # Also try simple rule-based variations
        if word.endswith('s') and len(word) > 3:
            variations.append(word[:-1])
        if word.endswith('ed') and len(word) > 4:
            variations.append(word[:-2])
        if word.endswith('ing') and len(word) > 5:
            variations.append(word[:-3])
        if word.endswith('ly') and len(word) > 4:
            variations.append(word[:-2])
        
        return list(set(variations))


class FastDictionaryAgent:
    """
    Fast Dictionary Agent using NLTK word corpus
    Ultra-fast alternative to PyMultiDictionary
    """
    
    def __init__(self):
        """Initialize the Fast Dictionary Agent"""
        load_dotenv()
        
        # Initialize dictionary tool
        self.dictionary_tool = FastDictionaryTool()
        
        if not self.dictionary_tool.initialized:
            print("[ERROR] Dictionary tool not available")
            self.agent = None
            return
        
        # Initialize Azure OpenAI model
        try:
            self.model = PatchedAzureOpenAIServerModel(
                model_id=os.getenv("AZURE_OPENAI_MODEL", "gpt-5-2024-11-20"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=0.0,  # Deterministic for consistency
                credentials=EnvironmentCredential()
            )
            
            # Create agent with dictionary tool
            self.agent = CodeAgent(
                tools=[self.dictionary_tool],
                model=self.model,
                max_iterations=3
            )
            
            print("[OK] Fast Dictionary Agent initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize agent: {e}")
            self.agent = None
    
    def check_word_fast(self, word: str) -> Dict:
        """
        Fast heuristic check for obvious dictionary/non-dictionary words
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
        if len(word) > 25:
            return {"word": word, "in_dictionary": False, "method": "too_long", "confidence": "medium"}
        
        # Contains repeating patterns (likely made up)
        if re.search(r'(.{3,})\1', word_lower):
            return {"word": word, "in_dictionary": False, "method": "repeating_pattern", "confidence": "high"}
        
        # Multiple consecutive same letters (likely typos)
        consecutive_letters = re.findall(r'([a-z])\1{2,}', word_lower)
        if consecutive_letters:
            return {"word": word, "in_dictionary": False, "method": "consecutive_letters", "confidence": "medium"}
        
        # Uncertain - needs NLTK lookup
        return {"word": word, "in_dictionary": None, "method": "uncertain", "confidence": "unknown"}
    
    def check_word_nltk(self, word: str) -> Dict:
        """
        NLTK dictionary check
        """
        if not self.dictionary_tool.initialized:
            return {
                "word": word,
                "in_dictionary": False,
                "method": "nltk_unavailable",
                "confidence": "error",
                "reason": "NLTK not available"
            }
        
        try:
            word_lower = word.lower().strip()
            
            # Direct lookup
            if word_lower in ENGLISH_WORDS:
                return {
                    "word": word,
                    "in_dictionary": True,
                    "method": "nltk_direct",
                    "confidence": "high"
                }
            
            # Try variations with lemmatizer
            if LEMMATIZER:
                variations = self.dictionary_tool._get_word_variations(word_lower)
                for variation in variations:
                    if variation in ENGLISH_WORDS:
                        return {
                            "word": word,
                            "in_dictionary": True,
                            "method": "nltk_variation",
                            "confidence": "high",
                            "found_as": variation
                        }
            
            return {
                "word": word,
                "in_dictionary": False,
                "method": "nltk_not_found",
                "confidence": "high"
            }
                
        except Exception as e:
            return {
                "word": word,
                "in_dictionary": False,
                "method": "nltk_error",
                "confidence": "error",
                "reason": f"Error: {str(e)[:50]}"
            }
    
    def analyze_terms_fast(self, terms: List[Dict], max_terms: Optional[int] = None) -> Tuple[List, List]:
        """
        Ultra-fast term analysis using NLTK - compatible with existing workflow
        Much faster than PyMultiDictionary API calls
        """
        if not self.dictionary_tool.initialized:
            print("[ERROR] NLTK dictionary not available. Please install: pip install nltk")
            return [], []
        
        # Limit terms if specified
        if max_terms:
            terms = terms[:max_terms]
        
        print(f"[START] FAST DICTIONARY ANALYSIS of {len(terms)} terms")
        start_time = time.time()
        
        dictionary_words = []
        non_dictionary_words = []
        
        # Stage 1: Fast heuristic filtering
        print("\n[SEARCH] Stage 1: Fast heuristic filtering...")
        certain_results = []
        uncertain_terms = []
        
        for i, term_data in enumerate(terms):
            if i % 2000 == 0:
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
        
        print(f"   [OK] Fast filtering complete: {len(certain_results)} certain, {len(uncertain_terms)} uncertain")
        
        # Stage 2: NLTK lookup for uncertain terms
        print(f"\n[DICT] Stage 2: NLTK dictionary verification...")
        print(f"   Checking {len(uncertain_terms)} uncertain terms...")
        
        for i, term_data in enumerate(uncertain_terms):
            if i % 1000 == 0:
                print(f"   Processed {i:,}/{len(uncertain_terms):,} uncertain terms...")
            
            term = term_data.get('term', '')
            nltk_result = self.check_word_nltk(term)
            
            # Update the analysis
            term_data["dictionary_analysis"] = nltk_result
            
            # Categorize
            if nltk_result["in_dictionary"]:
                dictionary_words.append(term_data)
            else:
                non_dictionary_words.append(term_data)
        
        # Results summary
        total_time = time.time() - start_time
        total_terms = len(dictionary_words) + len(non_dictionary_words)
        
        print(f"\n[STATS] FAST ANALYSIS COMPLETE!")
        print(f"   [TIME]  Total time: {total_time:.2f} seconds")
        print(f"   [FAST] Speed: {total_terms/total_time:.1f} terms/second")
        print(f"   [OK] Total terms analyzed: {total_terms:,}")
        print(f"   [BOOK] Dictionary words: {len(dictionary_words):,} ({len(dictionary_words)/total_terms*100:.1f}%)")
        print(f"   [SETUP] Non-dictionary words: {len(non_dictionary_words):,} ({len(non_dictionary_words)/total_terms*100:.1f}%)")
        print(f"   [DICT] Using NLTK offline dictionary - no API calls!")
        
        return dictionary_words, non_dictionary_words
    
    def query_agent(self, prompt: str) -> str:
        """Query the agent for dictionary-related tasks"""
        if not self.agent:
            return "[ERROR] Agent not available. Please check NLTK installation and Azure configuration."
        
        try:
            return run_with_retries(self.agent, prompt)
        except Exception as e:
            return f"[ERROR] Error querying agent: {str(e)}"
    
    def batch_lookup(self, words: List[str]) -> Dict:
        """Lookup multiple words efficiently"""
        results = {}
        
        print(f"[SEARCH] Looking up {len(words)} words in fast dictionary...")
        
        for i, word in enumerate(words):
            if i % 1000 == 0:
                print(f"   Progress: {i:,}/{len(words):,}")
            
            result = self.check_word_nltk(word)
            results[word] = result
        
        # Summary
        found_count = sum(1 for r in results.values() if r.get("in_dictionary"))
        print(f"[OK] Batch lookup complete: {found_count:,}/{len(words):,} words found")
        
        return results


def load_terms_data(file_path: str) -> List[Dict]:
    """Load terms data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'terms' in data:
            return data['terms']
        else:
            print(f"[ERROR] Unexpected data format in {file_path}")
            return []
            
    except Exception as e:
        print(f"[ERROR] Error loading {file_path}: {e}")
        return []


def save_results(dictionary_words: List[Dict], non_dictionary_words: List[Dict], output_dir: str = "."):
    """Save analysis results to JSON files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save dictionary words
    dict_file = os.path.join(output_dir, f"Fast_Dictionary_Terms_{timestamp}.json")
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_info": {
                "method": "nltk_fast_offline",
                "timestamp": datetime.now().isoformat(),
                "total_dictionary_terms": len(dictionary_words),
                "total_non_dictionary_terms": len(non_dictionary_words)
            },
            "dictionary_terms": dictionary_words
        }, f, indent=2, ensure_ascii=False)
    
    # Save non-dictionary words
    non_dict_file = os.path.join(output_dir, f"Fast_Non_Dictionary_Terms_{timestamp}.json")
    with open(non_dict_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_info": {
                "method": "nltk_fast_offline",
                "timestamp": datetime.now().isoformat(),
                "total_dictionary_terms": len(dictionary_words),
                "total_non_dictionary_terms": len(non_dictionary_words)
            },
            "non_dictionary_terms": non_dictionary_words
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved:")
    print(f"   [BOOK] Dictionary terms: {dict_file}")
    print(f"   [SETUP] Non-dictionary terms: {non_dict_file}")


def demo_fast_agent():
    """Demonstrate the Fast Dictionary Agent functionality"""
    print("[START] Fast Dictionary Agent Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = FastDictionaryAgent()
    if not agent.dictionary_tool.initialized:
        print("[ERROR] Demo aborted: Fast dictionary not available")
        return
    
    # Test individual word lookups
    test_words = ["hello", "computer", "blockchain", "xyzabc", "running", "beautiful", "cats", "programming"]
    
    print("\n1. Individual Word Lookups:")
    print("-" * 30)
    for word in test_words:
        result = agent.check_word_nltk(word)
        status = "[OK]" if result["in_dictionary"] else "[ERROR]"
        found_as = f" (as '{result.get('found_as')}')" if result.get('found_as') else ""
        print(f"{status} {word}: {result['method']} (confidence: {result['confidence']}){found_as}")
    
    # Test agent query
    print("\n2. Agent Query:")
    print("-" * 30)
    if agent.agent:
        response = agent.query_agent("Check if the word 'technology' is in the dictionary")
        print(f"Agent response: {response}")
    
    # Test batch lookup
    print("\n3. Batch Lookup Speed Test:")
    print("-" * 30)
    batch_words = ["python", "javascript", "algorithm", "xyz123", "beautiful"] * 100  # 500 words
    start_time = time.time()
    batch_results = agent.batch_lookup(batch_words)
    end_time = time.time()
    
    found_count = sum(1 for r in batch_results.values() if r.get("in_dictionary"))
    print(f"[OK] Processed {len(batch_words)} words in {end_time-start_time:.2f} seconds")
    print(f"[FAST] Speed: {len(batch_words)/(end_time-start_time):.1f} words/second")
    print(f"[STATS] Found: {found_count}/{len(batch_words)} words")
    
    print("\n[OK] Demo complete!")


def main():
    """Main function to run fast dictionary analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast Dictionary Term Analysis using NLTK")
    parser.add_argument("--input", "-i", help="Input JSON file with terms data")
    parser.add_argument("--max-terms", "-m", type=int, help="Maximum number of terms to analyze")
    parser.add_argument("--demo", action="store_true", help="Run demo instead of analysis")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_fast_agent()
        return
    
    # Initialize agent
    agent = FastDictionaryAgent()
    if not agent.dictionary_tool.initialized:
        print("[ERROR] Fast dictionary not available. Please install: pip install nltk")
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
            print("[ERROR] No input file specified and no default files found")
            print("   Use --input to specify a JSON file with terms data")
            return
    
    print(f"[DIR] Loading terms from: {input_file}")
    terms = load_terms_data(input_file)
    
    if not terms:
        print("[ERROR] No terms loaded")
        return
    
    print(f"[OK] Loaded {len(terms):,} terms")
    
    # Run analysis
    dictionary_words, non_dictionary_words = agent.analyze_terms_fast(
        terms, 
        max_terms=args.max_terms
    )
    
    # Save results
    save_results(dictionary_words, non_dictionary_words)
    
    print("\n[SUCCESS] Fast dictionary analysis complete!")


if __name__ == "__main__":
    main()
