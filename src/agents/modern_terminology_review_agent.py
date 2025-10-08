#!/usr/bin/env python3
"""
Modern Terminology Review Agent with DDGS Multi-Engine Search
Uses the latest DDGS library for robust multi-engine search capabilities
"""

import os
import sys
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv

# Import DDGS for modern multi-engine search
from ddgs import DDGS

# Import smolagents components
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel

# Import the terminology tool
from src.tools.terminology_tool import TerminologyTool


class ModernSearchEngine:
    """Modern search engine using DDGS library with multiple backend support"""
    
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        self.ddgs = DDGS()
        self.last_search_time = 0
        self.min_delay = 2  # Minimum delay between searches
        
        # Available backends in DDGS (as per documentation)
        self.available_backends = [
            'auto',      # Automatic backend selection
            'google',    # Google search
            'brave',     # Brave search
            'bing',      # Bing search
            'duckduckgo' # DuckDuckGo search
        ]
        
        # Preferred backends in order of reliability
        self.preferred_backends = ['auto', 'brave', 'google', 'bing', 'duckduckgo']
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to search engines"""
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_search_time = time.time()
    
    def search(self, query: str, preferred_backends: List[str] = None) -> Dict[str, str]:
        """Search using DDGS with multiple backend fallback"""
        if preferred_backends is None:
            preferred_backends = self.preferred_backends
        
        results = {}
        successful_searches = 0
        target_searches = min(2, len(preferred_backends))  # Try up to 2 backends
        
        for backend in preferred_backends:
            if successful_searches >= target_searches:
                break
                
            if backend not in self.available_backends:
                continue
            
            # Try each backend with retry logic
            search_successful = False
            for attempt in range(2):  # 2 attempts per backend
                try:
                    print(f"[SEARCH] Searching with backend: {backend}" + (f" (attempt {attempt + 1})" if attempt > 0 else ""))
                    self._rate_limit()
                    
                    # Use DDGS text search with specified backend
                    search_results = list(self.ddgs.text(
                        query, 
                        max_results=self.max_results,
                        backend=backend,
                        region='us-en',
                        safesearch='moderate'
                    ))
                    
                    if search_results:
                        # Format results as text
                        formatted_results = []
                        for result in search_results[:self.max_results]:
                            title = result.get('title', 'No title')
                            body = result.get('body', 'No description')
                            url = result.get('href', 'No URL')
                            formatted_results.append(f"Title: {title}\nDescription: {body}\nURL: {url}")
                        
                        results[backend] = '\n\n'.join(formatted_results)
                        successful_searches += 1
                        print(f"[OK] {backend} search successful: {len(search_results)} results")
                        search_successful = True
                        break  # Success, no need to retry
                    else:
                        if attempt == 0:
                            print(f"[WARNING] {backend} search returned no results, retrying...")
                            time.sleep(2)  # Brief delay before retry
                            continue
                        else:
                            results[backend] = f"{backend} search returned no results after retries"
                            print(f"[WARNING] {backend} search returned no results after {attempt + 1} attempts")
                        
                except Exception as e:
                    error_msg = f"{backend} search failed: {str(e)}"
                    if attempt == 0 and "rate" not in str(e).lower():  # Don't retry rate limit errors
                        print(f"[WARNING] {error_msg}, retrying...")
                        time.sleep(2)  # Brief delay before retry
                        continue
                    else:
                        results[backend] = error_msg
                        print(f"[ERROR] {error_msg}" + (f" after {attempt + 1} attempts" if attempt > 0 else ""))
                        break
        
        # If no searches succeeded, try a simplified fallback approach
        if successful_searches == 0:
            print("[CONFIG] All backends failed, trying fallback search...")
            try:
                self._rate_limit()
                # Try with minimal parameters as fallback
                fallback_results = list(self.ddgs.text(query, max_results=5))
                if fallback_results:
                    formatted_fallback = []
                    for result in fallback_results:
                        title = result.get('title', 'No title')
                        body = result.get('body', 'No description')
                        url = result.get('href', 'No URL')
                        formatted_fallback.append(f"Title: {title}\nDescription: {body}\nURL: {url}")
                    
                    results['fallback'] = '\n\n'.join(formatted_fallback)
                    print(f"[OK] Fallback search successful: {len(fallback_results)} results")
                else:
                    results['fallback'] = "Fallback search returned no results"
                    print("[WARNING] Fallback search returned no results")
            except Exception as e:
                results['fallback'] = f"Fallback search failed: {str(e)}"
                print(f"[ERROR] Fallback search failed: {str(e)}")
        
        return results


class PatchedAzureOpenAIServerModel(AzureOpenAIServerModel):
    """
    Patched version of AzureOpenAIServerModel with token refresh capabilities
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._credential = None
        self._token_expiry = None
        # Immediately acquire a fresh token on initialization
        print("[CONFIG] [ModernReviewAgent] Initializing with fresh token...")
        self._force_token_refresh()
    
    def _refresh_token_if_needed(self):
        """Check if token needs refresh"""
        try:
            # Always refresh if we don't have a token or expiry
            if not hasattr(self, '_token_expiry') or self._token_expiry is None or not hasattr(self, 'api_key'):
                print("[CONFIG] [ModernReviewAgent] No token found, acquiring new token...")
                return self._force_token_refresh()

            # Check if token expires in the next 10 minutes (increased buffer)
            if datetime.now() + timedelta(minutes=10) >= self._token_expiry:
                print("[CONFIG] [ModernReviewAgent] Token expiring soon, refreshing...")
                return self._force_token_refresh()
            
            return True
        except Exception as e:
            print(f"[ERROR] [ModernReviewAgent] Token check failed: {e}")
            return self._force_token_refresh()

    def _force_token_refresh(self):
        """Force token refresh regardless of current state"""
        try:
            print("[CONFIG] [ModernReviewAgent] Forcing token refresh...")
            if self._credential is None:
                from azure.identity import EnvironmentCredential
                self._credential = EnvironmentCredential()
            
            token_result = self._credential.get_token("https://cognitiveservices.azure.com/.default")
            self.api_key = token_result.token
            self._token_expiry = datetime.fromtimestamp(token_result.expires_on)
            print(f"[OK] [ModernReviewAgent] Token refreshed successfully, expires at {self._token_expiry.strftime('%H:%M:%S')}")
            return True
        except Exception as e:
            print(f"[ERROR] [ModernReviewAgent] Token refresh failed: {e}")
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


class ModernTerminologyReviewTool(Tool):
    name = "terminology_review_tool"
    description = "Advanced tool for validating terminology candidates with modern multi-engine web search and Autodesk context analysis"
    output_type = "string"
    
    inputs = {
        "action": {
            "type": "string", 
            "description": "Action to perform: 'validate_single' or 'validate_batch'"
        },
        "term": {
            "type": "string", 
            "description": "Single term to validate (for validate_single action)",
            "nullable": True
        },
        "terms": {
            "type": "string", 
            "description": "JSON array of terms to validate (for validate_batch action)",
            "nullable": True
        },
        "src_lang": {
            "type": "string", 
            "description": "Source language code (e.g., 'EN') - defaults to 'EN'",
            "nullable": True
        },
        "tgt_lang": {
            "type": "string",
            "description": "Target language code (e.g., 'CS', 'DE') - optional for terminology validation",
            "nullable": True
        },
        "industry_context": {
            "type": "string", 
            "description": "Industry context (e.g., 'General', 'CAD', 'Manufacturing') - defaults to 'General'",
            "nullable": True
        },
        "output_file": {
            "type": "string",
            "description": "Output JSON file path for saving results (optional)",
            "nullable": True
        },
        "original_texts": {
            "type": "string",
            "description": "JSON array of original text arrays for context analysis (optional)",
            "nullable": True
        },
        "translation_data": {
            "type": "string",
            "description": "JSON object containing translation analysis results from ultra-optimized runner (optional)",
            "nullable": True
        }
    }
    
    def __init__(self, terminology_tool: TerminologyTool, search_engine: ModernSearchEngine):
        super().__init__()
        self.terminology_tool = terminology_tool
        self.search_engine = search_engine
    
    def _search_web(self, query: str) -> tuple[str, bool]:
        """Perform web search using modern DDGS engine. Returns (result, success_flag)"""
        try:
            search_results = self.search_engine.search(query)
            
            # Check if we have any successful results
            successful_results = []
            for engine, result in search_results.items():
                if not any(fail_indicator in result.lower() for fail_indicator in 
                          ['failed', 'error', 'timeout', 'no results', 'timed out', 'connection']):
                    # Additional check: ensure result has meaningful content
                    if len(result.strip()) > 50:  # At least 50 characters of content
                        successful_results.append(result)
            
            if successful_results:
                # Clean up HTML/XML tags and problematic characters to prevent parsing errors
                cleaned_result = self._clean_search_result(successful_results[0])
                return cleaned_result, True
            else:
                # All searches failed
                failure_msg = "All web searches failed: " + "; ".join([f"{engine}: {result[:100]}" for engine, result in search_results.items()])
                return failure_msg, False
            
        except Exception as e:
            return f"Search error: {str(e)}", False
    
    def _clean_search_result(self, result: str) -> str:
        """Clean search results to prevent XML/HTML parsing errors"""
        import re
        
        # Ultra-aggressive cleaning to prevent XML parsing errors
        result = re.sub(r'<[^>]*>', '', result)  # Remove all HTML/XML tags
        result = re.sub(r'&[a-zA-Z0-9#]+;', ' ', result)  # Remove HTML entities
        result = re.sub(r'\[[^\]]*\]', '', result)  # Remove ALL bracket content including [/link]
        result = re.sub(r'\[/?[a-zA-Z0-9]+\]', '', result)  # Remove bracket tags like [/link]
        result = re.sub(r'</?[a-zA-Z0-9]+[^>]*>', '', result)  # Remove any remaining tags
        result = re.sub(r'[<>]', '', result)  # Remove any remaining < or > characters
        result = re.sub(r'[\[\]]', '', result)  # Remove any remaining [ or ] characters
        
        # Specific patterns that cause XML parsing errors
        result = re.sub(r'\[/?link\]', '', result, flags=re.IGNORECASE)  # Remove [link] and [/link] specifically
        result = re.sub(r'\[/?[a-zA-Z0-9_-]+\]', '', result)  # Remove any remaining bracket tags
        
        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = result.strip()
        
        return result
    
    def _clean_result_for_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean validation result to prevent XML parsing errors in JSON responses"""
        import copy
        
        # Create a deep copy to avoid modifying the original
        cleaned_result = copy.deepcopy(result)
        
        # Clean web research content
        if "web_research" in cleaned_result and isinstance(cleaned_result["web_research"], dict):
            for key, value in cleaned_result["web_research"].items():
                if isinstance(value, str):
                    cleaned_result["web_research"][key] = self._clean_search_result(value)
        
        # Clean context analysis if it contains problematic content
        if "context_analysis" in cleaned_result and isinstance(cleaned_result["context_analysis"], str):
            cleaned_result["context_analysis"] = self._clean_search_result(cleaned_result["context_analysis"])
        
        return cleaned_result
    
    def _clean_json_output(self, json_string: str) -> str:
        """Final cleanup of JSON output string to remove any remaining problematic patterns"""
        import re
        
        # Remove any remaining [/link] patterns that might have escaped previous cleaning
        json_string = re.sub(r'\[/?link\]', '', json_string, flags=re.IGNORECASE)
        json_string = re.sub(r'\[/?[a-zA-Z0-9_-]+\]', '', json_string)  # Remove any bracket tags
        json_string = re.sub(r'[\[\]]', '', json_string)  # Remove any remaining brackets
        
        # Clean up any malformed XML-like patterns
        json_string = re.sub(r'<[^>]*>', '', json_string)  # Remove any HTML/XML tags
        json_string = re.sub(r'&[a-zA-Z0-9#]+;', ' ', json_string)  # Remove HTML entities
        
        # Normalize whitespace
        json_string = re.sub(r'\s+', ' ', json_string)
        
        return json_string
    
    def _analyze_autodesk_context(self, term: str, src_lang: str = "EN", tgt_lang: str = None) -> Dict[str, Any]:
        """Analyze term against Autodesk glossaries for domain relevance"""
        analysis = {
            "already_exists": False,
            "duplicate_count": 0,
            "related_terms": [],
            "domain_relevance_score": 0.0,
            "confidence": 0.0
        }
        
        try:
            # Get available language pairs
            language_pairs = self.terminology_tool.get_available_language_pairs()
            
            # Check for exact matches and related terms
            for lang_pair in language_pairs:
                if isinstance(lang_pair, tuple) and len(lang_pair) >= 2:
                    src_check, tgt_check = lang_pair[0], lang_pair[1]
                elif isinstance(lang_pair, str) and '-' in lang_pair:
                    src_check, tgt_check = lang_pair.split('-', 1)
                else:
                    continue
                
                if src_check.upper() == src_lang.upper():
                    try:
                        # Get relevant terms for this language pair
                        relevant_terms = self.terminology_tool.get_relevant_terms(
                            src_lang=src_check, 
                            tgt_lang=tgt_check
                        )
                        
                        # Check for exact matches (relevant_terms is a dict with source_term as keys)
                        exact_matches = [src_term for src_term in relevant_terms.keys() if src_term.lower() == term.lower()]
                        if exact_matches:
                            analysis["already_exists"] = True
                            analysis["duplicate_count"] += len(exact_matches)
                        
                        # Check for related terms (partial matches)
                        related = [src_term for src_term in relevant_terms.keys() if 
                                 term.lower() in src_term.lower() or 
                                 src_term.lower() in term.lower()]
                        analysis["related_terms"].extend(related[:5])
                        
                    except Exception as e:
                        print(f"[WARNING] Error checking language pair {src_check}-{tgt_check}: {e}")
                        continue
            
            # Remove duplicates from related terms
            analysis["related_terms"] = list(set(analysis["related_terms"]))
            
            # Calculate domain relevance score
            if analysis["already_exists"]:
                # Minor penalty for potential duplicates (assuming pre-filtered input)
                analysis["domain_relevance_score"] -= 0.3
                analysis["confidence"] += 0.8  # High confidence in duplicate detection
            
            # Bonus for related terms (indicates domain relevance)
            if analysis["related_terms"]:
                analysis["domain_relevance_score"] += min(0.4, len(analysis["related_terms"]) * 0.1)
                analysis["confidence"] += 0.3
            
            # Industry context bonus
            industry_keywords = {
                'CAD': ['design', 'drawing', 'model', 'sketch', 'dimension', 'layer'],
                'Manufacturing': ['process', 'production', 'assembly', 'machine', 'tool', 'quality'],
                'AEC': ['building', 'construction', 'architecture', 'structure', 'civil', 'engineering']
            }
            
            term_lower = term.lower()
            for industry, keywords in industry_keywords.items():
                if any(keyword in term_lower for keyword in keywords):
                    analysis["domain_relevance_score"] += 0.2
                    break
            
            # Ensure score is within bounds
            analysis["domain_relevance_score"] = max(-1.0, min(1.0, analysis["domain_relevance_score"]))
            analysis["confidence"] = max(0.0, min(1.0, analysis["confidence"]))
            
        except Exception as e:
            print(f"[WARNING] Error in Autodesk context analysis: {e}")
            analysis["confidence"] = 0.1  # Low confidence due to error
        
        return analysis
    
    def _analyze_translation_insights(self, term: str, translation_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze translation insights to enhance validation scoring"""
        if not translation_data:
            return {
                "translation_available": False,
                "translatability_score": 0.0,
                "multilingual_validation": "No translation data available",
                "translation_confidence": 0.0,
                "language_coverage": 0,
                "translation_quality_indicators": []
            }
        
        # Extract key translation metrics
        total_languages = translation_data.get('total_languages', 0)
        translated_languages = translation_data.get('translated_languages', 0)
        same_languages = translation_data.get('same_languages', 0)
        error_languages = translation_data.get('error_languages', 0)
        translatability_score = translation_data.get('translatability_score', 0.0)
        
        # Calculate translation confidence based on multiple factors
        translation_confidence = 0.0
        
        # Factor 1: Success rate (no errors)
        if total_languages > 0:
            success_rate = (translated_languages + same_languages) / total_languages
            translation_confidence += success_rate * 0.4
        
        # Factor 2: Translatability score
        translation_confidence += translatability_score * 0.4
        
        # Factor 3: Language coverage (more languages = higher confidence)
        if total_languages >= 50:
            translation_confidence += 0.2
        elif total_languages >= 30:
            translation_confidence += 0.15
        elif total_languages >= 20:
            translation_confidence += 0.1
        
        # Quality indicators based on translation patterns
        quality_indicators = []
        
        if translatability_score > 0.8:
            quality_indicators.append("High translatability - term translates well across languages")
        elif translatability_score > 0.5:
            quality_indicators.append("Moderate translatability - some language variations")
        else:
            quality_indicators.append("Low translatability - may be language-specific or technical")
        
        if same_languages > translated_languages:
            quality_indicators.append("Term often remains unchanged - likely technical/universal")
        
        if error_languages == 0:
            quality_indicators.append("No translation errors - robust term")
        elif error_languages / total_languages > 0.2:
            quality_indicators.append("Some translation challenges detected")
        
        # Analyze sample translations for insights
        sample_translations = translation_data.get('sample_translations', {})
        if sample_translations:
            # Check for consistent patterns
            unique_translations = set(sample_translations.values())
            if len(unique_translations) == 1:
                quality_indicators.append("Highly consistent translations across languages")
            elif len(unique_translations) / len(sample_translations) < 0.3:
                quality_indicators.append("Good translation consistency")
        
        # Multilingual validation assessment
        multilingual_validation = "No assessment available"
        if translatability_score >= 0.8:
            multilingual_validation = "Excellent - Term is highly suitable for multilingual use"
        elif translatability_score >= 0.6:
            multilingual_validation = "Good - Term translates well with minor variations"
        elif translatability_score >= 0.4:
            multilingual_validation = "Fair - Some translation challenges, may need localization"
        else:
            multilingual_validation = "Limited - Term may be difficult to translate or highly technical"
        
        return {
            "translation_available": True,
            "translatability_score": round(translatability_score, 3),
            "multilingual_validation": multilingual_validation,
            "translation_confidence": round(translation_confidence, 3),
            "language_coverage": total_languages,
            "translation_metrics": {
                "total_languages": total_languages,
                "translated_languages": translated_languages,
                "same_languages": same_languages,
                "error_languages": error_languages,
                "success_rate": round((translated_languages + same_languages) / max(1, total_languages), 3)
            },
            "translation_quality_indicators": quality_indicators,
            "sample_translations": dict(list(sample_translations.items())[:5]) if sample_translations else {}
        }
    
    def _analyze_original_texts(self, original_texts: List[str], limit_for_efficiency: bool = False) -> Dict[str, Any]:
        """Analyze original texts for context clues"""
        if not original_texts:
            return {
                "product_mentions": [],
                "technical_indicators": [],
                "usage_patterns": [],
                "context_score": 0.0
            }
        
        # For extreme cases (gap filling), limit to first 10 texts to avoid max steps issues
        if limit_for_efficiency and len(original_texts) > 10:
            original_texts = original_texts[:10]
            print(f"[FAST] Limited original texts to 10 for processing efficiency")
        
        # Combine all texts for analysis
        combined_text = ' '.join(original_texts).lower()
        
        # Autodesk product mentions
        autodesk_products = ['autocad', 'inventor', 'fusion', 'maya', 'revit', '3ds max', 'civil 3d']
        product_mentions = [product for product in autodesk_products if product in combined_text]
        
        # Technical indicators
        technical_keywords = ['design', 'model', 'drawing', 'sketch', 'dimension', 'layer', 'feature', 
                            'assembly', 'component', 'manufacturing', 'engineering', 'cad', 'cam']
        technical_indicators = [keyword for keyword in technical_keywords if keyword in combined_text]
        
        # Usage patterns (extract context around terms)
        usage_patterns = []
        words = combined_text.split()
        for i, word in enumerate(words):
            if any(indicator in word for indicator in technical_indicators[:3]):  # Check first 3 indicators
                start = max(0, i-3)
                end = min(len(words), i+4)
                context = ' '.join(words[start:end])
                usage_patterns.append(context)
        
        # Calculate context score
        context_score = 0.0
        context_score += len(product_mentions) * 0.2  # Product mentions are highly relevant
        context_score += len(technical_indicators) * 0.1  # Technical terms are relevant
        context_score += len(usage_patterns) * 0.05  # Usage patterns provide context
        
        # Cap the score
        context_score = min(0.5, context_score)
        
        return {
            "product_mentions": product_mentions,
            "technical_indicators": technical_indicators,
            "usage_patterns": usage_patterns[:5],  # Limit to top 5
            "context_score": context_score
        }
    
    def _validate_single_term(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                            industry_context: str = "General", original_texts: List[str] = None,
                            limit_for_efficiency: bool = False, translation_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a single terminology candidate using modern search"""
        
        print(f"\n[SEARCH] Validating term: '{term}'")
        
        # Analyze Autodesk context
        autodesk_analysis = self._analyze_autodesk_context(term, src_lang, tgt_lang)
        
        # Analyze original texts if provided
        context_analysis = self._analyze_original_texts(original_texts or [], limit_for_efficiency)
        
        # Analyze translation insights if provided
        translation_insights = self._analyze_translation_insights(term, translation_data)
        
        # Log translation analysis
        if translation_insights["translation_available"]:
            print(f"🌍 Translation insights: {translation_insights['language_coverage']} languages, "
                  f"score: {translation_insights['translatability_score']:.3f}")
        else:
            print("🌍 No translation data available for this term")
        
        # Perform web research with multiple targeted queries
        research_queries = [
            f'"{term}" definition meaning',
            f'"{term}" {industry_context} industry terminology',
            f'"{term}" technical term explanation',
            f'"{term}" professional usage examples'
        ]
        
        # Add translation query only if target language is specified
        if tgt_lang and tgt_lang != "EN":
            research_queries.insert(2, f'"{term}" translation {tgt_lang} language')
        
        web_research = {}
        web_search_successful = True
        failed_searches = 0
        
        for i, query in enumerate(research_queries[:3]):  # Limit to 3 queries to avoid rate limits
            print(f"🌐 Research query {i+1}: {query}")
            result, success = self._search_web(query)
            web_research[f"query_{i+1}"] = result
            
            if not success:
                failed_searches += 1
                print(f"[ERROR] Search query {i+1} failed")
            else:
                print(f"[OK] Search query {i+1} successful")
            
            time.sleep(1)  # Small delay between queries
        
        # Much more lenient web search success detection
        # Focus on whether we got ANY useful information, not perfect search results
        if failed_searches >= 3:  # Only if ALL 3 searches failed completely
            web_search_successful = False
            print(f"[ERROR] Web search failed: ALL {failed_searches}/3 queries failed")
        else:
            # If ANY search succeeded, check if we got useful information
            has_useful_info = False
            total_content_length = 0
            
            for query_result in web_research.values():
                if query_result and isinstance(query_result, str):
                    # Remove common failure indicators
                    clean_result = query_result.lower()
                    if not any(fail_word in clean_result for fail_word in ['failed', 'error', 'timeout', 'no results found']):
                        total_content_length += len(query_result)
                        # Much lower threshold - even short results can be meaningful
                        if len(query_result) > 50:  # Very low threshold
                            has_useful_info = True
            
            if has_useful_info or total_content_length > 100:
                web_search_successful = True
                success_count = 3 - failed_searches
                print(f"[OK] Web search successful: {success_count}/3 queries provided useful information ({total_content_length} chars)")
            else:
                web_search_successful = False
                print(f"[ERROR] Web search failed: {failed_searches}/3 queries failed, no useful information retrieved")
        
        # Calculate overall score with enhanced translation-aware algorithm
        score = 0.0
        
        # Context score (from original texts) - up to 0.4 points (reduced to make room for translation)
        score += context_analysis["context_score"] * 0.8  # Scale down slightly
        
        # Domain relevance score - up to 0.3 points (reduced to make room for translation)
        score += max(0, autodesk_analysis["domain_relevance_score"]) * 0.3
        
        # Web research score - up to 0.25 points (reduced to make room for translation)
        web_score = 0.0
        for query_result in web_research.values():
            if query_result and len(query_result) > 100 and not any(
                fail_indicator in query_result.lower() 
                for fail_indicator in ['failed', 'error', 'timeout', 'no results']
            ):
                web_score += 0.08  # Slightly reduced per query
        score += min(0.25, web_score)
        
        # Translation insights score - up to 0.35 points (NEW)
        translation_score = 0.0
        if translation_insights["translation_available"]:
            # Base translatability score (up to 0.2 points)
            translation_score += translation_insights["translatability_score"] * 0.2
            
            # Translation confidence bonus (up to 0.1 points)
            translation_score += translation_insights["translation_confidence"] * 0.1
            
            # Language coverage bonus (up to 0.05 points)
            if translation_insights["language_coverage"] >= 50:
                translation_score += 0.05
            elif translation_insights["language_coverage"] >= 30:
                translation_score += 0.03
            elif translation_insights["language_coverage"] >= 20:
                translation_score += 0.02
            
            score += translation_score
            print(f"[TARGET] Translation contribution to score: +{translation_score:.3f}")
        else:
            print("[TARGET] No translation data - using traditional validation only")
        
        # Penalty for duplicates
        if autodesk_analysis["already_exists"]:
            score -= 0.25  # Slightly reduced penalty to balance with translation insights
        
        # Determine status based on score
        if score >= 0.8:
            status = "recommended"
        elif score >= 0.5:
            status = "needs_review"
        elif score >= 0.3:
            status = "low_priority"
        elif autodesk_analysis["already_exists"]:
            status = "unexpected_duplicate"
        else:
            status = "not_recommended"
        
        return {
            "term": term,
            "score": round(score, 3),
            "status": status,
            "autodesk_context": autodesk_analysis,
            "context_analysis": context_analysis if original_texts else "No original texts provided",
            "translation_insights": translation_insights,
            "web_research": web_research,
            "web_search_successful": web_search_successful,
            "validation_timestamp": datetime.now().isoformat(),
            "enhanced_validation": True,
            "scoring_components": {
                "context_score": round(context_analysis["context_score"] * 0.8, 3),
                "domain_relevance": round(max(0, autodesk_analysis["domain_relevance_score"]) * 0.3, 3),
                "web_research": round(min(0.25, web_score), 3),
                "translation_insights": round(translation_score if translation_insights["translation_available"] else 0.0, 3),
                "duplicate_penalty": -0.25 if autodesk_analysis["already_exists"] else 0.0
            },
            "parameters": {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "industry_context": industry_context,
                "translation_enhanced": translation_insights["translation_available"]
            }
        }
    
    def _save_results_to_file(self, results: List[Dict], output_file: str, metadata: Dict = None):
        """Save validation results to JSON file with metadata, but only if web searches were successful"""
        
        # Check if any results have failed web searches
        failed_search_count = 0
        for result in results:
            if not result.get('web_search_successful', True):
                failed_search_count += 1
        
        # Much more lenient batch-level protection - prioritize saving validation results
        # Only prevent saving in extreme cases where we have no useful data at all
        
        # Check if we have meaningful validation results regardless of web search status
        meaningful_results = 0
        for result in results:
            # Count as meaningful if we have ANY useful validation data:
            # - Autodesk context analysis (confidence > 0.1 or any related terms)
            # - Decent scoring (> 0.1)
            # - Context analysis from original texts
            # - Any structured validation data
            if (result.get('autodesk_context', {}).get('confidence', 0) > 0.1 or 
                len(result.get('autodesk_context', {}).get('related_terms', [])) > 0 or
                result.get('score', 0) > 0.1 or
                result.get('context_analysis') != "No original texts provided" or
                result.get('status') in ['recommended', 'needs_review', 'low_priority']):
                meaningful_results += 1
        
        # Only prevent saving if we have NO meaningful results AND all searches failed
        if failed_search_count == len(results) and meaningful_results == 0:
            print(f"[ERROR] Not saving {output_file}: ALL {failed_search_count} terms failed with no validation data")
            print(f"[WARNING] File not created due to complete validation failure")
            return False
        
        # Always save if we have any meaningful results, regardless of web search failures
        if meaningful_results > 0:
            if failed_search_count > 0:
                print(f"[WARNING] Saving with partial web search failures: {meaningful_results}/{len(results)} terms have meaningful validation")
                print(f"[STATS] Web search status: {len(results) - failed_search_count}/{len(results)} successful, {failed_search_count} failed")
            else:
                print(f"[OK] Saving with all web searches successful: {meaningful_results}/{len(results)} meaningful validations")
        
        # If we have some failed searches but not too many, warn but still save
        if failed_search_count > 0:
            print(f"[WARNING] Warning: {failed_search_count}/{len(results)} terms had failed web searches, but saving anyway")
        
        output_data = {
            "metadata": {
                "agent_version": "modern_terminology_review_agent_v2.0",
                "generation_timestamp": datetime.now().isoformat(),
                "total_terms": len(results),
                "failed_web_searches": failed_search_count,
                "search_engine": "DDGS Multi-Engine",
                **(metadata or {})
            },
            "results": results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"[OK] Results saved to: {output_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving results: {e}")
            return False
    
    def forward(self, action: str, term: str = "", terms: str = "", src_lang: str = "EN", 
               tgt_lang: str = None, industry_context: str = "General", output_file: str = "", 
               original_texts: str = "", translation_data: str = "") -> str:
        """Main entry point for terminology validation"""
        
        try:
            # Parse original_texts if provided
            parsed_original_texts = None
            if original_texts:
                try:
                    parsed_original_texts = json.loads(original_texts)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Warning: Could not parse original_texts JSON: {e}")
                    parsed_original_texts = None
            
            # Parse translation_data if provided
            parsed_translation_data = None
            if translation_data:
                try:
                    parsed_translation_data = json.loads(translation_data)
                    print(f"🌍 Translation data loaded for enhanced validation")
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Warning: Could not parse translation_data JSON: {e}")
                    parsed_translation_data = None
            
            if action == "validate_single":
                if not term:
                    return "Error: No term provided for validation"
                
                # Get original texts for this specific term if available
                term_original_texts = None
                if parsed_original_texts and isinstance(parsed_original_texts, list) and len(parsed_original_texts) > 0:
                    term_original_texts = parsed_original_texts[0] if isinstance(parsed_original_texts[0], list) else [str(parsed_original_texts[0])]
                
                # Check if we need efficiency mode for this term (many original texts)
                limit_efficiency = term_original_texts and len(term_original_texts) > 15
                
                # Get translation data for this specific term
                term_translation_data = None
                if parsed_translation_data:
                    # If it's a single term's translation data
                    if parsed_translation_data.get('term') == term:
                        term_translation_data = parsed_translation_data
                    # If it's a list of translation results, find matching term
                    elif isinstance(parsed_translation_data, list):
                        for trans_result in parsed_translation_data:
                            if isinstance(trans_result, dict) and trans_result.get('term') == term:
                                term_translation_data = trans_result
                                break
                
                result = self._validate_single_term(
                    term, src_lang, tgt_lang, industry_context, term_original_texts, limit_efficiency, term_translation_data
                )
                
                if output_file:
                    self._save_results_to_file([result], output_file)
                
                # Clean the result before returning as JSON to prevent XML parsing errors
                cleaned_result = self._clean_result_for_json(result)
                json_output = json.dumps(cleaned_result, indent=2)
                
                # Final pass: clean the JSON string itself to remove any remaining problematic patterns
                json_output = self._clean_json_output(json_output)
                return json_output
            
            elif action == "validate_batch":
                if not terms:
                    return "Error: No terms provided for batch validation"
                
                try:
                    terms_list = json.loads(terms)
                    if not isinstance(terms_list, list):
                        return "Error: Terms must be provided as a JSON array"
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format for terms"
                
                results = []
                for i, term_item in enumerate(terms_list):
                    # Handle both simple strings and term objects
                    if isinstance(term_item, dict):
                        current_term = term_item.get('term', str(term_item))
                    else:
                        current_term = str(term_item)
                    
                    # Get original texts for this specific term if available
                    term_original_texts = None
                    if parsed_original_texts and isinstance(parsed_original_texts, list) and i < len(parsed_original_texts):
                        if isinstance(parsed_original_texts[i], list):
                            term_original_texts = parsed_original_texts[i]
                        else:
                            term_original_texts = [str(parsed_original_texts[i])]
                    
                    # Check if we need efficiency mode for this term (many original texts)
                    limit_efficiency = term_original_texts and len(term_original_texts) > 15
                    
                    # Get translation data for this specific term
                    term_translation_data = None
                    if parsed_translation_data:
                        # If it's a list of translation results, find matching term
                        if isinstance(parsed_translation_data, list):
                            for trans_result in parsed_translation_data:
                                if isinstance(trans_result, dict) and trans_result.get('term') == current_term:
                                    term_translation_data = trans_result
                                    break
                        # If it's a single term's translation data (unlikely in batch)
                        elif parsed_translation_data.get('term') == current_term:
                            term_translation_data = parsed_translation_data
                    
                    result = self._validate_single_term(
                        current_term, src_lang, tgt_lang, industry_context, term_original_texts, limit_efficiency, term_translation_data
                    )
                    results.append(result)
                    
                    # Small delay between validations to be respectful
                    time.sleep(0.5)
                
                if output_file:
                    self._save_results_to_file(results, output_file, {
                        "batch_size": len(results),
                        "parameters": {
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "industry_context": industry_context
                        }
                    })
                
                # Clean the results before returning as JSON to prevent XML parsing errors
                cleaned_results = [self._clean_result_for_json(result) for result in results]
                
                batch_output = json.dumps({
                    "batch_summary": {
                        "total_terms": len(cleaned_results),
                        "recommended": len([r for r in cleaned_results if r["status"] == "recommended"]),
                        "needs_review": len([r for r in cleaned_results if r["status"] == "needs_review"]),
                        "low_priority": len([r for r in cleaned_results if r["status"] == "low_priority"]),
                        "not_recommended": len([r for r in cleaned_results if r["status"] == "not_recommended"])
                    },
                    "detailed_results": cleaned_results
                }, indent=2)
                
                # Final pass: clean the JSON string itself to remove any remaining problematic patterns
                batch_output = self._clean_json_output(batch_output)
                return batch_output
            
            else:
                return f"Error: Unknown action '{action}'. Use 'validate_single' or 'validate_batch'"
        
        except Exception as e:
            return f"Error in terminology validation: {str(e)}"


def is_server_error(error_msg: str) -> bool:
    """Check if error is a server-side issue"""
    server_indicators = [
        "500", "502", "503", "504", "internal server error", 
        "service unavailable", "bad gateway", "gateway timeout"
    ]
    return any(indicator in error_msg.lower() for indicator in server_indicators)

def is_content_filter_error(error_msg: str) -> bool:
    """Check if error is due to content filtering"""
    filter_indicators = [
        "content filter", "content policy", "filtered", 
        "inappropriate content", "policy violation"
    ]
    return any(indicator in error_msg.lower() for indicator in filter_indicators)

def is_xml_parsing_error(error_msg: str) -> bool:
    """Check if error is due to XML parsing issues"""
    xml_indicators = [
        "closing tag", "doesn't match any open tag", "xml", "parsing error",
        "malformed", "invalid xml", "tag mismatch"
    ]
    return any(indicator in error_msg.lower() for indicator in xml_indicators)

def is_token_expired(error_msg: str) -> bool:
    """Check if error is due to expired token"""
    token_indicators = [
        "401", "unauthorized", "access token", "expired", "invalid audience",
        "token is missing", "authentication failed", "credential"
    ]
    return any(indicator in error_msg.lower() for indicator in token_indicators)

def run_with_retries(func, max_retries: int = 5, base_delay: float = 2.0):
    """Run function with intelligent retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"[CONFIG] [ModernReviewAgent] Attempt {attempt + 1}: Executing with fresh authentication...")
            return func()
        except Exception as e:
            error_msg = str(e)
            print(f"[WARNING] Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt == max_retries - 1:
                raise e
            
            # Determine retry delay based on error type
            if is_xml_parsing_error(error_msg):
                delay = base_delay * 0.5  # Shorter delay for parsing errors
            elif is_token_expired(error_msg):
                delay = base_delay * 2  # Longer delay for token issues
            elif is_server_error(error_msg) or is_content_filter_error(error_msg):
                delay = base_delay * (2 ** attempt)  # Exponential backoff
            else:
                delay = base_delay * (1.5 ** attempt)  # Standard backoff
            
            print(f"[CONFIG] Retrying in {delay:.1f} seconds...")
            time.sleep(delay)


class ModernTerminologyReviewAgent:
    """Modern Terminology Review Agent with DDGS multi-engine search"""
    
    def __init__(self, model_name: str = "gpt-4.1"):
        load_dotenv()
        
        self.model_name = model_name
        self.model_configs = {
            "gpt-4.1": {
                "model_id": "gpt-4.1",
                "api_version": "2024-08-01-preview"
            },
            "gpt-4o": {
                "model_id": "gpt-4o",
                "api_version": "2024-10-21"
            },
            "gpt-5": {
                "model_id": "gpt-5",
                "api_version": "2025-08-07"
            }
        }
        
        # Initialize components
        self._credential = None
        self.access_token = None
        self.token_expiry = None
        self.model = None
        self.agent = None
        
        # Initialize search engine
        self.search_engine = ModernSearchEngine(max_results=20)
        
        # Setup model and agent
        self._setup_model()
        self._create_agent()
    
    def _get_fresh_token(self):
        """Get a fresh Azure AD token"""
        try:
            if not self._credential:
                self._credential = EnvironmentCredential()
            
            # Get token for Azure OpenAI
            token_result = self._credential.get_token("https://cognitiveservices.azure.com/.default")
            return token_result.token, token_result.expires_on
        except Exception as e:
            print(f"[ERROR] Failed to get fresh token: {e}")
            raise
    
    def _setup_model(self):
        """Setup the Azure OpenAI model with token management"""
        try:
            if self.model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            config = self.model_configs[self.model_name]
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")
            
            # Get fresh token
            self.access_token, self.token_expiry = self._get_fresh_token()
            
            # Create model with deterministic parameters
            self.model = PatchedAzureOpenAIServerModel(
                model_id=config["model_id"],
                azure_endpoint=endpoint,
                api_key=self.access_token,
                api_version=config["api_version"],
                temperature=0.0,  # Deterministic temperature
                top_p=1.0,  # Use full probability distribution
                frequency_penalty=0.0,  # No frequency penalty
                presence_penalty=0.0,  # No presence penalty
                seed=42,  # Fixed seed for reproducibility
                max_tokens=4000,  # Consistent response length
                custom_role_conversions={
                    "tool-call": "user",
                    "tool-response": "assistant",
                },
            )
            
            # Initialize token management in the model
            self.model._credential = self._credential
            self.model._token_expiry = datetime.now() + timedelta(seconds=self.token_expiry - int(datetime.now().timestamp()))
            
            print(f"[OK] Model {self.model_name} setup complete")
            
        except Exception as e:
            print(f"[ERROR] Failed to setup model: {e}")
            raise
    
    def refresh_model_token(self):
        """Recreate the model with a fresh token"""
        try:
            print("[CONFIG] Recreating model with fresh token...")
            new_token, new_expiry = self._get_fresh_token()
            
            # Update stored token info
            self.access_token = new_token
            self.token_expiry = new_expiry
            
            # Get model configuration
            config = self.model_configs[self.model_name]
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            
            # Create a completely new model instance with fresh token
            self.model = PatchedAzureOpenAIServerModel(
                model_id=config["model_id"],
                azure_endpoint=endpoint,
                api_key=new_token,
                api_version=config["api_version"],
                temperature=0.0,  # Set deterministic temperature
                top_p=1.0,  # Use full probability distribution
                frequency_penalty=0.0,  # No frequency penalty
                presence_penalty=0.0,  # No presence penalty
                seed=42,  # Fixed seed for reproducibility
                max_tokens=4000,  # Consistent response length
                custom_role_conversions={
                    "tool-call": "user",
                    "tool-response": "assistant",
                },
            )
            
            # Initialize token management in the new model
            self.model._credential = self._credential
            self.model._token_expiry = datetime.now() + timedelta(seconds=new_expiry - int(datetime.now().timestamp()))
            
            print(f"[OK] Model recreated with fresh token, expires at {self.model._token_expiry.strftime('%H:%M:%S')}")
            
            # Update the agent with the new model
            if hasattr(self, 'agent') and self.agent:
                self.agent.model = self.model
                print("[OK] Agent updated with new model")
                
        except Exception as e:
            print(f"[ERROR] Failed to recreate model with fresh token: {e}")
            raise
    
    def _create_agent(self):
        """Create the CodeAgent with tools"""
        try:
            # Initialize terminology tool with glossary folder
            glossary_folder = "glossary"  # Default glossary folder
            terminology_tool = TerminologyTool(glossary_folder)
            
            # Create the modern terminology review tool
            review_tool = ModernTerminologyReviewTool(terminology_tool, self.search_engine)
            
            # Create agent with tools and increased limits for complex processing
            self.agent = CodeAgent(
                tools=[review_tool],
                model=self.model,
                additional_authorized_imports=["json", "time", "datetime"],
                max_steps=75  # Further increased for problematic dictionary batches
            )
            
            print("[OK] Modern agent created with DDGS search engine")
            
        except Exception as e:
            print(f"[ERROR] Failed to create agent: {e}")
            raise
    
    def _create_standardized_prompt(self, base_prompt: str) -> str:
        """Create standardized prompt with consistent formatting"""
        return f"""
{base_prompt}

IMPORTANT INSTRUCTIONS:
- Use the terminology_review_tool for all validations
- Provide detailed analysis including web research and context
- Return results in JSON format with proper structure
- Be thorough but efficient in your analysis
- Use deterministic scoring based on evidence

Remember: You have access to modern multi-engine web search via DDGS library.
"""
    
    def run_review_task(self, prompt: str) -> str:
        """Run a review task with standardized prompt and error handling"""
        standardized_prompt = self._create_standardized_prompt(prompt)
        
        def execute_task():
            # Check if token refresh is needed
            if hasattr(self.model, '_refresh_token_if_needed') and self.model._refresh_token_if_needed():
                self.refresh_model_token()
            
            return self.agent.run(standardized_prompt)
        
        return run_with_retries(execute_task, max_retries=3)
    
    def validate_term_candidate(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                              industry_context: str = "General", save_to_file: str = "") -> str:
        """Validate a single terminology candidate"""
        
        prompt = f"""
Validate the terminology candidate: "{term}"

Parameters:
- Source Language: {src_lang}
- Target Language: {tgt_lang if tgt_lang else 'EN'}
- Industry Context: {industry_context}
- Save to file: {save_to_file if save_to_file else 'No'}

Use the terminology_review_tool with action 'validate_single' to perform comprehensive validation.
Include web research, Autodesk context analysis, and provide a detailed assessment.
"""
        
        return self.run_review_task(prompt)
    
    def validate_term_candidate_efficient(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                                        industry_context: str = "General", save_to_file: str = "", 
                                        original_texts: List[str] = None, translation_data: Dict[str, Any] = None) -> str:
        """Validate a single terminology candidate with efficiency optimizations for gap filling"""
        
        # Use efficiency mode for problematic terms with many original texts
        limit_efficiency = original_texts and len(original_texts) > 15
        if limit_efficiency:
            print("[FAST] Using efficiency mode for term with many original texts")
        
        # Prepare translation data parameter
        translation_data_param = ""
        if translation_data:
            try:
                translation_data_param = json.dumps(translation_data)
                print(f"🌍 Including translation insights in validation for '{term}'")
            except Exception as e:
                print(f"[WARNING] Warning: Could not serialize translation_data: {e}")
                translation_data_param = ""

        prompt = f"""
Validate the terminology candidate: "{term}"

Parameters:
- Source Language: {src_lang}
- Target Language: {tgt_lang if tgt_lang else 'EN'}
- Industry Context: {industry_context}
- Save to file: {save_to_file if save_to_file else 'No'}
- Efficiency Mode: {'Yes' if limit_efficiency else 'No'}
- Translation Data: {translation_data_param if translation_data_param else 'No translation data available'}

Use the terminology_review_tool with action 'validate_single' to perform comprehensive validation.
Include web research, Autodesk context analysis, translation insights analysis, and provide a detailed assessment.
The translation data provides multilingual validation insights that should enhance the overall scoring.
"""
        
        return self.run_review_task(prompt)
    
    def batch_validate_terms(self, terms: List[str], src_lang: str = "EN", tgt_lang: str = None, 
                           industry_context: str = "General", save_to_file: str = "", 
                           original_texts: List[List[str]] = None) -> str:
        """Validate multiple terminology candidates in batch"""
        
        # Prepare original_texts parameter
        original_texts_param = ""
        if original_texts:
            try:
                original_texts_param = json.dumps(original_texts)
            except Exception as e:
                print(f"[WARNING] Warning: Could not serialize original_texts: {e}")
                original_texts_param = ""
        
        prompt = f"""
Validate these terminology candidates in batch: {json.dumps(terms)}

Parameters:
- Source Language: {src_lang}
- Target Language: {tgt_lang if tgt_lang else 'EN'}
- Industry Context: {industry_context}
- Save to file: {save_to_file if save_to_file else 'No'}
- Original texts: {original_texts_param if original_texts_param else 'None provided'}

Use the terminology_review_tool with action 'validate_batch' to validate all terms.
Provide comprehensive analysis for each term including web research and scoring.
"""
        
        return self.run_review_task(prompt)
    
    def generate_comprehensive_report(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                                    industry_context: str = "General", save_to_file: str = "") -> str:
        """Generate a comprehensive validation report for a term"""
        
        prompt = f"""
Generate a comprehensive validation report for: "{term}"

Parameters:
- Source Language: {src_lang}
- Target Language: {tgt_lang if tgt_lang else 'EN'}
- Industry Context: {industry_context}

Perform detailed analysis including:
1. Autodesk terminology database context
2. Multi-engine web research with DDGS
3. Industry relevance assessment
4. Technical validity evaluation
5. Usage examples and patterns
6. Final recommendation with scoring

Save results to: {save_to_file if save_to_file else 'No file specified'}
"""
        
        return self.run_review_task(prompt)


def main():
    """Main function for testing the modern agent"""
    print("[LAUNCH] Modern Terminology Review Agent with DDGS")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = ModernTerminologyReviewAgent("gpt-4.1")
        
        # Test single term validation
        print("\n[LOG] Testing single term validation...")
        result = agent.validate_term_candidate(
            term="extrude", 
            industry_context="CAD",
            save_to_file="modern_test_result.json"
        )
        print("[OK] Single term validation completed")
        
        # Test batch validation
        print("\n[PACKAGE] Testing batch validation...")
        batch_result = agent.batch_validate_terms(
            terms=["chamfer", "fillet", "revolve"],
            industry_context="CAD",
            save_to_file="modern_batch_test.json"
        )
        print("[OK] Batch validation completed")
        
        print("\n[SUCCESS] All tests completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
