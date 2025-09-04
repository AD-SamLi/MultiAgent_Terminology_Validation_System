#!/usr/bin/env python3
"""
Terminology Review Agent with Web Search Capabilities
Uses smolagents with Azure OpenAI GPT-5 and DuckDuckGo search to validate term candidates
for Autodesk by researching industry usage and context.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from azure.identity import EnvironmentCredential
from dotenv import load_dotenv

# Import smolagents components
from smolagents import CodeAgent, Tool, AzureOpenAIServerModel, DuckDuckGoSearchTool

# Import the terminology tool
from terminology_tool import TerminologyTool


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
            completion_kwargs.setdefault("max_completion_tokens", 256)
        else:
            # For GPT-4.1 and other models, ensure temperature is set for deterministic results
            completion_kwargs.setdefault("temperature", 0.0)
            # Add other parameters for consistent output
            completion_kwargs.setdefault("top_p", 1.0)  # Use full probability distribution
            completion_kwargs.setdefault("frequency_penalty", 0.0)  # No frequency penalty
            completion_kwargs.setdefault("presence_penalty", 0.0)  # No presence penalty
            completion_kwargs.setdefault("seed", 42)  # Fixed seed for reproducibility
            # Set max tokens for consistent response length
            completion_kwargs.setdefault("max_tokens", 4000)
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


class TerminologyReviewTool(Tool):
    """
    Advanced terminology review tool that combines glossary analysis with web search
    to validate term candidates for Autodesk industry context.
    """
    
    name = "terminology_reviewer"
    description = """
    A comprehensive terminology review and validation tool for Autodesk industry context.
    
    This tool can:
    - Analyze term candidates against existing Autodesk glossaries
    - Research industry usage and context via web search
    - Validate terms for CAD/AEC/Manufacturing industries
    - Generate detailed reports with web research findings
    - Create JSON output files with validation data
    - Identify industry-specific terminology patterns
    - Compare against approved Autodesk terminology
    
    The tool leverages both internal glossary knowledge and external web research
    to provide comprehensive term validation for technical translation projects.
    """
    
    inputs = {
        "action": {
            "type": "string",
            "description": "Action to perform: 'validate_term', 'research_industry_usage', 'analyze_glossary_patterns', 'generate_term_report', 'batch_validate_terms'"
        },
        "term": {
            "type": "string",
            "description": "Term or phrase to analyze/validate (required for single term actions)",
            "nullable": True
        },
        "terms": {
            "type": "string", 
            "description": "JSON array of terms for batch processing (required for batch_validate_terms)",
            "nullable": True
        },
        "src_lang": {
            "type": "string",
            "description": "Source language code (e.g., 'EN', 'en') - defaults to 'EN'",
            "nullable": True
        },
        "tgt_lang": {
            "type": "string",
            "description": "Target language code (e.g., 'CS', 'DE') - defaults to 'CS'",
            "nullable": True
        },
        "industry_context": {
            "type": "string",
            "description": "Industry context: 'CAD', 'AEC', 'Manufacturing', 'General' - defaults to 'General'",
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
        }
    }
    
    output_type = "string"
    
    def __init__(self, glossary_folder: str, search_tool: DuckDuckGoSearchTool):
        """Initialize the terminology review tool"""
        super().__init__()
        self.glossary_folder = glossary_folder
        self.search_tool = search_tool
        self.terminology_tool = None
        self.results_cache = {}
        self._initialize_tool()
    
    def _initialize_tool(self):
        """Initialize the underlying TerminologyTool"""
        try:
            self.terminology_tool = TerminologyTool(self.glossary_folder)
            print(f"✅ Terminology review tool initialized with glossary folder: {self.glossary_folder}")
        except Exception as e:
            print(f"❌ Failed to initialize terminology tool: {e}")
            self.terminology_tool = None
    
    def _search_web(self, query: str) -> str:
        """Perform web search using the DuckDuckGo search tool"""
        try:
            results = self.search_tool(query)
            return results
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    def _analyze_autodesk_context(self, term: str) -> Dict[str, Any]:
        """Analyze how a term fits within Autodesk's existing terminology domain"""
        context_analysis = {
            "existing_translations": {},
            "related_terms": [],
            "industry_categories": [],
            "domain_relevance_score": 0.0,
            "already_exists": False,
            "duplicate_count": 0
        }
        
        if not self.terminology_tool:
            return context_analysis
        
        # Check all available language pairs for this term
        language_pairs = self.terminology_tool.get_available_language_pairs()
        
        for lang_pair in language_pairs:
            # Handle both tuple and string formats
            if isinstance(lang_pair, tuple) and len(lang_pair) == 2:
                src_lang, tgt_lang = lang_pair
            elif isinstance(lang_pair, str) and '-' in lang_pair:
                src_lang, tgt_lang = lang_pair.split('-', 1)
            else:
                continue  # Skip invalid pairs
                
            glossary = self.terminology_tool.get_relevant_terms(src_lang, tgt_lang)
            
            # Check for exact match (shouldn't happen with pre-filtered input)
            if term in glossary:
                context_analysis["existing_translations"][f"{src_lang}-{tgt_lang}"] = glossary[term]
                context_analysis["already_exists"] = True
                context_analysis["duplicate_count"] += 1
            
            # Find related terms for domain context analysis
            for glossary_term in glossary:
                # More sophisticated similarity matching
                term_lower = term.lower()
                glossary_lower = glossary_term.lower()
                
                # Check for various types of relationships
                is_related = False
                if term_lower in glossary_lower or glossary_lower in term_lower:
                    is_related = True
                elif any(word in glossary_lower for word in term_lower.split() if len(word) > 3):
                    is_related = True
                elif any(word in term_lower for word in glossary_lower.split() if len(word) > 3):
                    is_related = True
                
                if is_related and not any(rt["term"] == glossary_term for rt in context_analysis["related_terms"]):
                    context_analysis["related_terms"].append({
                        "term": glossary_term,
                        "translation": glossary[glossary_term],
                        "language_pair": f"{src_lang}-{tgt_lang}"
                    })
                    # Bonus for domain relevance
                    context_analysis["domain_relevance_score"] += 0.1
        
        # Determine industry categories based on language pair patterns
        for related_term in context_analysis["related_terms"]:
            pair_str = related_term["language_pair"]
            if "ACAD" in pair_str and "CAD" not in context_analysis["industry_categories"]:
                context_analysis["industry_categories"].append("CAD")
                context_analysis["domain_relevance_score"] += 0.2
            if "AEC" in pair_str and "AEC" not in context_analysis["industry_categories"]:
                context_analysis["industry_categories"].append("AEC")
                context_analysis["domain_relevance_score"] += 0.2
            if "MNE" in pair_str and "Manufacturing" not in context_analysis["industry_categories"]:
                context_analysis["industry_categories"].append("Manufacturing")
                context_analysis["domain_relevance_score"] += 0.2
        
        # Additional scoring based on number of related terms
        related_count = len(context_analysis["related_terms"])
        if related_count >= 10:
            context_analysis["domain_relevance_score"] += 0.3
        elif related_count >= 5:
            context_analysis["domain_relevance_score"] += 0.2
        elif related_count >= 2:
            context_analysis["domain_relevance_score"] += 0.1
        
        return context_analysis
    
    def _analyze_original_texts(self, term: str, original_texts: List[str]) -> Dict[str, Any]:
        """Analyze original texts where the term appears for context validation"""
        if not original_texts:
            return {"context_score": 0.0, "usage_patterns": [], "product_mentions": [], "technical_indicators": []}
        
        context_analysis = {
            "context_score": 0.0,
            "usage_patterns": [],
            "product_mentions": [],
            "technical_indicators": [],
            "text_count": len(original_texts)
        }
        
        # Autodesk product names to look for
        autodesk_products = [
            "AutoCAD", "Revit", "Inventor", "3ds Max", "Maya", "Fusion 360",
            "Civil 3D", "Navisworks", "Vault", "Mudbox", "MotionBuilder",
            "Alias", "VRED", "Flame", "Smoke", "Arnold", "Shotgun",
            "Building Design Suite", "Infrastructure Design Suite",
            "Product Design Suite", "Entertainment Creation Suite"
        ]
        
        # Technical indicators for CAD/AEC/Manufacturing
        technical_keywords = [
            "mesh", "model", "design", "rendering", "simulation", "analysis",
            "geometry", "surface", "solid", "parametric", "BIM", "CAD",
            "drawing", "sketch", "extrude", "revolve", "sweep", "loft",
            "assembly", "component", "feature", "constraint", "dimension",
            "material", "texture", "lighting", "animation", "workflow",
            "documentation", "visualization", "fabrication", "manufacturing"
        ]
        
        technical_count = 0
        product_count = 0
        
        for text in original_texts:
            text_lower = text.lower()
            
            # Check for Autodesk product mentions
            for product in autodesk_products:
                if product.lower() in text_lower:
                    context_analysis["product_mentions"].append({
                        "product": product,
                        "text": text[:100] + "..." if len(text) > 100 else text
                    })
                    product_count += 1
                    break
            
            # Check for technical keywords
            tech_words_found = []
            for keyword in technical_keywords:
                if keyword in text_lower:
                    tech_words_found.append(keyword)
                    technical_count += 1
            
            if tech_words_found:
                context_analysis["technical_indicators"].append({
                    "keywords": tech_words_found[:5],  # Limit to first 5
                    "text": text[:100] + "..." if len(text) > 100 else text
                })
            
            # Analyze usage patterns
            term_lower = term.lower()
            if term_lower in text_lower:
                # Check context around the term
                words = text_lower.split()
                term_indices = [i for i, word in enumerate(words) if term_lower in word]
                
                for idx in term_indices:
                    context_words = []
                    # Get 2 words before and after
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    context_words = words[start:end]
                    
                    context_analysis["usage_patterns"].append({
                        "context": " ".join(context_words),
                        "full_text": text[:150] + "..." if len(text) > 150 else text
                    })
        
        # Calculate context score
        base_score = 0.0
        
        # Score for product mentions (up to 0.3)
        if product_count > 0:
            base_score += min(0.3, product_count * 0.05)
        
        # Score for technical context (up to 0.4)
        if technical_count > 0:
            base_score += min(0.4, technical_count * 0.02)
        
        # Score for usage diversity (up to 0.2)
        unique_patterns = len(set(p["context"] for p in context_analysis["usage_patterns"]))
        if unique_patterns > 0:
            base_score += min(0.2, unique_patterns * 0.02)
        
        # Bonus for high text count (up to 0.1)
        if len(original_texts) >= 20:
            base_score += 0.1
        elif len(original_texts) >= 10:
            base_score += 0.05
        
        context_analysis["context_score"] = min(base_score, 1.0)
        
        return context_analysis
    
    def _validate_single_term(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                            industry_context: str = "General", original_texts: List[str] = None) -> Dict[str, Any]:
        """Validate a single term using both glossary analysis and web search"""
        
        validation_result = {
            "term": term,
            "timestamp": datetime.now().isoformat(),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "industry_context": industry_context,
            "autodesk_analysis": {},
            "context_analysis": {},
            "web_research": {},
            "validation_score": 0.0,
            "recommendations": [],
            "status": "unknown"
        }
        
        try:
            # 1. Analyze against Autodesk glossaries
            print(f"🔍 Analyzing '{term}' against Autodesk glossaries...")
            validation_result["autodesk_analysis"] = self._analyze_autodesk_context(term)
            
            # 2. Analyze original texts for context (if available)
            if original_texts:
                print(f"📝 Analyzing original usage contexts for '{term}'...")
                validation_result["context_analysis"] = self._analyze_original_texts(term, original_texts)
            else:
                validation_result["context_analysis"] = {"context_score": 0.0, "note": "No original texts provided"}
            
            # 3. Web search for industry usage
            print(f"🌐 Researching '{term}' industry usage...")
            
            # Search for general industry usage
            general_query = f'"{term}" CAD software engineering technical terminology'
            general_results = self._search_web(general_query)
            
            # Search for Autodesk-specific usage
            autodesk_query = f'"{term}" Autodesk AutoCAD Revit Inventor Maya 3ds Max'
            autodesk_results = self._search_web(autodesk_query)
            
            # Search for industry-specific context
            if industry_context != "General":
                industry_query = f'"{term}" {industry_context} industry terminology definition'
                industry_results = self._search_web(industry_query)
            else:
                industry_results = "No industry-specific search performed"
            
            validation_result["web_research"] = {
                "general_industry": general_results,
                "autodesk_specific": autodesk_results,
                "industry_context": industry_results,
                "search_queries": [general_query, autodesk_query]
            }
            
            # 4. Calculate validation score
            score = 0.0
            autodesk_analysis = validation_result["autodesk_analysis"]
            context_analysis = validation_result["context_analysis"]
            
            # NOTE: Input terms are pre-filtered to exclude existing glossary terms
            # Focus on domain relevance, context usage, and technical validity
            
            # Check if term unexpectedly exists (shouldn't happen with clean input)
            if autodesk_analysis.get("already_exists", False):
                duplicate_count = autodesk_analysis.get("duplicate_count", 0)
                validation_result["recommendations"].append(f"⚠️ WARNING: Term found in {duplicate_count} glossaries despite pre-filtering")
                score -= 0.2  # Minor penalty - shouldn't happen with clean input
            
            # MAJOR BONUS for original text context analysis (up to 0.5 points)
            context_score = context_analysis.get("context_score", 0.0)
            if context_score > 0:
                score += min(context_score, 0.5)
                
                # Detailed context recommendations
                product_mentions = len(context_analysis.get("product_mentions", []))
                technical_indicators = len(context_analysis.get("technical_indicators", []))
                text_count = context_analysis.get("text_count", 0)
                
                if product_mentions > 0:
                    validation_result["recommendations"].append(f"🎯 STRONG: Found in {product_mentions} Autodesk product contexts")
                
                if technical_indicators > 0:
                    validation_result["recommendations"].append(f"⚙️ TECHNICAL: Used in {technical_indicators} technical contexts")
                
                if text_count >= 20:
                    validation_result["recommendations"].append(f"📊 HIGH USAGE: Appears in {text_count} original texts")
                elif text_count >= 10:
                    validation_result["recommendations"].append(f"📊 GOOD USAGE: Appears in {text_count} original texts")
            
            # BONUS for related terms (shows term is in relevant domain) (up to 0.3)
            if autodesk_analysis.get("related_terms"):
                score += 0.3
                related_count = len(autodesk_analysis["related_terms"])
                validation_result["recommendations"].append(f"✅ Domain relevance: {related_count} related terms found in glossaries")
            
            # BONUS for industry categorization (up to 0.2)
            if autodesk_analysis.get("industry_categories"):
                score += 0.2
                categories = ", ".join(autodesk_analysis["industry_categories"])
                validation_result["recommendations"].append(f"✅ Industry context: {categories}")
            
            # Score from web presence (technical validity) (up to 0.2)
            if "definition" in general_results.lower() or "technical" in general_results.lower():
                score += 0.1
                validation_result["recommendations"].append("✅ Technical definition found online")
            
            if "autodesk" in autodesk_results.lower() or "autocad" in autodesk_results.lower():
                score += 0.1
                validation_result["recommendations"].append("✅ Term used in Autodesk product contexts online")
            
            # Include domain relevance score from glossary analysis (up to 0.2)
            domain_score = autodesk_analysis.get("domain_relevance_score", 0.0)
            score += min(domain_score * 0.02, 0.2)  # Scale down glossary contribution
            
            if domain_score > 0.3:
                validation_result["recommendations"].append("✅ Excellent domain fit with existing terminology")
            elif domain_score > 0.1:
                validation_result["recommendations"].append("✅ Good contextual fit with existing terminology")
            elif domain_score > 0.0:
                validation_result["recommendations"].append("✅ Some domain context found")
            
            validation_result["validation_score"] = max(0.0, min(score, 1.0))  # Clamp between 0 and 1
            
            # 4. Determine status based on validation score
            # Input terms are pre-filtered, so focus on domain fit and technical validity
            if validation_result["validation_score"] >= 0.8:
                validation_result["status"] = "recommended"
                validation_result["recommendations"].append("🎯 RECOMMENDED: Strong domain relevance and technical validity")
            elif validation_result["validation_score"] >= 0.5:
                validation_result["status"] = "needs_review"
                validation_result["recommendations"].append("⚠️ NEEDS REVIEW: Good potential, requires human evaluation")
            elif validation_result["validation_score"] >= 0.3:
                validation_result["status"] = "low_priority"
                validation_result["recommendations"].append("📋 LOW PRIORITY: Limited domain context, consider for future")
            else:
                validation_result["status"] = "not_recommended"
                validation_result["recommendations"].append("❌ NOT RECOMMENDED: Insufficient evidence of technical relevance")
            
            # Special note for unexpected duplicates
            if autodesk_analysis.get("already_exists", False):
                validation_result["status"] = "unexpected_duplicate"
                validation_result["recommendations"].append("🔍 INVESTIGATE: Found in glossary despite pre-filtering")
            
            print(f"✅ Validation complete for '{term}' - Score: {validation_result['validation_score']:.2f}")
            
        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["status"] = "error"
            print(f"❌ Error validating '{term}': {e}")
        
        return validation_result
    
    def forward(self, action: str, term: str = "", terms: str = "", src_lang: str = "EN", 
                tgt_lang: str = None, industry_context: str = "General", output_file: str = "",
                original_texts: str = "") -> str:
        """Execute the requested terminology review operation"""
        
        try:
            if action == "validate_term":
                if not term:
                    return json.dumps({"error": "Term parameter is required for validate_term action"})
                
                result = self._validate_single_term(term, src_lang, tgt_lang, industry_context)
                
                # Save to file if specified
                if output_file:
                    self._save_results_to_file([result], output_file)
                
                return json.dumps(result, indent=2)
            
            elif action == "batch_validate_terms":
                if not terms:
                    return json.dumps({"error": "Terms parameter (JSON array) is required for batch_validate_terms action"})
                
                try:
                    term_list = json.loads(terms)
                    if not isinstance(term_list, list):
                        return json.dumps({"error": "Terms parameter must be a JSON array"})
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid JSON format for terms parameter"})
                
                # Parse original_texts if provided
                original_texts_list = []
                if original_texts:
                    try:
                        original_texts_list = json.loads(original_texts)
                        if not isinstance(original_texts_list, list):
                            original_texts_list = []
                    except json.JSONDecodeError:
                        original_texts_list = []
                
                results = []
                for i, single_term in enumerate(term_list):
                    print(f"📊 Processing term {i+1}/{len(term_list)}: {single_term}")
                    
                    # Get original texts for this term if available
                    term_original_texts = []
                    if i < len(original_texts_list) and isinstance(original_texts_list[i], list):
                        term_original_texts = original_texts_list[i]
                    
                    result = self._validate_single_term(single_term, src_lang, tgt_lang, industry_context, term_original_texts)
                    results.append(result)
                    # Small delay to be respectful to web search API
                    time.sleep(1)
                
                batch_result = {
                    "action": "batch_validate_terms",
                    "total_terms": len(term_list),
                    "processed": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                }
                
                # Save to file if specified
                if output_file:
                    # Save the individual results, not the wrapper
                    self._save_results_to_file(results, output_file)
                
                return json.dumps(batch_result, indent=2)
            
            elif action == "research_industry_usage":
                if not term:
                    return json.dumps({"error": "Term parameter is required for research_industry_usage action"})
                
                # Focused web research on industry usage
                research_queries = [
                    f'"{term}" technical definition engineering',
                    f'"{term}" CAD software terminology',
                    f'"{term}" {industry_context} industry standard',
                    f'"{term}" Autodesk documentation manual'
                ]
                
                research_results = {}
                for i, query in enumerate(research_queries):
                    print(f"🔎 Research query {i+1}/{len(research_queries)}: {query}")
                    research_results[f"query_{i+1}"] = {
                        "query": query,
                        "results": self._search_web(query)
                    }
                    time.sleep(0.5)  # Rate limiting
                
                result = {
                    "action": "research_industry_usage",
                    "term": term,
                    "industry_context": industry_context,
                    "timestamp": datetime.now().isoformat(),
                    "research_data": research_results
                }
                
                if output_file:
                    self._save_results_to_file([result], output_file)
                
                return json.dumps(result, indent=2)
            
            elif action == "analyze_glossary_patterns":
                if not self.terminology_tool:
                    return json.dumps({"error": "Terminology tool not initialized"})
                
                # Analyze patterns in existing glossaries
                patterns = {
                    "language_coverage": {},
                    "industry_distribution": {},
                    "term_length_analysis": {},
                    "common_prefixes": {},
                    "common_suffixes": {}
                }
                
                language_pairs = self.terminology_tool.get_available_language_pairs()
                
                for src_lang, tgt_lang in language_pairs:
                    glossary = self.terminology_tool.get_relevant_terms(src_lang, tgt_lang)
                    pair_key = f"{src_lang}-{tgt_lang}"
                    patterns["language_coverage"][pair_key] = len(glossary)
                    
                    # Analyze term lengths
                    if glossary:
                        term_lengths = [len(term.split()) for term in glossary.keys()]
                        patterns["term_length_analysis"][pair_key] = {
                            "avg_length": sum(term_lengths) / len(term_lengths),
                            "min_length": min(term_lengths),
                            "max_length": max(term_lengths)
                        }
                
                result = {
                    "action": "analyze_glossary_patterns",
                    "timestamp": datetime.now().isoformat(),
                    "analysis": patterns,
                    "summary": {
                        "total_language_pairs": len(language_pairs),
                        "total_unique_languages": len(set([lang for pair in language_pairs for lang in pair]))
                    }
                }
                
                if output_file:
                    self._save_results_to_file([result], output_file)
                
                return json.dumps(result, indent=2)
            
            elif action == "generate_term_report":
                if not term:
                    return json.dumps({"error": "Term parameter is required for generate_term_report action"})
                
                # Comprehensive report combining all analysis types
                validation = self._validate_single_term(term, src_lang, tgt_lang, industry_context)
                
                # Additional detailed research
                extended_research = {}
                research_queries = [
                    f'"{term}" definition technical dictionary',
                    f'"{term}" usage examples engineering',
                    f'"{term}" Autodesk software documentation'
                ]
                
                # Add translation query only if target language is specified
                if tgt_lang and tgt_lang != "EN":
                    research_queries.insert(2, f'"{term}" translation {tgt_lang} language')
                
                for i, query in enumerate(research_queries):
                    extended_research[f"extended_query_{i+1}"] = {
                        "query": query,
                        "results": self._search_web(query)
                    }
                    time.sleep(0.5)
                
                comprehensive_report = {
                    "action": "generate_term_report",
                    "term": term,
                    "comprehensive_analysis": validation,
                    "extended_research": extended_research,
                    "timestamp": datetime.now().isoformat(),
                    "report_summary": {
                        "validation_score": validation.get("validation_score", 0),
                        "status": validation.get("status", "unknown"),
                        "key_findings": validation.get("recommendations", [])
                    }
                }
                
                if output_file:
                    self._save_results_to_file([comprehensive_report], output_file)
                
                return json.dumps(comprehensive_report, indent=2)
            
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["validate_term", "research_industry_usage", "analyze_glossary_patterns", 
                                    "generate_term_report", "batch_validate_terms"]
                })
                
        except Exception as e:
            return json.dumps({
                "error": f"Error executing {action}: {str(e)}",
                "action": action
            })
    
    def _standardize_result_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize the format of a single validation result"""
        standardized = {
            "term": result.get("term", "unknown"),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "src_lang": result.get("src_lang", "EN"),
            "tgt_lang": result.get("tgt_lang", "CS"),
            "industry_context": result.get("industry_context", "General"),
            "validation_score": round(float(result.get("validation_score", 0.0)), 3),  # Standardize to 3 decimals
            "status": result.get("status", "unknown"),
            "autodesk_analysis": result.get("autodesk_analysis", {}),
            "web_research": result.get("web_research", {}),
            "recommendations": result.get("recommendations", []),
        }
        
        # Add error field if present
        if "error" in result:
            standardized["error"] = result["error"]
            
        return standardized

    def _save_results_to_file(self, results: List[Dict], output_file: str):
        """Save results to JSON file with standardized formatting"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Standardize all results
            standardized_results = [self._standardize_result_format(result) for result in results]
            
            # Create standardized output structure
            output_data = {
                "metadata": {
                    "generated_by": "TerminologyReviewAgent",
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "total_results": len(standardized_results),
                    "model_used": getattr(self, 'model_name', 'unknown'),
                    "parameters": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 42,
                        "max_search_results": 50
                    }
                },
                "results": standardized_results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, sort_keys=True)
            
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results to {output_file}: {e}")


class TerminologyReviewAgent:
    """
    Advanced Terminology Review Agent with Web Search Capabilities
    Powered by smolagents, Azure OpenAI GPT-5, and DuckDuckGo search
    """
    
    def __init__(self, glossary_folder: str, model_name: str = "gpt-5"):
        """Initialize the terminology review agent"""
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
            
            # Create patched model for GPT-5 compatibility with deterministic parameters
            self.model = PatchedAzureOpenAIServerModel(
                model_id=config["model_id"],
                azure_endpoint=endpoint,
                api_key=token_result.token,
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
            
            print(f"✅ Model {self.model_name} initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup model: {e}")
            raise
    
    def _create_agent(self):
        """Create the smolagents CodeAgent with terminology review and web search tools"""
        print("🤖 Creating terminology review agent...")
        
        try:
            # Create web search tool with maximum results
            search_tool = DuckDuckGoSearchTool(max_results=50)
            
            # Create the terminology review tool
            review_tool = TerminologyReviewTool(self.glossary_folder, search_tool)
            
            # Create the agent with both tools and additional imports
            self.agent = CodeAgent(
                model=self.model,
                tools=[review_tool, search_tool],  # Include both review tool and direct search access
                add_base_tools=True,  # Include other base tools
                max_steps=20,  # Allow more steps for complex research operations
                additional_authorized_imports=["json"]  # Allow JSON import for parsing results
            )
            
            print("✅ Terminology review agent created successfully")
            print(f"📁 Using glossary folder: {self.glossary_folder}")
            print("🌐 Web search capabilities enabled via DuckDuckGo")
            
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            raise
    
    def validate_term_candidate(self, term: str, src_lang: str = "EN", tgt_lang: str = None, 
                              industry_context: str = "General", save_to_file: str = "") -> str:
        """Validate a single term candidate with comprehensive research"""
        
        output_file_param = f', output_file="{save_to_file}"' if save_to_file else ""
        
        prompt = f"""
        Please validate the term candidate "{term}" for Autodesk terminology using comprehensive research.
        
        Use the terminology_reviewer tool with the following parameters:
        - action: "validate_term"
        - term: "{term}"
        - src_lang: "{src_lang}"
        - tgt_lang: "{tgt_lang if tgt_lang else 'EN'}"  # Terminology validation only
        - industry_context: "{industry_context}"{output_file_param}
        
        The validation should:
        1. Check against existing Autodesk glossaries for similar or related terms
        2. Research the term's usage in CAD/AEC/Manufacturing industries via web search
        3. Evaluate the term's technical validity and industry acceptance
        4. Provide recommendations on whether to include this term in Autodesk terminology
        5. Generate a comprehensive validation report
        
        Present the results in a clear, structured format highlighting:
        - Validation score and status
        - Key findings from glossary analysis
        - Industry usage research results
        - Final recommendations
        """
        
        return self.run_review_task(prompt)
    
    def batch_validate_terms(self, terms: List[str], src_lang: str = "EN", tgt_lang: str = None,
                           industry_context: str = "General", save_to_file: str = "",
                           original_texts: List[List[str]] = None) -> str:
        """Validate multiple term candidates in batch"""
        
        terms_json = json.dumps(terms)
        output_file_param = f', output_file="{save_to_file}"' if save_to_file else ""
        
        tgt_lang_param = f'- tgt_lang: "{tgt_lang}"' if tgt_lang else '- tgt_lang: "EN"  # Terminology validation only'
        
        # Prepare original_texts parameter
        original_texts_param = ""
        if original_texts:
            original_texts_json = json.dumps(original_texts)
            original_texts_param = f', original_texts: \'{original_texts_json}\''
        
        prompt = f"""
        Please validate the following term candidates for Autodesk terminology using batch processing.
        
        Terms to validate: {terms_json}
        
        Use the terminology_reviewer tool with:
        - action: "batch_validate_terms"
        - terms: '{terms_json}'
        - src_lang: "{src_lang}"
        {tgt_lang_param}
        - industry_context: "{industry_context}"{output_file_param}{original_texts_param}
        
        For each term, the system should:
        1. Analyze against Autodesk glossaries
        2. Research industry usage patterns
        3. Calculate validation scores
        4. Provide specific recommendations
        
        Present a summary showing:
        - Overall batch statistics
        - Terms recommended for inclusion
        - Terms needing further review
        - Terms not recommended
        - Key patterns discovered
        """
        
        return self.run_review_task(prompt)
    
    def research_industry_usage(self, term: str, industry_context: str = "General", 
                              save_to_file: str = "") -> str:
        """Deep research on industry usage patterns for a term"""
        
        output_file_param = f', output_file="{save_to_file}"' if save_to_file else ""
        
        prompt = f"""
        Please conduct deep research on the industry usage of the term "{term}".
        
        Use the terminology_reviewer tool with:
        - action: "research_industry_usage"
        - term: "{term}"
        - industry_context: "{industry_context}"{output_file_param}
        
        The research should explore:
        1. Technical definitions and usage in engineering contexts
        2. Prevalence in CAD software terminology
        3. Industry-specific applications and meanings
        4. Usage in Autodesk product documentation
        5. Comparison with related technical terms
        
        Analyze the research findings and provide insights on:
        - Term's legitimacy in technical contexts
        - Consistency of usage across sources
        - Relevance to Autodesk's product ecosystem
        - Potential translation challenges
        """
        
        return self.run_review_task(prompt)
    
    def analyze_glossary_patterns(self, save_to_file: str = "") -> str:
        """Analyze patterns in existing Autodesk glossaries"""
        
        output_file_param = f', output_file="{save_to_file}"' if save_to_file else ""
        
        prompt = f"""
        Please analyze the patterns and characteristics of existing Autodesk terminology glossaries.
        
        Use the terminology_reviewer tool with:
        - action: "analyze_glossary_patterns"{output_file_param}
        
        The analysis should examine:
        1. Language coverage and distribution
        2. Industry-specific terminology patterns
        3. Term complexity and structure patterns
        4. Common terminology themes
        5. Gaps or opportunities for expansion
        
        Provide insights that could help guide:
        - New term candidate evaluation criteria
        - Industry focus areas for terminology development
        - Quality standards for term inclusion
        - Translation consistency patterns
        """
        
        return self.run_review_task(prompt)
    
    def generate_comprehensive_report(self, term: str, src_lang: str = "EN", tgt_lang: str = None,
                                   industry_context: str = "General", save_to_file: str = "") -> str:
        """Generate a comprehensive validation report for a term"""
        
        output_file_param = f', output_file="{save_to_file}"' if save_to_file else ""
        
        prompt = f"""
        Please generate a comprehensive validation report for the term "{term}".
        
        Use the terminology_reviewer tool with:
        - action: "generate_term_report"
        - term: "{term}"
        - src_lang: "{src_lang}"
        - tgt_lang: "{tgt_lang if tgt_lang else 'EN'}"  # Terminology validation only
        - industry_context: "{industry_context}"{output_file_param}
        
        The comprehensive report should include:
        1. Executive summary with validation decision
        2. Detailed glossary analysis results
        3. Extensive web research findings
        4. Industry context evaluation
        5. Translation considerations
        6. Implementation recommendations
        7. Risk assessment for term adoption
        
        Structure the report for stakeholder review and decision-making.
        """
        
        return self.run_review_task(prompt)
    
    def _create_standardized_prompt(self, base_prompt: str) -> str:
        """Create a standardized prompt with consistent formatting and instructions"""
        standardized_prompt = f"""
{base_prompt}

IMPORTANT STANDARDIZATION INSTRUCTIONS:
- Use consistent JSON formatting with proper indentation
- Always include validation scores as decimal numbers (e.g., 0.75, not 75%)
- Use standard status values: "recommended", "needs_review", "not_recommended", "error"
- Include timestamps in ISO format: YYYY-MM-DDTHH:MM:SS
- Provide recommendations as clear, actionable bullet points
- Use consistent terminology: "glossary analysis", "web research", "validation score"
- Ensure all numeric scores are between 0.0 and 1.0
- Format language pairs consistently as "SRC-TGT" (e.g., "EN-CS")
"""
        return standardized_prompt

    def run_review_task(self, prompt: str) -> str:
        """Run a terminology review task using the agent with standardized prompts"""
        print(f"🚀 Running terminology review task...")
        print(f"⏱️  Started at {time.strftime('%H:%M:%S')}")
        
        if self.model_name == "gpt-5":
            print("ℹ️  Using GPT-5 - reasoning may take 30-60+ seconds...")
        
        # Apply standardization to the prompt
        standardized_prompt = self._create_standardized_prompt(prompt)
        
        start_time = time.time()
        
        try:
            # Run the agent with retry logic
            response = run_with_retries(self.agent, standardized_prompt, max_retries=3)
            
            duration = time.time() - start_time
            print(f"⏱️  Completed in {duration:.1f} seconds")
            print(f"✅ Review task completed successfully")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Review task failed after {duration:.1f} seconds: {e}")
            raise


def demo_terminology_review_agent():
    """Demonstrate the terminology review agent capabilities"""
    
    print("🌟 TERMINOLOGY REVIEW AGENT DEMO")
    print("=" * 60)
    
    # Initialize the agent
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"📁 Using glossary folder: {glossary_folder}")
    
    try:
        # Create agent with GPT-5 (change to "gpt-4.1" for faster testing)
        agent = TerminologyReviewAgent(glossary_folder, model_name="gpt-5")
        
        print("\n1️⃣ VALIDATING SINGLE TERM CANDIDATE")
        print("-" * 50)
        validation = agent.validate_term_candidate(
            term="parametric modeling",
            industry_context="CAD",
            save_to_file="term_validation_parametric_modeling.json"
        )
        print(validation)
        
        print("\n2️⃣ RESEARCHING INDUSTRY USAGE")
        print("-" * 50)
        research = agent.research_industry_usage(
            term="mesh topology",
            industry_context="CAD",
            save_to_file="industry_research_mesh_topology.json"
        )
        print(research)
        
        print("\n3️⃣ BATCH VALIDATING TERM CANDIDATES")
        print("-" * 50)
        candidate_terms = [
            "spline interpolation",
            "boolean operations", 
            "surface continuity",
            "parametric constraints"
        ]
        batch_validation = agent.batch_validate_terms(
            terms=candidate_terms,
            industry_context="CAD",
            save_to_file="batch_validation_cad_terms.json"
        )
        print(batch_validation)
        
        print("\n4️⃣ ANALYZING GLOSSARY PATTERNS")
        print("-" * 50)
        patterns = agent.analyze_glossary_patterns(
            save_to_file="glossary_pattern_analysis.json"
        )
        print(patterns)
        
        print("\n5️⃣ COMPREHENSIVE TERM REPORT")
        print("-" * 50)
        report = agent.generate_comprehensive_report(
            term="NURBS surface",
            industry_context="CAD",
            save_to_file="comprehensive_report_nurbs.json"
        )
        print(report)
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\n📄 Generated JSON files:")
        print("  • term_validation_parametric_modeling.json")
        print("  • industry_research_mesh_topology.json")
        print("  • batch_validation_cad_terms.json")
        print("  • glossary_pattern_analysis.json")
        print("  • comprehensive_report_nurbs.json")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_terminology_review_agent()
