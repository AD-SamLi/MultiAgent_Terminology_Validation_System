#!/usr/bin/env python3
"""
Generic Term Detection System - NO TRAINING REQUIRED
Implements Methods 1, 2, and 5 using heuristics and pattern matching
With graduated penalties and technical term protections
"""

import re
import logging
from typing import Dict, List, Tuple
from collections import Counter
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GenericTermDetectorEnsemble:
    """
    Generic Term Detector combining multiple detection methods:
    - Method 1: Statistical Discrimination Score (analyzes context distribution)
    - Method 2: Comparative Frequency Analysis (heuristic-based)
    - Method 5: Semantic Similarity (pattern matching with known generics)
    
    NO TRAINING DATA REQUIRED - all methods work out of the box!
    """
    
    def __init__(self):
        """Initialize detector with technical term patterns"""
        # Technical term protection patterns
        self.technical_suffixes = {
            'tion', 'sion', 'ization', 'isation', 'ment', 'ness', 'ity', 'ence', 'ance',
            'able', 'ible', 'ive', 'al', 'ic', 'ous', 'ful', 'less', 'er', 'or', 'ist'
        }
        
        self.domain_roots = {
            # CAD/Manufacturing
            'extrude', 'revolve', 'loft', 'sweep', 'fillet', 'chamfer', 'shell', 'draft',
            'serialize', 'deserialize', 'batch', 'inventory', 'warehouse', 'production',
            'material', 'component', 'assembly', 'operation', 'cnc', 'gcode', 'toolpath',
            # Engineering
            'metric', 'dimension', 'tolerance', 'constraint', 'parameter', 'variable',
            'offset', 'pattern', 'array', 'mirror', 'scale', 'rotate', 'translate',
            # Software/API
            'api', 'endpoint', 'webhook', 'authentication', 'authorization', 'token',
            'json', 'xml', 'http', 'rest', 'graphql', 'sdk', 'integration'
        }
        
        logger.info("=" * 80)
        logger.info("INITIALIZING GENERIC TERM DETECTOR (NO TRAINING REQUIRED)")
        logger.info("=" * 80)
        logger.info("\n[METHOD 1] ✓ Statistical Discrimination Score ready")
        logger.info("[METHOD 2] ✓ Comparative Frequency Analysis ready")
        logger.info("[METHOD 5] ✓ Semantic Similarity ready")
        logger.info("\n[SUCCESS] Generic Term Detector initialized successfully!")
        logger.info("          All methods work WITHOUT historical training data!")
        logger.info("=" * 80)
        
    def detect_generic_term(self, term: str, contexts: List[str] = None) -> Dict:
        """
        Detect if a term is generic using ensemble voting
        
        Args:
            term: The term to check
            contexts: Optional list of context strings where term appears
        
        Returns:
            Dict with is_generic, confidence, penalty, votes, reasoning, etc.
        """
        # Check if term has technical protections
        has_protection, protection_reason = self._check_technical_protection(term)
        
        # Run all detection methods
        method_results = {}
        
        # Method 1: Statistical Discrimination Score
        method_results['method1'] = self._detect_method1(term, contexts)
        
        # Method 2: Comparative Frequency Analysis
        method_results['method2'] = self._detect_method2(term)
        
        # Method 5: Semantic Similarity
        method_results['method5'] = self._detect_method5(term)
        
        # Count votes (how many methods flagged it as generic)
        votes = sum(1 for r in method_results.values() if r['is_generic'])
        total_confidence = sum(r['confidence'] for r in method_results.values()) / 3
        
        # Determine penalty using graduated system
        base_penalty = self._calculate_graduated_penalty(votes, total_confidence)
        
        # Apply technical term protection (reduce penalty by 50% if protected)
        if has_protection and base_penalty < 0:
            final_penalty = base_penalty * 0.5
            protection_applied = True
        else:
            final_penalty = base_penalty
            protection_applied = False
        
        # Build reasoning
        reasoning_parts = []
        if votes > 0:
            flagged_methods = [name for name, result in method_results.items() if result['is_generic']]
            reasoning_parts.append(f"Flagged by {votes}/3 methods: {', '.join(flagged_methods)}")
            for method_name, result in method_results.items():
                if result['is_generic']:
                    reasoning_parts.append(f"  • {method_name}: {result['reason']}")
        
        if has_protection:
            reasoning_parts.append(f"Technical protection: {protection_reason}")
            if protection_applied:
                reasoning_parts.append(f"Penalty reduced by 50% (from {base_penalty:.3f} to {final_penalty:.3f})")
        
        if votes == 0:
            reasoning_parts.append("No methods flagged this term as generic")
        
        return {
            'is_generic': votes >= 2,  # Need at least 2 out of 3 methods to agree
            'confidence': total_confidence,
            'penalty': final_penalty,
            'method_votes': {name: result['is_generic'] for name, result in method_results.items()},
            'votes': votes,
            'has_protection': has_protection,
            'protection_reason': protection_reason if has_protection else None,
            'reasoning': ' | '.join(reasoning_parts),
            'method_details': method_results
        }
    
    def _calculate_graduated_penalty(self, votes: int, confidence: float) -> float:
        """
        Calculate penalty based on number of methods that flagged the term
        
        Graduated penalties:
        - 3 votes: -0.15 (very likely generic)
        - 2 votes: -0.07 (likely generic)
        - 1 vote:  -0.03 (possibly generic)
        - 0 votes:  0.0  (likely technical)
        """
        if votes >= 3:
            # All methods agree: strong penalty
            return -0.15
        elif votes == 2:
            # Majority agrees: moderate penalty
            return -0.07
        elif votes == 1:
            # Weak signal: light penalty
            return -0.03
        else:
            # No agreement: no penalty
            return 0.0
    
    def _check_technical_protection(self, term: str) -> Tuple[bool, str]:
        """
        Check if term has technical protections that should reduce penalty
        
        Returns:
            (has_protection, reason)
        """
        term_lower = term.lower()
        words = term_lower.split()
        
        # Check for technical suffixes
        for word in words:
            for suffix in self.technical_suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    return True, f"Has technical suffix '-{suffix}'"
        
        # Check for domain-specific roots
        for word in words:
            for root in self.domain_roots:
                if root in word:
                    return True, f"Contains domain root '{root}'"
        
        # Check for compound technical terms (3+ words often technical)
        if len(words) >= 3:
            return True, "Multi-word compound term"
        
        return False, ""
    
    # ========================================================================
    # METHOD 1: STATISTICAL DISCRIMINATION SCORE (SDS)
    # ========================================================================
    
    def _detect_method1(self, term: str, contexts: List[str] = None) -> Dict:
        """Detect generic term using Statistical Discrimination Score (NO TRAINING REQUIRED)"""
        # Method 1 works WITHOUT training if we have real-time contexts
        if not contexts or len(contexts) < 3:
            return {'is_generic': False, 'confidence': 0.0, 'reason': 'Method 1: Insufficient context data (need 3+ contexts)'}
        
        # Analyze real-time context distribution
        if len(contexts) >= 3:
            doc_types = []
            for context in contexts:
                context_lower = context.lower()
                if any(kw in context_lower for kw in ['click', 'button', 'screen', 'interface', 'ui', 'menu']):
                    doc_types.append('UI')
                elif any(kw in context_lower for kw in ['api', 'endpoint', 'json', 'http', 'request', 'response']):
                    doc_types.append('API')
                elif any(kw in context_lower for kw in ['product', 'inventory', 'warehouse', 'material', 'stock']):
                    doc_types.append('Inventory')
                elif any(kw in context_lower for kw in ['production', 'operation', 'worker', 'machine', 'shift']):
                    doc_types.append('Production')
                elif any(kw in context_lower for kw in ['cad', '3d', 'model', 'extrude', 'revolve', 'assembly']):
                    doc_types.append('CAD')
                elif any(kw in context_lower for kw in ['report', 'dashboard', 'analytics', 'chart', 'graph']):
                    doc_types.append('Analytics')
                else:
                    doc_types.append('General')
            
            if doc_types:
                type_counts = Counter(doc_types)
                unique_types = len(type_counts)
                total = len(doc_types)
                probs = [count / total for count in type_counts.values()]
                term_entropy = entropy(probs)
                
                # Low entropy (< 1.0) and appears in many types = generic
                # High entropy (> 1.5) or few types = technical
                if term_entropy < 1.0 and unique_types >= 4:
                    return {
                        'is_generic': True,
                        'confidence': 0.7,
                        'reason': f'Appears uniformly across {unique_types} document types (entropy: {term_entropy:.2f})'
                    }
        
        return {'is_generic': False, 'confidence': 0.3, 'reason': 'Appears in specific contexts only'}
    
    # ========================================================================
    # METHOD 2: COMPARATIVE FREQUENCY ANALYSIS (CFA)
    # ========================================================================
    
    def _detect_method2(self, term: str) -> Dict:
        """Detect generic term using Comparative Frequency Analysis (NO TRAINING REQUIRED - HEURISTIC MODE)"""
        # Method 2 works WITHOUT training by using heuristics based on term characteristics
        
        term_lower = term.lower()
        words = term_lower.split()
        
        # Heuristic 1: Very short common words (likely generic)
        if len(words) == 1 and len(term_lower) <= 4:
            # Common short generic words
            very_common_short = {'add', 'new', 'old', 'get', 'set', 'put', 'run', 'use', 'end', 'top', 'big', 'low', 'high', 'read', 'last', 'next', 'prev', 'back', 'main', 'view'}
            if term_lower in very_common_short:
                return {'is_generic': True, 'confidence': 0.85, 'reason': 'Very common short word'}
        
        # Heuristic 2: Common action verbs (likely generic)
        common_actions = {'add', 'create', 'delete', 'edit', 'move', 'copy', 'paste', 'cut', 'open', 'close', 'save', 'load', 'update', 'insert', 'remove', 'clear', 'reset', 'undo', 'redo', 'select', 'click'}
        if len(words) == 1 and term_lower in common_actions:
            return {'is_generic': True, 'confidence': 0.80, 'reason': 'Common action verb'}
        
        # Heuristic 3: Common adjectives/descriptors (likely generic)
        common_adjectives = {'new', 'old', 'big', 'small', 'easy', 'hard', 'fast', 'slow', 'high', 'low', 'simple', 'basic', 'advanced', 'general', 'common', 'normal', 'standard', 'default', 'custom', 'quick', 'different'}
        if len(words) == 1 and term_lower in common_adjectives:
            return {'is_generic': True, 'confidence': 0.75, 'reason': 'Common adjective/descriptor'}
        
        # Heuristic 4: Common UI/interaction terms (likely generic)
        common_ui = {'button', 'click', 'menu', 'dialog', 'window', 'screen', 'panel', 'tab', 'icon', 'label', 'field', 'form', 'page', 'view', 'list', 'table', 'row', 'column'}
        if len(words) == 1 and term_lower in common_ui:
            return {'is_generic': True, 'confidence': 0.70, 'reason': 'Common UI term'}
        
        # Heuristic 5: Color/appearance terms (likely generic unless compound)
        colors = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'orange', 'purple', 'pink', 'brown'}
        if len(words) == 1 and term_lower in colors:
            return {'is_generic': True, 'confidence': 0.85, 'reason': 'Color term'}
        
        # Heuristic 6: Common prepositions/directions in single-word form (likely generic)
        directions = {'up', 'down', 'left', 'right', 'top', 'bottom', 'front', 'back', 'next', 'previous', 'first', 'last'}
        if len(words) == 1 and term_lower in directions:
            return {'is_generic': True, 'confidence': 0.80, 'reason': 'Common direction/position term'}
        
        # Heuristic 7: Multi-word terms starting with generic action verbs (likely generic UI actions)
        if len(words) >= 2:
            first_word = words[0]
            # "add inventory", "create work", "new quality", "save client", "edit product"
            if first_word in common_actions or first_word in {'new', 'old'}:
                return {'is_generic': True, 'confidence': 0.75, 'reason': f'Generic action pattern: "{first_word} ..."'}
            
            # Terms ending with common UI elements: "inventory menu", "rework button", "scheduling view"
            last_word = words[-1]
            if last_word in common_ui:
                return {'is_generic': True, 'confidence': 0.70, 'reason': f'UI pattern: "... {last_word}"'}
        
        # If none of the heuristics match, it's likely technical
        return {'is_generic': False, 'confidence': 0.3, 'reason': 'No generic patterns detected'}
    
    # ========================================================================
    # METHOD 5: SEMANTIC SIMILARITY TO KNOWN GENERICS
    # ========================================================================
    
    def _detect_method5(self, term: str) -> Dict:
        """Detect generic term using Semantic Similarity (NO TRAINING REQUIRED - PATTERN MATCHING MODE)"""
        # Method 5 works WITHOUT training by using known generic term patterns
        
        term_lower = term.lower()
        
        # Known generic term patterns (semantically similar to common generics)
        generic_patterns = {
            # Action patterns
            'add', 'remove', 'create', 'delete', 'edit', 'modify', 'change', 'update', 
            'insert', 'append', 'push', 'pull', 'get', 'set', 'put', 'fetch',
            'open', 'close', 'start', 'stop', 'begin', 'end', 'finish', 'complete',
            'save', 'load', 'export', 'import', 'upload', 'download',
            'click', 'select', 'choose', 'pick', 'activate', 'enable', 'disable',
            'show', 'hide', 'display', 'view', 'preview', 'refresh', 'reload',
            
            # State/quality descriptors
            'new', 'old', 'current', 'previous', 'next', 'first', 'last',
            'active', 'inactive', 'enabled', 'disabled', 'available', 'unavailable',
            'valid', 'invalid', 'empty', 'full', 'complete', 'incomplete',
            'simple', 'complex', 'easy', 'hard', 'basic', 'advanced',
            'general', 'specific', 'common', 'uncommon', 'normal', 'abnormal',
            'standard', 'custom', 'default', 'optional', 'required',
            'quick', 'slow', 'fast', 'rapid', 'instant', 'immediate',
            'different', 'same', 'similar', 'identical', 'unique',
            
            # UI/interaction
            'button', 'link', 'menu', 'dialog', 'window', 'screen', 'page',
            'panel', 'tab', 'pane', 'section', 'area', 'region', 'zone',
            'field', 'box', 'form', 'input', 'output', 'label', 'caption',
            'icon', 'image', 'picture', 'photo', 'graphic', 'chart', 'graph',
            'list', 'table', 'grid', 'tree', 'row', 'column', 'cell', 'item',
            
            # Size/measurement (generic unless compound)
            'small', 'large', 'big', 'tiny', 'huge', 'wide', 'narrow',
            'high', 'low', 'tall', 'short', 'long', 'brief',
            'top', 'bottom', 'left', 'right', 'center', 'middle',
            'up', 'down', 'front', 'back', 'side', 'corner', 'edge',
            
            # Colors
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown',
            
            # Time
            'time', 'date', 'day', 'week', 'month', 'year', 'hour', 'minute',
            'second', 'today', 'yesterday', 'tomorrow', 'now', 'later', 'soon',
            
            # Generic nouns
            'thing', 'item', 'object', 'element', 'component', 'part', 'piece',
            'data', 'info', 'information', 'detail', 'content', 'text', 'value',
            'number', 'count', 'total', 'amount', 'quantity', 'size', 'length',
            'name', 'title', 'description', 'note', 'comment', 'message',
            'type', 'kind', 'category', 'class', 'group', 'set', 'collection',
            'status', 'state', 'mode', 'option', 'setting', 'preference',
            'file', 'folder', 'directory', 'path', 'location', 'place', 'position',
            'user', 'admin', 'owner', 'member', 'person', 'account', 'profile',
            'system', 'application', 'program', 'tool', 'utility', 'feature',
            'error', 'warning', 'alert', 'notification', 'message',
            'language', 'format', 'version', 'level', 'priority', 'order'
        }
        
        # Check if term matches any generic pattern (exact match for single words)
        words = term_lower.split()
        if len(words) == 1 and term_lower in generic_patterns:
            return {'is_generic': True, 'confidence': 0.75, 'reason': 'Matches known generic term pattern'}
        
        # Check if term contains generic root words (for multi-word terms)
        if len(words) > 1:
            generic_word_count = sum(1 for word in words if word in generic_patterns)
            
            # For 2-word terms: if first word is generic action/UI, likely generic
            if len(words) == 2 and words[0] in generic_patterns:
                # Examples: "add inventory", "create work", "new quality", "save client"
                # These are likely generic UI actions, not technical terms
                action_verbs = {'add', 'create', 'delete', 'edit', 'remove', 'new', 'old', 'save', 'load', 'open', 'close', 'view', 'show', 'hide', 'select', 'click'}
                ui_terms = {'menu', 'button', 'view', 'screen', 'dialog', 'panel', 'tab', 'page'}
                
                if words[0] in action_verbs or words[1] in ui_terms:
                    return {
                        'is_generic': True,
                        'confidence': 0.70,
                        'reason': f'Generic action/UI pattern: "{words[0]} {words[1]}"'
                    }
            
            # For any multi-word: if 50%+ are generic, likely generic
            if generic_word_count >= len(words) * 0.5:  # Lowered from 60% to 50%
                return {
                    'is_generic': True, 
                    'confidence': 0.60, 
                    'reason': f'Contains {generic_word_count}/{len(words)} generic words'
                }
        
        return {'is_generic': False, 'confidence': 0.4, 'reason': 'Does not match generic patterns'}


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def apply_generic_term_penalty(term: str, base_score: float, detector: GenericTermDetectorEnsemble,
                               contexts: List[str] = None) -> Tuple[float, Dict]:
    """
    Apply generic term penalty to a base score
    
    Args:
        term: The term to check
        base_score: The current comprehensive score
        detector: GenericTermDetectorEnsemble instance
        contexts: Optional list of context strings for the term
    
    Returns:
        (final_score, detection_info)
    """
    detection_result = detector.detect_generic_term(term, contexts)
    
    penalty = detection_result['penalty']
    final_score = max(0.0, min(1.0, base_score + penalty))  # Clamp to [0, 1]
    
    return final_score, detection_result

