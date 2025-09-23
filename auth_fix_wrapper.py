#!/usr/bin/env python3
"""
Unified authentication fix wrapper for all Azure OpenAI agents
This ensures consistent token refresh behavior across all components
"""

import time
from datetime import datetime, timedelta
from typing import Any, Callable

def is_token_expired(error_msg: str) -> bool:
    """Check if error is due to expired token"""
    token_indicators = [
        "401", "unauthorized", "access token", "expired", "invalid audience",
        "token is missing", "authentication failed", "credential"
    ]
    return any(indicator in error_msg.lower() for indicator in token_indicators)

def is_server_error(error_msg: str) -> bool:
    """Check if error is a server error"""
    server_indicators = ["500", "502", "503", "504", "server error", "internal server error"]
    return any(indicator in error_msg.lower() for indicator in server_indicators)

def is_content_filter_error(error_msg: str) -> bool:
    """Check if error is due to content filtering"""
    filter_indicators = ["content filter", "content policy", "responsible ai", "filtered"]
    return any(indicator in error_msg.lower() for indicator in filter_indicators)

def robust_agent_call_with_retries(agent_func: Callable, *args, max_retries: int = 3, base_delay: float = 2.0, **kwargs) -> Any:
    """
    Universal retry wrapper for any agent function call with robust error handling
    
    Args:
        agent_func: The agent method to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result from the agent function call
    """
    
    for attempt in range(max_retries):
        try:
            # Check if the agent has a token refresh method and call it proactively
            agent_instance = None
            if hasattr(agent_func, '__self__'):
                agent_instance = agent_func.__self__
                
                # Try to refresh token proactively if needed
                if hasattr(agent_instance, 'model') and hasattr(agent_instance.model, '_refresh_token_if_needed'):
                    if agent_instance.model._refresh_token_if_needed():
                        print(f"🔄 [AuthFix] Proactive token refresh triggered for {agent_instance.__class__.__name__}")
                        if hasattr(agent_instance, 'refresh_model_token'):
                            agent_instance.refresh_model_token()
            
            # Execute the function
            result = agent_func(*args, **kwargs)
            
            if attempt > 0:
                print(f"✅ [AuthFix] Function succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ [AuthFix] Attempt {attempt + 1}/{max_retries} failed: {error_msg[:200]}...")
            
            if attempt == max_retries - 1:
                print(f"❌ [AuthFix] All {max_retries} attempts failed")
                raise e
            
            # Determine retry strategy based on error type
            if is_token_expired(error_msg):
                print("🔄 [AuthFix] Token expired - forcing refresh...")
                delay = base_delay * 2  # Longer delay for token issues
                
                # Force token refresh if possible
                if agent_instance and hasattr(agent_instance, 'refresh_model_token'):
                    try:
                        agent_instance.refresh_model_token()
                        print("✅ [AuthFix] Token refresh completed")
                    except Exception as refresh_error:
                        print(f"❌ [AuthFix] Token refresh failed: {refresh_error}")
                        
            elif is_server_error(error_msg):
                delay = base_delay * (2 ** attempt)  # Exponential backoff for server errors
                print(f"🔄 [AuthFix] Server error - exponential backoff ({delay:.1f}s)")
                
            elif is_content_filter_error(error_msg):
                delay = base_delay * (2 ** attempt)  # Exponential backoff for content filter
                print(f"🔄 [AuthFix] Content filter error - exponential backoff ({delay:.1f}s)")
                
            else:
                delay = base_delay * (1.5 ** attempt)  # Standard backoff
                print(f"🔄 [AuthFix] General error - standard backoff ({delay:.1f}s)")
            
            print(f"⏳ [AuthFix] Waiting {delay:.1f} seconds before retry...")
            time.sleep(delay)

def apply_auth_fix_to_agent(agent_instance):
    """
    Apply authentication fixes to an agent instance
    
    Args:
        agent_instance: The agent instance to fix
    """
    
    # Store original methods
    original_methods = {}
    
    # List of methods that typically make API calls
    api_methods = [
        'analyze_text_terminology',
        'validate_term_candidate',
        'validate_term_candidate_efficient', 
        'run_review_task',
        'run_terminology_task',
        'get_glossary_overview'
    ]
    
    for method_name in api_methods:
        if hasattr(agent_instance, method_name):
            original_method = getattr(agent_instance, method_name)
            original_methods[method_name] = original_method
            
            # Create wrapped method
            def create_wrapped_method(orig_method, name):
                def wrapped_method(*args, **kwargs):
                    return robust_agent_call_with_retries(orig_method, *args, **kwargs)
                wrapped_method.__name__ = name
                return wrapped_method
            
            # Replace the method with wrapped version
            wrapped_method = create_wrapped_method(original_method, method_name)
            setattr(agent_instance, method_name, wrapped_method)
            print(f"✅ [AuthFix] Applied authentication fix to {agent_instance.__class__.__name__}.{method_name}")
    
    # Store original methods for potential restoration
    agent_instance._original_methods = original_methods
    agent_instance._auth_fix_applied = True
    
    return agent_instance

def ensure_agent_auth_fix(agent_instance):
    """
    Ensure an agent has authentication fixes applied (idempotent)
    
    Args:
        agent_instance: The agent instance to check/fix
    
    Returns:
        The agent instance with auth fixes applied
    """
    
    if not hasattr(agent_instance, '_auth_fix_applied'):
        print(f"🔧 [AuthFix] Applying authentication fixes to {agent_instance.__class__.__name__}")
        return apply_auth_fix_to_agent(agent_instance)
    else:
        print(f"✅ [AuthFix] Authentication fixes already applied to {agent_instance.__class__.__name__}")
        return agent_instance

def test_agent_auth_fix():
    """Test the authentication fix with a real agent"""
    print("🧪 TESTING AUTHENTICATION FIX WRAPPER")
    print("=" * 50)
    
    try:
        from terminology_agent import TerminologyAgent
        from modern_terminology_review_agent import ModernTerminologyReviewAgent
        
        # Test with TerminologyAgent
        print("\n[TEST 1] Testing TerminologyAgent with auth fix...")
        term_agent = TerminologyAgent(
            glossary_folder="AutodeskGlossary_Terminology_Tool",
            model_name="gpt-4.1"
        )
        
        # Apply auth fix
        term_agent = ensure_agent_auth_fix(term_agent)
        
        # Test call
        result = term_agent.analyze_text_terminology("test term", "EN", "EN")
        print("✅ TerminologyAgent test passed")
        
        # Test with ModernTerminologyReviewAgent
        print("\n[TEST 2] Testing ModernTerminologyReviewAgent with auth fix...")
        review_agent = ModernTerminologyReviewAgent(model_name="gpt-4.1")
        
        # Apply auth fix
        review_agent = ensure_agent_auth_fix(review_agent)
        
        # Test call
        result = review_agent.validate_term_candidate("test term", "EN", "EN", "General")
        print("✅ ModernTerminologyReviewAgent test passed")
        
        print("\n🎉 All authentication fix tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication fix test failed: {e}")
        return False

if __name__ == "__main__":
    test_agent_auth_fix()

