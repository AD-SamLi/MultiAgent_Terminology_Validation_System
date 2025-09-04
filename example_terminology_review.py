#!/usr/bin/env python3
"""
Example script demonstrating the Terminology Review Agent
Shows how to validate term candidates for Autodesk using web search
"""

import os
import sys
from terminology_review_agent import TerminologyReviewAgent


def main():
    """Example usage of the Terminology Review Agent"""
    
    print("🔍 TERMINOLOGY REVIEW AGENT - EXAMPLE USAGE")
    print("=" * 60)
    
    # Setup
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    
    # Initialize the agent (use gpt-4.1 for faster testing, gpt-5 for best results)
    print("🤖 Initializing Terminology Review Agent...")
    agent = TerminologyReviewAgent(glossary_folder, model_name="gpt-4.1")
    
    # Example 1: Validate a single term candidate
    print("\n📝 Example 1: Validating single term candidate")
    print("-" * 50)
    
    term_to_validate = "parametric modeling"
    print(f"Validating term: '{term_to_validate}'")
    
    try:
        result = agent.validate_term_candidate(
            term=term_to_validate,
            src_lang="EN",
            tgt_lang="DE",
            industry_context="CAD",
            save_to_file="validation_result.json"
        )
        print("✅ Validation completed!")
        print("📄 Results saved to: validation_result.json")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Example 2: Batch validate multiple terms
    print("\n📝 Example 2: Batch validating term candidates")
    print("-" * 50)
    
    candidate_terms = [
        "mesh generation",
        "surface tessellation", 
        "boolean union",
        "parametric constraints"
    ]
    
    print(f"Validating {len(candidate_terms)} terms: {candidate_terms}")
    
    try:
        batch_result = agent.batch_validate_terms(
            terms=candidate_terms,
            src_lang="EN", 
            tgt_lang="CS",
            industry_context="CAD",
            save_to_file="batch_validation.json"
        )
        print("✅ Batch validation completed!")
        print("📄 Results saved to: batch_validation.json")
        
    except Exception as e:
        print(f"❌ Batch validation failed: {e}")
    
    # Example 3: Research industry usage
    print("\n📝 Example 3: Researching industry usage")
    print("-" * 50)
    
    research_term = "NURBS surface"
    print(f"Researching industry usage for: '{research_term}'")
    
    try:
        research_result = agent.research_industry_usage(
            term=research_term,
            industry_context="CAD",
            save_to_file="industry_research.json"
        )
        print("✅ Industry research completed!")
        print("📄 Results saved to: industry_research.json")
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
    
    print("\n🎉 Example completed! Check the generated JSON files for detailed results.")
    print("\n💡 Tips:")
    print("   • Use 'CAD', 'AEC', or 'Manufacturing' for industry_context")
    print("   • JSON files contain comprehensive validation data")
    print("   • Web search provides real-time industry usage validation")
    print("   • Existing Autodesk glossaries inform validation decisions")


if __name__ == "__main__":
    main()
