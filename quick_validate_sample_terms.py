#!/usr/bin/env python3
"""
Quick validation script for testing the terminology review agent
with a small sample of terms from New_Terms_Candidates_Clean.json
"""

import json
import os
from terminology_review_agent import TerminologyReviewAgent


def main():
    """Quick test with a small sample of terms"""
    
    print("🚀 QUICK TERM VALIDATION TEST")
    print("=" * 50)
    
    # Sample terms to test (manually selected interesting technical terms)
    sample_terms = [
        "parametric",
        "viewport", 
        "tessellation",
        "wireframe",
        "boolean"
    ]
    
    print(f"🎯 Testing with sample terms: {sample_terms}")
    
    # Initialize agent
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"📁 Using glossary folder: {glossary_folder}")
    
    try:
        # Use gpt-4.1 for faster testing (change to gpt-5 for best results)
        agent = TerminologyReviewAgent(glossary_folder, model_name="gpt-4.1")
        
        print("\n📝 Validating sample terms...")
        
        # Validate the sample terms as a batch
        result = agent.batch_validate_terms(
            terms=sample_terms,
            src_lang="EN",
            tgt_lang="CS", 
            industry_context="CAD",
            save_to_file="quick_validation_test.json"
        )
        
        print("\n✅ Validation completed!")
        print("📄 Results saved to: quick_validation_test.json")
        print("\n🔍 Agent Response:")
        print(result)
        
        print("\n💡 Next steps:")
        print("   • Check quick_validation_test.json for detailed results")
        print("   • Run validate_new_terms_candidates.py for full processing")
        print("   • Adjust parameters based on these test results")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
