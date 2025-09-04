#!/usr/bin/env python3
"""
Test script to demonstrate enhanced validation with original texts context analysis.
"""

import json
import os
from terminology_review_agent import TerminologyReviewAgent

def test_enhanced_validation():
    """Test the enhanced validation with original texts"""
    
    # Sample original texts for "visualization" from the JSON file
    sample_original_texts = [
        "This parameter section controls the visualization of point properties of the mesh.",
        "Submit Visualization DWF/DXF Jobs for Inventor Files",
        "Added a visualization option on SlabShapeEditor to turn off control points at the user's preference.",
        "Use Revit® BIM (Building Information Modeling) software to drive efficiency and accuracy across the project lifecycle, from conceptual design, visualization, and analysis to fabrication and construction.",
        "With Infrastructure Design Suite, Autodesk offers a powerful BIM solution where AutoCAD Civil 3D and Autodesk 3ds Max Design work together using a process that enables virtually anyone who works on civil engineering design projects to produce compelling project visualizations, regardless of size.",
        "Use visualization, simulation, and water analysis tools to improve project delivery and decision-making",
        "The Change visualization interface helps multi-firm teams efficiently understand design changes in 3D models.",
        "Using several visualization themes for the surface, or by watching the trends of the convergence plots, you can stop and change designs or let it run until a desired grading solution is found.",
        "Visualization tools extend the design workflow with cinematic-quality rendering and 3D animation.",
        "Autodesk Building Design Suite gives you the power of BIM, with tools for modeling, visualization, and documentation all in a cost-effective solution so you can compete for new work, whether the project requires CAD or BIM."
    ]
    
    print("🚀 ENHANCED VALIDATION TEST WITH ORIGINAL TEXTS")
    print("=" * 60)
    print(f"🎯 Testing term: 'visualization'")
    print(f"📝 Using {len(sample_original_texts)} original text samples")
    print()
    
    try:
        # Initialize the agent
        print("🔧 Setting up terminology review agent...")
        glossary_folder = os.path.join(os.getcwd(), 'glossary')
        agent = TerminologyReviewAgent(glossary_folder, 'gpt-4.1')
        
        # Get the terminology review tool to test direct validation
        print("🔍 Looking for terminology review tool...")
        print(f"Available tools: {[type(tool).__name__ for tool in agent.agent.tools]}")
        
        review_tool = None
        for tool in agent.agent.tools:
            print(f"Checking tool: {type(tool).__name__}")
            if hasattr(tool, '_validate_single_term'):
                review_tool = tool
                print("✅ Found terminology review tool with _validate_single_term method")
                break
            elif 'TerminologyReview' in type(tool).__name__:
                review_tool = tool
                print("✅ Found TerminologyReviewTool")
                break
        
        if review_tool:
            # Test validation with original texts
            print("\n📊 Running enhanced validation...")
            result = review_tool._validate_single_term(
                term="visualization",
                src_lang="EN",
                tgt_lang="CS",
                industry_context="CAD",
                original_texts=sample_original_texts
            )
            
            print("\n🎯 ENHANCED VALIDATION RESULTS:")
            print("=" * 50)
            print(f"Term: {result['term']}")
            print(f"Validation Score: {result['validation_score']:.3f}")
            print(f"Status: {result['status']}")
            print()
            
            # Display context analysis
            if 'context_analysis' in result:
                context = result['context_analysis']
                print("📝 CONTEXT ANALYSIS:")
                print(f"  • Context Score: {context.get('context_score', 0):.3f}")
                print(f"  • Text Count: {context.get('text_count', 0)}")
                print(f"  • Product Mentions: {len(context.get('product_mentions', []))}")
                print(f"  • Technical Indicators: {len(context.get('technical_indicators', []))}")
                print(f"  • Usage Patterns: {len(context.get('usage_patterns', []))}")
                
                # Show some product mentions
                if context.get('product_mentions'):
                    print("\n🎯 PRODUCT MENTIONS:")
                    for mention in context['product_mentions'][:3]:
                        print(f"  • {mention['product']}: {mention['text'][:80]}...")
                
                # Show some technical indicators
                if context.get('technical_indicators'):
                    print("\n⚙️ TECHNICAL INDICATORS:")
                    for indicator in context['technical_indicators'][:3]:
                        keywords = ", ".join(indicator['keywords'])
                        print(f"  • Keywords: {keywords}")
                        print(f"    Context: {indicator['text'][:80]}...")
                
                print()
            
            # Display recommendations
            print("💡 RECOMMENDATIONS:")
            for rec in result.get('recommendations', []):
                print(f"  • {rec}")
            
            print()
            print("✅ Enhanced validation test completed successfully!")
            
            # Save detailed results
            output_file = "enhanced_validation_test_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"📄 Detailed results saved to: {output_file}")
            
        else:
            print("❌ Could not find terminology review tool")
            
    except Exception as e:
        print(f"❌ Error during enhanced validation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_validation()
