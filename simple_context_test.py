#!/usr/bin/env python3
"""
Simple test to demonstrate enhanced validation with original texts.
"""

import json
import os
from terminology_review_agent import TerminologyReviewAgent

def main():
    """Test enhanced validation through the agent interface"""
    
    print("🚀 TESTING ENHANCED VALIDATION WITH ORIGINAL TEXTS")
    print("=" * 60)
    
    # Sample original texts for "visualization"
    sample_texts = [
        "This parameter section controls the visualization of point properties of the mesh.",
        "Submit Visualization DWF/DXF Jobs for Inventor Files",
        "Use Revit® BIM software for visualization and analysis",
        "AutoCAD Civil 3D and Autodesk 3ds Max Design work together for project visualizations",
        "Visualization tools extend the design workflow with rendering and 3D animation"
    ]
    
    try:
        # Initialize agent
        print("🔧 Setting up terminology review agent...")
        glossary_folder = os.path.join(os.getcwd(), 'glossary')
        agent = TerminologyReviewAgent(glossary_folder, 'gpt-4.1')
        
        # Test with a prompt that includes original texts context
        prompt = f'''
        Please validate the term candidate "visualization" for Autodesk terminology.
        
        IMPORTANT: This term appears in the following original contexts from Autodesk documentation:
        
        Original Usage Examples:
        1. "This parameter section controls the visualization of point properties of the mesh."
        2. "Submit Visualization DWF/DXF Jobs for Inventor Files"
        3. "Use Revit® BIM software for visualization and analysis"
        4. "AutoCAD Civil 3D and Autodesk 3ds Max Design work together for project visualizations"
        5. "Visualization tools extend the design workflow with rendering and 3D animation"
        
        Use the terminology_reviewer tool with:
        - action: "validate_term"
        - term: "visualization"
        - src_lang: "EN"
        - tgt_lang: "CS"
        - industry_context: "CAD"
        - output_file: "context_enhanced_test.json"
        
        Focus on:
        1. Product integration (Revit, AutoCAD Civil 3D, 3ds Max)
        2. Technical usage contexts (mesh properties, DWF/DXF jobs, rendering)
        3. Domain relevance for CAD/BIM workflows
        4. Translation priority based on usage frequency
        '''
        
        print("📊 Running enhanced validation with original texts context...")
        result = agent.run_review_task(prompt)
        
        print("\n✅ Enhanced validation completed!")
        print("📄 Check context_enhanced_test.json for detailed results")
        
        # Try to load and display key results
        try:
            with open("context_enhanced_test.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"\n🎯 Validation Score: {data.get('validation_score', 'N/A')}")
                print(f"📊 Status: {data.get('status', 'N/A')}")
                if 'recommendations' in data:
                    print("\n💡 Key Recommendations:")
                    for rec in data['recommendations'][:5]:
                        print(f"  • {rec}")
        except:
            print("📄 Results saved to context_enhanced_test.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
