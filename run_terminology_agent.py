#!/usr/bin/env python3
"""
Simple runner for the Terminology Agent
Provides easy examples and testing scenarios
"""

import os
import sys
from terminology_agent import TerminologyAgent

def main():
    """Main function with interactive menu"""
    
    print("🌟 TERMINOLOGY AGENT - Azure OpenAI + smolagents")
    print("=" * 55)
    
    # Set up glossary folder
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"📁 Glossary folder: {glossary_folder}")
    
    if not os.path.exists(glossary_folder):
        print(f"⚠️  Glossary folder not found: {glossary_folder}")
        print("   Please ensure the 'glossary' folder exists with your CSV files")
        return False
    
    # Choose model
    print("\n🤖 Choose model:")
    print("1. GPT-5 (slower but more capable, requires approval)")
    print("2. GPT-4.1 (faster, no special approval needed)")
    
    choice = input("Enter choice (1 or 2, default=2): ").strip()
    model_name = "gpt-5" if choice == "1" else "gpt-4.1"
    
    try:
        # Initialize agent
        print(f"\n🔧 Initializing agent with {model_name}...")
        agent = TerminologyAgent(glossary_folder, model_name=model_name)
        
        # Show menu
        while True:
            print("\n" + "=" * 55)
            print("📋 TERMINOLOGY AGENT MENU")
            print("=" * 55)
            print("1. Get glossary overview")
            print("2. Analyze text for terminology")
            print("3. Translate with terminology consistency")
            print("4. Custom terminology query")
            print("5. Quick terminology check")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            
            elif choice == "1":
                print("\n📊 Getting glossary overview...")
                result = agent.get_glossary_overview()
                print("\n" + "="*40)
                print(result)
            
            elif choice == "2":
                print("\n📝 Text terminology analysis")
                text = input("Enter text to analyze: ").strip()
                if not text:
                    text = "The AutoCAD file contains vertex data that needs to be compiled."
                    print(f"Using default text: {text}")
                
                src_lang = input("Source language (default=EN): ").strip() or "EN"
                tgt_lang = input("Target language (default=CS): ").strip() or "CS"
                
                print(f"\n🔍 Analyzing text for {src_lang}-{tgt_lang} terminology...")
                result = agent.analyze_text_terminology(text, src_lang, tgt_lang)
                print("\n" + "="*40)
                print(result)
            
            elif choice == "3":
                print("\n🌐 Text translation with terminology")
                text = input("Enter text to translate: ").strip()
                if not text:
                    text = "Please compile the template and import the file with correct vertex orientation."
                    print(f"Using default text: {text}")
                
                src_lang = input("Source language (default=EN): ").strip() or "EN"
                tgt_lang = input("Target language (default=CS): ").strip() or "CS"
                
                print(f"\n🔄 Translating {src_lang}-{tgt_lang} with terminology consistency...")
                result = agent.translate_with_terminology(text, src_lang, tgt_lang)
                print("\n" + "="*40)
                print(result)
            
            elif choice == "4":
                print("\n💬 Custom terminology query")
                query = input("Enter your terminology question: ").strip()
                if not query:
                    query = "What language pairs are available and which has the most terms?"
                    print(f"Using default query: {query}")
                
                print(f"\n🤔 Processing custom query...")
                result = agent.run_terminology_task(query)
                print("\n" + "="*40)
                print(result)
            
            elif choice == "5":
                print("\n⚡ Quick terminology check")
                # Quick predefined examples
                examples = [
                    {
                        "text": "vertex",
                        "src": "EN", 
                        "tgt": "CS",
                        "description": "Check single term 'vertex'"
                    },
                    {
                        "text": "AutoCAD file import",
                        "src": "EN",
                        "tgt": "CS", 
                        "description": "Check technical phrase"
                    },
                    {
                        "text": "compile template orientation",
                        "src": "EN",
                        "tgt": "CS",
                        "description": "Check multiple technical terms"
                    }
                ]
                
                print("Available quick checks:")
                for i, ex in enumerate(examples, 1):
                    print(f"{i}. {ex['description']}: '{ex['text']}' ({ex['src']}-{ex['tgt']})")
                
                quick_choice = input("Choose example (1-3): ").strip()
                try:
                    ex_idx = int(quick_choice) - 1
                    if 0 <= ex_idx < len(examples):
                        ex = examples[ex_idx]
                        print(f"\n🔍 Quick check: {ex['description']}")
                        result = agent.analyze_text_terminology(ex['text'], ex['src'], ex['tgt'])
                        print("\n" + "="*40)
                        print(result)
                    else:
                        print("Invalid choice!")
                except ValueError:
                    print("Invalid choice!")
            
            else:
                print("Invalid choice! Please enter 0-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

