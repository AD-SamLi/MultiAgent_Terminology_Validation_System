#!/usr/bin/env python3
"""
Test the terminology agent setup and basic functionality
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        # Test terminology_tool import
        from terminology_tool import TerminologyTool
        print("✅ terminology_tool imported successfully")
        
        # Test smolagents imports
        from smolagents import CodeAgent, Tool
        print("✅ smolagents imported successfully")
        
        # Test Azure imports
        from azure.identity import EnvironmentCredential
        print("✅ Azure identity imported successfully")
        
        # Test our custom modules (now independent)
        from terminology_agent import TerminologySmolTool, TerminologyAgent
        print("✅ terminology_agent imported successfully (now independent)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMissing dependencies? Try:")
        print("pip install smolagents[toolkit] azure-identity python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_glossary_folder():
    """Test glossary folder setup"""
    print("\n📁 Testing glossary folder setup...")
    
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    print(f"Looking for glossary folder: {glossary_folder}")
    
    if not os.path.exists(glossary_folder):
        print(f"⚠️  Glossary folder not found: {glossary_folder}")
        print("   This is expected if you haven't set up glossaries yet")
        return False
    
    # Check for data subfolder
    data_folder = os.path.join(glossary_folder, "data")
    if os.path.exists(data_folder):
        print(f"✅ Found data folder: {data_folder}")
        
        # Look for CSV files
        csv_files = list(Path(data_folder).rglob("*.csv"))
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files")
            for csv_file in csv_files[:5]:  # Show first 5
                print(f"   - {csv_file.name}")
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more")
        else:
            print("⚠️  No CSV files found in data folder")
    else:
        print(f"⚠️  No data subfolder found in {glossary_folder}")
    
    return True

def test_terminology_tool():
    """Test basic terminology tool functionality"""
    print("\n🔧 Testing terminology tool...")
    
    try:
        from terminology_tool import TerminologyTool
        
        glossary_folder = os.path.join(os.getcwd(), "glossary")
        tool = TerminologyTool(glossary_folder)
        
        # Test basic methods
        languages = tool.get_available_languages()
        print(f"✅ Available languages: {languages}")
        
        pairs = tool.get_available_language_pairs()
        print(f"✅ Available language pairs: {pairs}")
        
        # Test with sample text
        sample_text = "Hello vertex compilation"
        if pairs:
            src, tgt = pairs[0] if pairs else ("EN", "CS")
            used_terms = tool.get_used_terms(sample_text, src, tgt)
            print(f"✅ Found terms in '{sample_text}': {used_terms}")
        
        return True
        
    except Exception as e:
        print(f"❌ Terminology tool test failed: {e}")
        return False

def test_environment_variables():
    """Test Azure OpenAI environment variables"""
    print("\n🔐 Testing environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_CLIENT_ID", 
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # Show partial value for security
            if len(value) > 20:
                display_value = f"{value[:10]}...{value[-10:]}"
            else:
                display_value = f"{value[:5]}..."
            print(f"✅ {var}: {display_value}")
        else:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Please ensure these are set in your .env file")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🌟 TERMINOLOGY AGENT SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Glossary Folder", test_glossary_folder),
        ("Terminology Tool", test_terminology_tool),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to use terminology agent.")
        print("\n📋 Next steps:")
        print("1. Run: python run_terminology_agent.py")
        print("2. Or run: python terminology_agent.py (for full demo)")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before using the agent.")
        
        if not results.get("Environment Variables", False):
            print("\n💡 For Azure OpenAI setup:")
            print("   - Create a .env file with your Azure credentials")
            print("   - Set AZURE_OPENAI_ENDPOINT, AZURE_CLIENT_ID, etc.")
        
        if not results.get("Glossary Folder", False):
            print("\n💡 For glossary setup:")
            print("   - Create a 'glossary' folder in your project")
            print("   - Add CSV files with 'source' and 'target' columns")
            print("   - Or use the provided sample glossaries")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
