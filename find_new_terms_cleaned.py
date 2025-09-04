#!/usr/bin/env python3
"""
Find new terms from Cleaned_Complete_Terms_Data.json that are not in existing glossaries
Uses the terminology agent to check against all available glossaries
Creates a comprehensive new terms candidate file with all information
"""

import json
import os
from typing import Dict, List, Set, Optional
from terminology_tool import TerminologyTool

def load_cleaned_terms_data(json_file: str) -> Dict:
    """Load the cleaned complete terms data"""
    print(f"📖 Loading cleaned terms data from: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {data['metadata']['total_unique_terms']:,} unique terms")
        print(f"   Total original entries: {data['metadata']['total_original_entries']:,}")
        print(f"   Total original texts: {data['metadata']['total_original_texts']:,}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading cleaned terms data: {e}")
        return {}

def get_all_glossary_terms(terminology_tool: TerminologyTool) -> Set[str]:
    """Get all unique terms from all available glossaries"""
    print("🔍 Extracting all terms from available glossaries...")
    
    all_terms = set()
    
    try:
        # Get all available language pairs
        language_pairs = terminology_tool.get_available_language_pairs()
        print(f"   Found {len(language_pairs)} language pairs")
        
        # Collect terms from all glossaries
        glossary_stats = {}
        
        for src_lang, tgt_lang in language_pairs:
            try:
                # Get terms for this language pair
                glossary = terminology_tool.get_relevant_terms(src_lang, tgt_lang)
                pair_key = f"{src_lang}-{tgt_lang}"
                glossary_stats[pair_key] = len(glossary)
                
                # Add all source terms to our set (case insensitive)
                for term in glossary.keys():
                    # Store terms in lowercase for comparison
                    all_terms.add(term.lower().strip())
                    
            except Exception as e:
                print(f"   ⚠️ Error loading {src_lang}-{tgt_lang}: {e}")
                continue
        
        print(f"✅ Collected {len(all_terms):,} unique terms from all glossaries")
        
        # Show statistics
        if glossary_stats:
            print("📊 Glossary statistics:")
            for pair, count in sorted(glossary_stats.items()):
                print(f"   {pair}: {count:,} terms")
        
        return all_terms
        
    except Exception as e:
        print(f"❌ Error getting glossary terms: {e}")
        return set()

def find_new_terms_candidates(terms_data: Dict, glossary_terms: Set[str]) -> List[Dict]:
    """Find terms that are not in any glossary"""
    print("\n🔍 Finding new terms candidates...")
    
    new_terms_candidates = []
    total_terms = len(terms_data['terms'])
    
    for idx, term_entry in enumerate(terms_data['terms']):
        if idx % 5000 == 0 and idx > 0:
            print(f"   Processed {idx:,}/{total_terms:,} terms...")
        
        term = term_entry['term']
        term_lower = term.lower().strip()
        
        # Skip empty terms
        if not term_lower:
            continue
        
        # Check if this term exists in any glossary
        if term_lower not in glossary_terms:
            # This is a new term candidate
            new_terms_candidates.append({
                "term": term,
                "frequency": term_entry['frequency'],
                "pos_tag_variations": term_entry['pos_tag_variations'],
                "original_texts": term_entry['original_texts'],
                "analysis": {
                    "length": len(term),
                    "word_count": len(term.split()),
                    "contains_numbers": any(c.isdigit() for c in term),
                    "contains_special_chars": any(not c.isalnum() and not c.isspace() for c in term),
                    "is_acronym": term.isupper() and len(term) <= 5,
                    "is_mixed_case": term != term.lower() and term != term.upper()
                }
            })
    
    print(f"✅ Found {len(new_terms_candidates):,} new terms candidates")
    return new_terms_candidates

def categorize_new_terms(new_terms: List[Dict]) -> Dict:
    """Categorize new terms by various criteria for better analysis"""
    print("📊 Categorizing new terms...")
    
    categories = {
        "by_frequency": {
            "critical": [],    # frequency >= 100
            "high": [],        # frequency >= 20
            "medium": [],      # frequency >= 5
            "low": []          # frequency < 5
        },
        "by_type": {
            "acronyms": [],
            "technical_terms": [],
            "compound_terms": [],
            "mixed_case": [],
            "with_numbers": [],
            "with_special_chars": [],
            "simple_terms": []
        },
        "by_domain": {
            "file_operations": [],
            "ui_elements": [],
            "technical_concepts": [],
            "commands": [],
            "general": []
        }
    }
    
    # Domain keywords for classification
    domain_keywords = {
        "file_operations": ["file", "save", "open", "import", "export", "load", "path", "directory", "folder"],
        "ui_elements": ["button", "menu", "dialog", "window", "panel", "tab", "toolbar", "icon"],
        "technical_concepts": ["layer", "vertex", "mesh", "geometry", "coordinate", "dimension", "scale"],
        "commands": ["compile", "run", "execute", "process", "generate", "create", "delete", "modify"]
    }
    
    for term_entry in new_terms:
        term = term_entry['term']
        frequency = term_entry['frequency']
        analysis = term_entry['analysis']
        
        # Categorize by frequency
        if frequency >= 100:
            categories["by_frequency"]["critical"].append(term_entry)
        elif frequency >= 20:
            categories["by_frequency"]["high"].append(term_entry)
        elif frequency >= 5:
            categories["by_frequency"]["medium"].append(term_entry)
        else:
            categories["by_frequency"]["low"].append(term_entry)
        
        # Categorize by type
        if analysis["is_acronym"]:
            categories["by_type"]["acronyms"].append(term_entry)
        elif analysis["is_mixed_case"]:
            categories["by_type"]["mixed_case"].append(term_entry)
        elif analysis["word_count"] > 1:
            categories["by_type"]["compound_terms"].append(term_entry)
        elif analysis["contains_numbers"]:
            categories["by_type"]["with_numbers"].append(term_entry)
        elif analysis["contains_special_chars"]:
            categories["by_type"]["with_special_chars"].append(term_entry)
        else:
            categories["by_type"]["simple_terms"].append(term_entry)
        
        # Categorize by domain
        term_lower = term.lower()
        domain_assigned = False
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in term_lower for keyword in keywords):
                categories["by_domain"][domain].append(term_entry)
                domain_assigned = True
                break
        
        if not domain_assigned:
            categories["by_domain"]["general"].append(term_entry)
    
    # Sort each category by frequency
    for category_type in categories:
        for category_name in categories[category_type]:
            categories[category_type][category_name].sort(key=lambda x: x['frequency'], reverse=True)
    
    return categories

def create_new_terms_file(new_terms: List[Dict], output_file: str) -> bool:
    """Create clean new terms candidate file with essential information only"""
    print(f"\n💾 Creating clean new terms candidate file: {output_file}")
    
    try:
        # Calculate basic statistics
        total_new_terms = len(new_terms)
        total_frequency = sum(term['frequency'] for term in new_terms)
        
        # Create clean, focused output structure - just the essential data
        clean_new_terms = []
        for term_entry in new_terms:
            clean_entry = {
                "term": term_entry['term'],
                "frequency": term_entry['frequency'],
                "pos_tag_variations": term_entry['pos_tag_variations'],
                "original_texts": term_entry['original_texts']
            }
            clean_new_terms.append(clean_entry)
        
        # Sort by frequency (most frequent first)
        clean_new_terms.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Simple output structure with just essential metadata and terms
        output_data = {
            "metadata": {
                "source": "Cleaned_Complete_Terms_Data.json",
                "description": "New terms candidates not found in existing glossaries - cleaned and verified data",
                "total_new_terms": total_new_terms,
                "total_combined_frequency": total_frequency,
                "analysis_date": __import__('datetime').datetime.now().isoformat(),
                "data_quality": "High - all terms verified to appear in their original texts"
            },
            "new_terms": clean_new_terms
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"✅ Clean new terms candidate file created successfully!")
        print(f"📊 Summary:")
        print(f"   Total new terms: {total_new_terms:,}")
        print(f"   Combined frequency: {total_frequency:,}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print(f"   File location: {output_file}")
        
        # Show simple priority breakdown
        critical = sum(1 for term in clean_new_terms if term['frequency'] >= 100)
        high = sum(1 for term in clean_new_terms if 20 <= term['frequency'] < 100)
        medium = sum(1 for term in clean_new_terms if 5 <= term['frequency'] < 20)
        low = sum(1 for term in clean_new_terms if term['frequency'] < 5)
        
        print(f"\n🎯 Priority Distribution:")
        print(f"   Critical (≥100x): {critical:,} terms")
        print(f"   High (≥20x): {high:,} terms")  
        print(f"   Medium (≥5x): {medium:,} terms")
        print(f"   Low (<5x): {low:,} terms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating new terms file: {e}")
        return False

def show_sample_new_terms(new_terms: List[Dict], num_samples: int = 20):
    """Show sample new terms for quick review"""
    print(f"\n🔍 Sample new terms (top {num_samples} by frequency):")
    print("-" * 80)
    
    # Show top terms by frequency
    top_terms = sorted(new_terms, key=lambda x: x['frequency'], reverse=True)[:num_samples]
    
    for i, term_entry in enumerate(top_terms, 1):
        term = term_entry['term']
        freq = term_entry['frequency']
        pos_count = term_entry['pos_tag_variations']['count']
        text_count = term_entry['original_texts']['count']
        
        print(f"{i:2}. '{term}' - {freq:,}x (POS: {pos_count}, Texts: {text_count:,})")
        
        # Show first POS tag as example
        if term_entry['pos_tag_variations']['tags']:
            print(f"     POS example: {term_entry['pos_tag_variations']['tags'][0]}")

def main():
    """Main function"""
    print("🔍 NEW TERMS CANDIDATE FINDER (FROM CLEANED DATA)")
    print("=" * 70)
    
    # File paths
    cleaned_json_file = "Cleaned_Complete_Terms_Data.json"
    output_file = "New_Terms_Candidates_Clean.json"
    glossary_folder = os.path.join(os.getcwd(), "glossary")
    
    # Check if files exist
    if not os.path.exists(cleaned_json_file):
        print(f"❌ Cleaned terms file not found: {cleaned_json_file}")
        return
    
    if not os.path.exists(glossary_folder):
        print(f"❌ Glossary folder not found: {glossary_folder}")
        return
    
    # Load cleaned terms data
    terms_data = load_cleaned_terms_data(cleaned_json_file)
    if not terms_data:
        print("❌ Failed to load cleaned terms data")
        return
    
    # Initialize terminology tool
    print(f"\n🔧 Initializing terminology tool with glossary folder: {glossary_folder}")
    try:
        terminology_tool = TerminologyTool(glossary_folder)
        print("✅ Terminology tool initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize terminology tool: {e}")
        return
    
    # Get all glossary terms
    glossary_terms = get_all_glossary_terms(terminology_tool)
    if not glossary_terms:
        print("❌ No glossary terms found")
        return
    
    # Find new terms candidates
    new_terms = find_new_terms_candidates(terms_data, glossary_terms)
    if not new_terms:
        print("🎉 No new terms found - all terms are already in glossaries!")
        return
    
    # Show sample new terms
    show_sample_new_terms(new_terms)
    
    # Create clean new terms file (without extensive categorization)
    success = create_new_terms_file(new_terms, output_file)
    
    if success:
        print(f"\n🎉 Process completed successfully!")
        print(f"✅ Cleaned terms analyzed: {terms_data['metadata']['total_unique_terms']:,}")
        print(f"✅ Glossary terms checked: {len(glossary_terms):,}")
        print(f"✅ New terms candidates found: {len(new_terms):,}")
        print(f"✅ Output file: {output_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the categorized new terms in {output_file}")
        print(f"   2. Prioritize critical and high-frequency terms for glossary addition")
        print(f"   3. Consider domain-specific terms for specialized glossaries")
        print(f"   4. Use the verified, cleaned data for high-quality terminology expansion")
    else:
        print(f"\n❌ Process failed")

if __name__ == "__main__":
    main()
