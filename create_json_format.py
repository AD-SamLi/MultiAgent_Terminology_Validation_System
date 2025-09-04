import pandas as pd
import json
from collections import defaultdict

def create_complete_json_format(input_file: str, output_file: str):
    """Create a complete JSON format that preserves ALL data without any truncation"""
    print("CREATING COMPLETE JSON FORMAT")
    print("="*50)
    print("This will preserve ALL text data with no truncation!")
    
    # Load the cleaned combined data
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from original data")
    
    # Re-aggregate the data
    term_data = defaultdict(lambda: {
        'pos_tags': set(),
        'original_texts': set(),
        'frequency': 0
    })
    
    print("Aggregating data by unique terms...")
    for _, row in df.iterrows():
        term = str(row['Terms']) if pd.notna(row['Terms']) else 'UNKNOWN'
        pos_tag = str(row['pos_tags']) if pd.notna(row['pos_tags']) else ''
        original_text = str(row['Original Text']) if pd.notna(row['Original Text']) else ''
        
        if pos_tag.strip():
            term_data[term]['pos_tags'].add(pos_tag)
        
        if original_text.strip():
            term_data[term]['original_texts'].add(original_text)
        
        term_data[term]['frequency'] += 1
    
    print(f"Aggregated {len(term_data)} unique terms")
    
    # Create the complete JSON structure
    print("Building complete JSON structure...")
    complete_data = {
        "metadata": {
            "total_unique_terms": len(term_data),
            "total_original_entries": len(df),
            "description": "Complete CLEANED terms data with all verified POS tags and original texts preserved - every term confirmed to appear in its text",
            "format_version": "1.0"
        },
        "terms": []
    }
    
    # Convert each term to JSON format
    total_pos_variations = 0
    total_texts = 0
    
    for term, data in term_data.items():
        pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
        original_texts_list = list(data['original_texts'])  # Keep ALL texts - NO TRUNCATION
        
        # Count for statistics
        total_pos_variations += len(pos_tags_list)
        total_texts += len(original_texts_list)
        
        term_entry = {
            "term": term,
            "frequency": data['frequency'],
            "pos_tag_variations": {
                "count": len(pos_tags_list),
                "tags": pos_tags_list
            },
            "original_texts": {
                "count": len(original_texts_list),
                "texts": original_texts_list  # ALL texts preserved
            }
        }
        
        complete_data["terms"].append(term_entry)
    
    # Sort terms by frequency (most frequent first)
    complete_data["terms"].sort(key=lambda x: x["frequency"], reverse=True)
    
    # Update metadata with final statistics
    complete_data["metadata"]["total_pos_variations"] = total_pos_variations
    complete_data["metadata"]["total_original_texts"] = total_texts
    complete_data["metadata"]["most_frequent_term"] = complete_data["terms"][0]["term"]
    complete_data["metadata"]["highest_frequency"] = complete_data["terms"][0]["frequency"]
    
    print(f"JSON structure complete:")
    print(f"- {len(complete_data['terms'])} unique terms")
    print(f"- {total_pos_variations:,} total POS tag variations") 
    print(f"- {total_texts:,} total original texts (ALL preserved)")
    
    # Save to JSON file
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Complete JSON file created: {output_file}")
    
    # Calculate file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    return complete_data

def create_summary_json(complete_data, output_file: str):
    """Create a lighter summary JSON for quick overview"""
    print(f"\nCreating summary JSON: {output_file}")
    
    summary_data = {
        "metadata": complete_data["metadata"].copy(),
        "top_terms": []
    }
    
    # Include only top 1000 terms with limited text samples
    for term_entry in complete_data["terms"][:1000]:
        summary_entry = {
            "term": term_entry["term"],
            "frequency": term_entry["frequency"],
            "pos_tag_count": term_entry["pos_tag_variations"]["count"],
            "pos_tags": term_entry["pos_tag_variations"]["tags"],
            "original_text_count": term_entry["original_texts"]["count"],
            "sample_texts": term_entry["original_texts"]["texts"][:3]  # Only first 3 texts
        }
        summary_data["top_terms"].append(summary_entry)
    
    # Save summary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Calculate file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Summary JSON created: {output_file} ({file_size_mb:.1f} MB)")

def test_json_files():
    """Test that the JSON files can be loaded and used properly"""
    print(f"\nTesting JSON files...")
    
    # Test complete JSON
    try:
        print("Testing complete JSON...")
        with open("Cleaned_Complete_Terms_Data.json", 'r', encoding='utf-8') as f:
            complete_data = json.load(f)
        
        print(f"✅ Complete JSON loaded successfully")
        print(f"   - {len(complete_data['terms'])} terms")
        print(f"   - {complete_data['metadata']['total_original_texts']} total texts")
        
        # Test accessing a specific term
        first_term = complete_data["terms"][0]
        print(f"   - Most frequent term: '{first_term['term']}' ({first_term['frequency']} times)")
        print(f"   - Has {len(first_term['original_texts']['texts'])} original texts")
        print(f"   - Has {len(first_term['pos_tag_variations']['tags'])} POS variations")
        
    except Exception as e:
        print(f"❌ Error testing complete JSON: {e}")
    
    # Test summary JSON  
    try:
        print("\nTesting summary JSON...")
        with open("Cleaned_Summary_Terms_Data.json", 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        print(f"✅ Summary JSON loaded successfully")
        print(f"   - {len(summary_data['top_terms'])} top terms")
        
    except Exception as e:
        print(f"❌ Error testing summary JSON: {e}")

def show_json_usage_examples():
    """Show how to use the JSON files in Python"""
    print(f"\n" + "="*60)
    print("USAGE EXAMPLES - How to use the JSON files:")
    print("="*60)
    
    usage_examples = '''
# Load the complete data
import json
with open("Cleaned_Complete_Terms_Data.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get basic statistics
print(f"Total unique terms: {data['metadata']['total_unique_terms']}")
print(f"Total original texts: {data['metadata']['total_original_texts']}")

# Find a specific term
def find_term(term_name):
    for term_entry in data["terms"]:
        if term_entry["term"].lower() == term_name.lower():
            return term_entry
    return None

# Example: Get all data for "autocad"
autocad_data = find_term("autocad")
if autocad_data:
    print(f"AutoCAD appears {autocad_data['frequency']} times")
    print(f"POS variations: {autocad_data['pos_tag_variations']['tags']}")
    print(f"Number of unique texts: {len(autocad_data['original_texts']['texts'])}")
    # Access ALL original texts (no truncation!)
    all_autocad_texts = autocad_data['original_texts']['texts']

# Get top 10 most frequent terms
top_10 = data["terms"][:10]
for i, term in enumerate(top_10, 1):
    print(f"{i}. {term['term']}: {term['frequency']} times")

# Find terms with multiple POS tags
multi_pos_terms = [term for term in data["terms"] 
                   if term["pos_tag_variations"]["count"] > 5]
print(f"Terms with >5 POS variations: {len(multi_pos_terms)}")
'''
    
    print(usage_examples)

if __name__ == "__main__":
    print("JSON FORMAT CREATOR - PRESERVES ALL CLEANED DATA!")
    print("="*60)
    print("Creating JSON from verified Cleaned_Terms_Data.csv")
    
    # Create complete JSON (with ALL cleaned data)
    complete_data = create_complete_json_format("Cleaned_Terms_Data.csv", "Cleaned_Complete_Terms_Data.json")
    
    # Create summary JSON (lighter version)
    create_summary_json(complete_data, "Cleaned_Summary_Terms_Data.json")
    
    # Test the files
    test_json_files()
    
    # Show usage examples
    show_json_usage_examples()
    
    print(f"\n" + "="*60)
    print("CLEANED JSON FILES CREATED:")
    print("="*60)
    print("1. Cleaned_Complete_Terms_Data.json - ALL verified data preserved, no truncation")
    print("2. Cleaned_Summary_Terms_Data.json - Top 1000 verified terms with sample texts")
    print("")
    print("✅ JSON format preserves ALL your CLEANED data perfectly!")
    print("✅ Every term verified to appear in its original text!")
    print("✅ Easy to use in Python, R, JavaScript, etc.")
    print("✅ No CSV formatting issues!")
    print("✅ Perfect for high-quality data analysis and processing!")
