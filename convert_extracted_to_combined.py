#!/usr/bin/env python3
"""
Convert Term_Extracted_result.csv to Combined_Terms_Data.csv
Processes extracted terms and adds POS tagging to create the combined dataset
"""

import pandas as pd
import json
import re
from collections import defaultdict
from datetime import datetime
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK data ready")
except Exception as e:
    print(f"⚠️ NLTK download issue: {e}")

def clean_term(term):
    """Clean and normalize a term"""
    # Handle different data types
    if pd.isna(term):
        return None
    
    # If it's a dict (JSON object), extract the 'term' field
    if isinstance(term, dict):
        if 'term' in term:
            term = term['term']
        else:
            return None
    
    # Convert to string and clean
    term = str(term).strip()
    
    # Check if empty after conversion
    if not term:
        return None
    
    # Remove extra whitespace
    term = re.sub(r'\s+', ' ', term)
    
    # Remove quotes if they wrap the entire term
    if term.startswith('"') and term.endswith('"'):
        term = term[1:-1]
    if term.startswith("'") and term.endswith("'"):
        term = term[1:-1]
    
    # Skip very short or very long terms
    if len(term) < 2 or len(term) > 100:
        return None
    
    # Skip terms that are mostly numbers or special characters
    if re.match(r'^[0-9\.\-\+\s]+$', term):
        return None
    
    return term.lower()

def extract_terms_from_field(field_value):
    """Extract individual terms from a field that might contain multiple terms"""
    if pd.isna(field_value) or not str(field_value).strip():
        return []
    
    field_str = str(field_value)
    terms = []
    
    # Try to parse as JSON array first (for Final_Curated_Terms_Detailed format)
    try:
        if field_str.startswith('[') and field_str.endswith(']'):
            parsed = json.loads(field_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'term' in item:
                        # Extract term from JSON object like {"term": "acad", "confidence": 0.87}
                        terms.append(clean_term(item['term']))
                    else:
                        # Handle simple list items
                        terms.append(clean_term(item))
        else:
            # Split by common separators for other formats
            separators = [',', ';', '|', '\n']
            current_terms = [field_str]
            
            for sep in separators:
                new_terms = []
                for term in current_terms:
                    new_terms.extend(term.split(sep))
                current_terms = new_terms
            
            terms.extend([clean_term(t) for t in current_terms])
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ JSON parsing error for field: {str(field_value)[:100]}... Error: {e}")
        # Fallback: treat as single term
        terms.append(clean_term(field_str))
    
    # Filter out None values
    return [t for t in terms if t is not None]

def get_pos_tag(term):
    """Get POS tag for a term using NLTK"""
    try:
        # Tokenize and get POS tags
        tokens = word_tokenize(term)
        pos_tags = pos_tag(tokens)
        
        # Return the POS tag of the first (main) word
        if pos_tags:
            return pos_tags[0][1]
        else:
            return 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'

def convert_extracted_to_combined(input_file: str, output_file: str):
    """Convert Term_Extracted_result.csv to Combined_Terms_Data.csv format"""
    
    print("🔄 CONVERTING EXTRACTED TERMS TO COMBINED FORMAT")
    print("=" * 60)
    print(f"📂 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    
    # Load the extracted results
    print(f"\n📥 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} rows")
        print(f"📋 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Show sample of source data
    print(f"\n🔍 Sample source data:")
    for col in df.columns:
        if not df[col].isna().all():
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            print(f"   {col}: {str(sample_val)[:100]}...")
    
    # Process each row to extract terms
    print(f"\n🔄 Processing terms and generating POS tags...")
    
    combined_data = []
    processed_terms = set()  # Track unique term-text combinations
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"   📊 Processed {idx:,}/{len(df):,} rows...")
        
        original_text = str(row.get('Original_Text', '')).strip()
        if not original_text or original_text == 'nan':
            continue
        
        # Extract terms from Final_Curated_Terms_Detailed column
        all_terms = []
        
        # Use the Final_Curated_Terms_Detailed column which contains JSON with all term information
        if 'Final_Curated_Terms_Detailed' in df.columns:
            terms_from_field = extract_terms_from_field(row['Final_Curated_Terms_Detailed'])
            all_terms.extend(terms_from_field)
        else:
            # Fallback to other term columns if Final_Curated_Terms_Detailed is not available
            for col in df.columns:
                if 'term' in col.lower() and col != 'Original_Text':
                    terms_from_field = extract_terms_from_field(row[col])
                    all_terms.extend(terms_from_field)
        
        # Process each unique term
        for term in set(all_terms):  # Use set to avoid duplicates within same text
            if term:
                # Create unique key to avoid duplicate term-text pairs
                unique_key = f"{term}||{original_text}"
                
                if unique_key not in processed_terms:
                    processed_terms.add(unique_key)
                    
                    # Get POS tag
                    pos_tag_result = get_pos_tag(term)
                    
                    combined_data.append({
                        'Terms': term,
                        'pos_tags': pos_tag_result,
                        'Original Text': original_text
                    })
    
    print(f"✅ Processed {len(df):,} source rows")
    print(f"✅ Generated {len(combined_data):,} term entries")
    
    # Create DataFrame and save
    print(f"\n💾 Creating Combined_Terms_Data.csv...")
    
    combined_df = pd.DataFrame(combined_data)
    
    # Sort by term for consistency
    combined_df = combined_df.sort_values(['Terms', 'pos_tags']).reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Generate statistics
    print(f"\n📊 CONVERSION STATISTICS:")
    print(f"   📥 Source rows: {len(df):,}")
    print(f"   📤 Output rows: {len(combined_df):,}")
    print(f"   🔤 Unique terms: {combined_df['Terms'].nunique():,}")
    print(f"   📝 Unique texts: {combined_df['Original Text'].nunique():,}")
    print(f"   🏷️  POS tag distribution:")
    
    pos_counts = combined_df['pos_tags'].value_counts().head(10)
    for pos, count in pos_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"      {pos}: {count:,} ({percentage:.1f}%)")
    
    # Show sample of output
    print(f"\n🔍 Sample output data:")
    sample_df = combined_df.head(5)
    for idx, row in sample_df.iterrows():
        print(f"   Term: '{row['Terms']}' | POS: {row['pos_tags']} | Text: {row['Original Text'][:50]}...")
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📁 Output saved to: {output_file}")
    
    # File size comparison
    import os
    input_size = os.path.getsize(input_file) / (1024 * 1024)
    output_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"📊 File size: {input_size:.1f} MB → {output_size:.1f} MB")
    
    return True

def main():
    """Main conversion function"""
    input_file = "Term_Extracted_result.csv"
    output_file = "Combined_Terms_Data.csv"
    
    print("🔄 TERM EXTRACTION TO COMBINED DATA CONVERTER")
    print("=" * 60)
    print("Converts Term_Extracted_result.csv to Combined_Terms_Data.csv format")
    print("Adds POS tagging and creates the standard 3-column format")
    
    # Check if input file exists
    import os
    if not os.path.exists(input_file):
        print(f"\n❌ Input file not found: {input_file}")
        print("💡 Make sure you have the Term_Extracted_result.csv file in the current directory")
        return False
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"\n⚠️  Output file already exists: {output_file}")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Conversion cancelled")
            return False
    
    # Run the conversion
    try:
        success = convert_extracted_to_combined(input_file, output_file)
        
        if success:
            print(f"\n🎉 CONVERSION SUCCESSFUL!")
            print(f"   ✅ {output_file} created from {input_file}")
            print(f"   🔄 Ready for further processing with verify_terms_in_text.py")
            print(f"   📊 Ready for analysis with create_clean_csv.py")
        else:
            print(f"\n❌ Conversion failed")
            
        return success
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Conversion interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()

