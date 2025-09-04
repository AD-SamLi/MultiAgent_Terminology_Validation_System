import pandas as pd
import csv
from collections import defaultdict

def create_clean_csv_formats(input_file: str):
    """Create clean, simple CSV formats that are easy to work with"""
    print("CREATING CLEAN CSV FORMATS")
    print("="*50)
    
    # Load the original combined data
    print("Loading Combined_Terms_Data.csv...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
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
    
    # Create different clean formats
    create_simple_list_format(term_data, "Clean_Terms_Simple.csv")
    create_numbered_format(term_data, "Clean_Terms_Numbered.csv")
    create_length_limited_format(term_data, "Clean_Terms_Preview.csv")
    create_separate_files_format(term_data, "Clean_Terms_Summary.csv")

def create_simple_list_format(term_data, filename):
    """Create a simple format with basic separators"""
    print(f"\nCreating simple format: {filename}")
    
    output_data = []
    
    for term, data in term_data.items():
        pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
        original_texts_list = list(data['original_texts'])
        
        # Use simple separator that's less likely to conflict
        pos_tags_str = " ;; ".join(pos_tags_list)
        
        # For texts, just show count and first few examples
        texts_count = len(original_texts_list)
        if texts_count <= 3:
            texts_str = " ;; ".join(original_texts_list)
        else:
            sample_texts = original_texts_list[:3]
            texts_str = " ;; ".join(sample_texts) + f" ;; [... and {texts_count - 3} more texts]"
        
        output_data.append({
            'Terms': term,
            'POS_Tags': pos_tags_str,
            'Original_Texts': texts_str,
            'Text_Count': texts_count,
            'Frequency': data['frequency']
        })
    
    # Sort by frequency
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # Save with proper CSV settings
    output_df.to_csv(filename, index=False, encoding='utf-8', 
                     quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    
    print(f"✅ Created {filename} with {len(output_df)} terms")

def create_numbered_format(term_data, filename):
    """Create a format with numbered items for clarity"""
    print(f"\nCreating numbered format: {filename}")
    
    output_data = []
    
    for term, data in term_data.items():
        pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
        original_texts_list = list(data['original_texts'])
        
        # Number the POS tags
        if pos_tags_list:
            pos_tags_str = " | ".join([f"{i+1}.{tag}" for i, tag in enumerate(pos_tags_list)])
        else:
            pos_tags_str = ""
        
        # Show text count and samples
        texts_count = len(original_texts_list)
        if texts_count <= 5:
            texts_str = " | ".join([f"[{i+1}]{text[:100]}{'...' if len(text) > 100 else ''}" 
                                   for i, text in enumerate(original_texts_list)])
        else:
            sample_texts = original_texts_list[:5]
            texts_str = " | ".join([f"[{i+1}]{text[:100]}{'...' if len(text) > 100 else ''}" 
                                   for i, text in enumerate(sample_texts)])
            texts_str += f" | [... {texts_count - 5} more texts not shown]"
        
        output_data.append({
            'Terms': term,
            'POS_Tags_Numbered': pos_tags_str,
            'Sample_Texts': texts_str,
            'Total_Text_Count': texts_count,
            'Frequency': data['frequency']
        })
    
    # Sort by frequency
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # Save
    output_df.to_csv(filename, index=False, encoding='utf-8', 
                     quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    
    print(f"✅ Created {filename} with {len(output_df)} terms")

def create_length_limited_format(term_data, filename):
    """Create a format with length limits to prevent huge cells"""
    print(f"\nCreating length-limited format: {filename}")
    
    output_data = []
    MAX_CELL_LENGTH = 1000  # Limit each cell to 1000 characters
    
    for term, data in term_data.items():
        pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
        original_texts_list = list(data['original_texts'])
        
        # Limit POS tags string
        pos_tags_str = " | ".join(pos_tags_list)
        if len(pos_tags_str) > MAX_CELL_LENGTH:
            pos_tags_str = pos_tags_str[:MAX_CELL_LENGTH] + "... [TRUNCATED]"
        
        # Limit texts string  
        texts_preview = ""
        for i, text in enumerate(original_texts_list):
            text_preview = f"[{i+1}] {text}"
            if len(texts_preview + text_preview) > MAX_CELL_LENGTH:
                texts_preview += f"... [SHOWING {i} of {len(original_texts_list)} TEXTS]"
                break
            texts_preview += text_preview + " | "
        
        output_data.append({
            'Terms': term,
            'POS_Tags': pos_tags_str,
            'Text_Samples': texts_preview.rstrip(" | "),
            'POS_Count': len(pos_tags_list),
            'Text_Count': len(original_texts_list),
            'Frequency': data['frequency']
        })
    
    # Sort by frequency
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # Save
    output_df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Created {filename} with {len(output_df)} terms (cells limited to {MAX_CELL_LENGTH} chars)")

def create_separate_files_format(term_data, filename):
    """Create a summary file with details in separate files"""
    print(f"\nCreating summary format: {filename}")
    
    output_data = []
    
    for term, data in term_data.items():
        pos_tags_count = len(data['pos_tags'])
        texts_count = len(data['original_texts'])
        
        # Just summary info in main file
        output_data.append({
            'Terms': term,
            'POS_Variations_Count': pos_tags_count,
            'Unique_Texts_Count': texts_count,
            'Frequency': data['frequency']
        })
    
    # Sort by frequency
    output_df = pd.DataFrame(output_data)
    output_df = output_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # Save summary
    output_df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"✅ Created {filename} with {len(output_df)} terms (summary only)")
    
    # Create top terms detail file
    print("Creating detailed file for top 100 terms...")
    top_terms = output_df.head(100)
    
    detailed_data = []
    for _, row in top_terms.iterrows():
        term = row['Terms']
        data = term_data[term]
        
        pos_tags_list = sorted(list(data['pos_tags']))
        original_texts_list = list(data['original_texts'])
        
        detailed_data.append({
            'Terms': term,
            'Frequency': row['Frequency'],
            'POS_Tags': " | ".join(pos_tags_list),
            'First_3_Texts': " ||| ".join(original_texts_list[:3]),
            'Total_Unique_Texts': len(original_texts_list)
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_filename = "Clean_Terms_Top100_Details.csv"
    detailed_df.to_csv(detailed_filename, index=False, encoding='utf-8')
    
    print(f"✅ Created {detailed_filename} with top 100 terms detailed")

def test_all_files():
    """Test that all created files can be read properly"""
    files = [
        "Clean_Terms_Simple.csv",
        "Clean_Terms_Numbered.csv", 
        "Clean_Terms_Preview.csv",
        "Clean_Terms_Summary.csv",
        "Clean_Terms_Top100_Details.csv"
    ]
    
    print(f"\nTesting all created files...")
    for filename in files:
        try:
            df = pd.read_csv(filename)
            print(f"✅ {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if frequency column is numeric
            if 'Frequency' in df.columns:
                if pd.api.types.is_numeric_dtype(df['Frequency']):
                    print(f"   ✅ Frequency column is numeric")
                else:
                    print(f"   ❌ Frequency column is not numeric!")
            
        except Exception as e:
            print(f"❌ {filename}: Error reading - {e}")

if __name__ == "__main__":
    create_clean_csv_formats("Combined_Terms_Data.csv")
    test_all_files()
    
    print(f"\n" + "="*60)
    print("SUMMARY OF CLEAN FORMATS CREATED:")
    print("="*60)
    print("1. Clean_Terms_Simple.csv - Basic format with simple separators")
    print("2. Clean_Terms_Numbered.csv - Items are numbered for clarity")
    print("3. Clean_Terms_Preview.csv - Length-limited cells (max 1000 chars)")
    print("4. Clean_Terms_Summary.csv - Summary only (counts, no details)")
    print("5. Clean_Terms_Top100_Details.csv - Full details for top 100 terms")
    print("\nAll files use proper CSV formatting that won't break!")
