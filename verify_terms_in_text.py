#!/usr/bin/env python3
"""
Verify that extracted terms actually appear in their original texts
Remove rows where terms don't match and create a cleaned CSV file
"""

import pandas as pd
import re
import csv
from typing import List, Set

def check_term_in_text(term: str, text: str, case_sensitive: bool = False) -> bool:
    """
    Check if a term appears in the text
    
    Args:
        term: The term to search for
        text: The text to search in
        case_sensitive: Whether to perform case-sensitive matching
        
    Returns:
        True if term appears in text, False otherwise
    """
    if not term or not text:
        return False
    
    # Convert to string and strip whitespace
    term = str(term).strip()
    text = str(text).strip()
    
    if not term or not text:
        return False
    
    # Simple containment check
    if case_sensitive:
        return term in text
    else:
        return term.lower() in text.lower()

def check_term_variations(term: str, text: str) -> bool:
    """
    Check if a term or its variations appear in text
    This handles common NLP extraction variations
    """
    if not term or not text:
        return False
    
    term = str(term).strip()
    text = str(text).strip()
    
    if not term or not text:
        return False
    
    # Convert to lowercase for checking
    term_lower = term.lower()
    text_lower = text.lower()
    
    # Check exact match
    if term_lower in text_lower:
        return True
    
    # Check word boundaries (avoid partial matches in larger words)
    word_pattern = r'\b' + re.escape(term_lower) + r'\b'
    if re.search(word_pattern, text_lower):
        return True
    
    # Check for common variations
    # Remove common punctuation that might interfere
    clean_term = re.sub(r'[^\w\s-]', '', term_lower)
    if clean_term != term_lower and clean_term in text_lower:
        return True
    
    # Check for hyphenated versions
    if ' ' in term_lower:
        hyphenated = term_lower.replace(' ', '-')
        if hyphenated in text_lower:
            return True
    
    if '-' in term_lower:
        spaced = term_lower.replace('-', ' ')
        if spaced in text_lower:
            return True
    
    # Check for possessive forms or plurals
    if term_lower + 's' in text_lower or term_lower + "'s" in text_lower:
        return True
    
    return False

def analyze_term_matching(input_file: str) -> dict:
    """
    Analyze how many terms actually match their original texts
    
    Returns:
        Dictionary with analysis results
    """
    print(f"📊 Analyzing term matching in: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} rows")
        
        if 'Terms' not in df.columns or 'Original Text' not in df.columns:
            print(f"❌ Required columns not found. Available columns: {list(df.columns)}")
            return {}
        
        print("🔍 Analyzing term-text matching...")
        
        matching_stats = {
            'total_rows': len(df),
            'exact_matches': 0,
            'variation_matches': 0,
            'no_matches': 0,
            'empty_terms': 0,
            'empty_texts': 0,
            'invalid_rows': []
        }
        
        # Analyze each row
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(df):,} rows...")
            
            term = row.get('Terms', '')
            text = row.get('Original Text', '')
            
            # Check for empty values
            if pd.isna(term) or str(term).strip() == '':
                matching_stats['empty_terms'] += 1
                matching_stats['invalid_rows'].append(idx)
                continue
                
            if pd.isna(text) or str(text).strip() == '':
                matching_stats['empty_texts'] += 1
                matching_stats['invalid_rows'].append(idx)
                continue
            
            # Check for exact match (case insensitive)
            if check_term_in_text(term, text, case_sensitive=False):
                matching_stats['exact_matches'] += 1
            # Check for variations
            elif check_term_variations(term, text):
                matching_stats['variation_matches'] += 1
            else:
                matching_stats['no_matches'] += 1
                matching_stats['invalid_rows'].append(idx)
        
        # Calculate percentages
        total_valid = matching_stats['total_rows'] - matching_stats['empty_terms'] - matching_stats['empty_texts']
        if total_valid > 0:
            matching_stats['exact_match_pct'] = (matching_stats['exact_matches'] / total_valid) * 100
            matching_stats['variation_match_pct'] = (matching_stats['variation_matches'] / total_valid) * 100
            matching_stats['no_match_pct'] = (matching_stats['no_matches'] / total_valid) * 100
        
        return matching_stats
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return {}

def create_cleaned_csv(input_file: str, output_file: str) -> bool:
    """
    Create a cleaned CSV file with only rows where terms match their texts
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🧹 Creating cleaned CSV file: {output_file}")
    
    try:
        # Read the original file
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        print(f"📊 Original file: {original_count:,} rows")
        
        # Filter valid rows
        valid_rows = []
        removed_count = 0
        
        print("🔍 Filtering valid rows...")
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processed {idx:,}/{len(df):,} rows...")
            
            term = row.get('Terms', '')
            text = row.get('Original Text', '')
            
            # Skip empty rows
            if pd.isna(term) or str(term).strip() == '':
                removed_count += 1
                continue
                
            if pd.isna(text) or str(text).strip() == '':
                removed_count += 1
                continue
            
            # Check if term appears in text (exact or variations)
            if check_term_in_text(term, text) or check_term_variations(term, text):
                valid_rows.append(row)
            else:
                removed_count += 1
                # Optionally log some examples of removed terms
                if removed_count <= 10:  # Log first 10 for debugging
                    print(f"   Removed: '{term}' not found in '{str(text)[:100]}...'")
        
        # Create new DataFrame from valid rows
        if valid_rows:
            cleaned_df = pd.DataFrame(valid_rows)
        else:
            print("⚠️ No valid rows found!")
            return False
        
        # Save cleaned file
        cleaned_df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        
        # Statistics
        cleaned_count = len(cleaned_df)
        removal_pct = (removed_count / original_count) * 100
        
        print(f"\n✅ Cleaned CSV created successfully!")
        print(f"📊 Results:")
        print(f"   Original rows: {original_count:,}")
        print(f"   Valid rows: {cleaned_count:,}")
        print(f"   Removed rows: {removed_count:,} ({removal_pct:.1f}%)")
        print(f"   File saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating cleaned file: {e}")
        return False

def display_sample_removals(input_file: str, num_samples: int = 20):
    """Display sample terms that would be removed for manual review"""
    print(f"\n🔍 Sample terms that will be removed (first {num_samples}):")
    print("-" * 80)
    
    try:
        df = pd.read_csv(input_file)
        removal_count = 0
        
        for idx, row in df.iterrows():
            if removal_count >= num_samples:
                break
                
            term = row.get('Terms', '')
            text = row.get('Original Text', '')
            
            # Skip empty rows
            if pd.isna(term) or str(term).strip() == '' or pd.isna(text) or str(text).strip() == '':
                continue
            
            # Check if this would be removed
            if not (check_term_in_text(term, text) or check_term_variations(term, text)):
                removal_count += 1
                print(f"{removal_count:2}. Term: '{term}'")
                print(f"    Text: '{str(text)[:150]}...'")
                print()
        
    except Exception as e:
        print(f"❌ Error getting samples: {e}")

def main():
    """Main function"""
    print("🧹 CSV TERM VERIFICATION AND CLEANUP")
    print("=" * 60)
    
    input_file = "Combined_Terms_Data.csv"
    output_file = "Cleaned_Terms_Data.csv"
    
    # Check if input file exists
    if not pd.io.common.file_exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Analyze term matching first
    analysis = analyze_term_matching(input_file)
    
    if analysis:
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"Total rows: {analysis['total_rows']:,}")
        print(f"Exact matches: {analysis['exact_matches']:,} ({analysis.get('exact_match_pct', 0):.1f}%)")
        print(f"Variation matches: {analysis['variation_matches']:,} ({analysis.get('variation_match_pct', 0):.1f}%)")
        print(f"No matches: {analysis['no_matches']:,} ({analysis.get('no_match_pct', 0):.1f}%)")
        print(f"Empty terms: {analysis['empty_terms']:,}")
        print(f"Empty texts: {analysis['empty_texts']:,}")
        
        total_valid = analysis['exact_matches'] + analysis['variation_matches']
        total_invalid = analysis['no_matches'] + analysis['empty_terms'] + analysis['empty_texts']
        
        print(f"\n📈 SUMMARY:")
        print(f"Valid rows to keep: {total_valid:,}")
        print(f"Invalid rows to remove: {total_invalid:,}")
        
        if total_invalid > 0:
            # Show sample removals
            display_sample_removals(input_file)
            
            # Confirm before proceeding
            response = input(f"\nProceed with creating cleaned file? (y/n): ").strip().lower()
            
            if response == 'y':
                # Create cleaned file
                success = create_cleaned_csv(input_file, output_file)
                
                if success:
                    print(f"\n🎉 Process completed successfully!")
                    print(f"✅ Original file: {input_file}")
                    print(f"✅ Cleaned file: {output_file}")
                    print(f"💡 You now have a verified dataset with terms that actually appear in their texts")
                else:
                    print(f"\n❌ Failed to create cleaned file")
            else:
                print("Operation cancelled by user")
        else:
            print(f"\n✅ All terms match their texts! No cleaning needed.")

if __name__ == "__main__":
    main()

