#!/usr/bin/env python3
"""
Direct Unified Processor - Embedded version to avoid subprocess issues
Implements the core functionality of unified_term_processor.py directly
"""

import pandas as pd
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def clean_term(term):
    """Clean and validate a term"""
    if not term or pd.isna(term):
        return None
    
    # Handle dictionary objects
    if isinstance(term, dict):
        if 'term' in term:
            term = term['term']
        else:
            return None
    
    # Convert to string and clean
    term = str(term).strip()
    
    # Skip empty or very short terms
    if len(term) < 2:
        return None
    
    # Skip terms that are just numbers
    if term.isdigit():
        return None
    
    # Basic cleaning
    term = re.sub(r'[^\w\s\-\.]', '', term)
    term = re.sub(r'\s+', ' ', term)
    term = term.strip()
    
    return term if term else None

def extract_terms_from_field(field_value):
    """Extract terms from a field value (handles JSON arrays, pipe-separated, etc.)"""
    if not field_value or pd.isna(field_value):
        return []
    
    terms = []
    
    try:
        # Try to parse as JSON first
        if isinstance(field_value, str) and (field_value.startswith('[') or field_value.startswith('{')):
            json_data = json.loads(field_value)
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict) and 'term' in item:
                        clean_t = clean_term(item['term'])
                        if clean_t:
                            terms.append(clean_t)
                    elif isinstance(item, str):
                        clean_t = clean_term(item)
                        if clean_t:
                            terms.append(clean_t)
            elif isinstance(json_data, dict) and 'term' in json_data:
                clean_t = clean_term(json_data['term'])
                if clean_t:
                    terms.append(clean_t)
    except (json.JSONDecodeError, ValueError):
        # Not JSON, try other formats
        pass
    
    # If no terms found yet, try pipe-separated or comma-separated
    if not terms:
        field_str = str(field_value)
        # Try pipe-separated first
        if '|' in field_str:
            parts = field_str.split('|')
        elif ',' in field_str:
            parts = field_str.split(',')
        else:
            parts = [field_str]
        
        for part in parts:
            clean_t = clean_term(part)
            if clean_t:
                terms.append(clean_t)
    
    return terms

def process_terms_directly(input_file: str, output_prefix: str) -> Tuple[str, str]:
    """
    Process terms directly without subprocess call
    Returns: (combined_file, cleaned_file)
    """
    
    print(f"[DIRECT] Processing {input_file} with embedded unified processor...")
    
    # Read the input CSV
    try:
        df = pd.read_csv(input_file)
        print(f"[OK] Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read input file: {e}")
    
    # Extract terms from Final_Curated_Terms and Final_Curated_Terms_Detailed
    all_terms_data = []
    
    target_columns = ['Final_Curated_Terms', 'Final_Curated_Terms_Detailed']
    available_columns = [col for col in target_columns if col in df.columns]
    
    if not available_columns:
        raise RuntimeError(f"Required columns not found: {target_columns}")
    
    print(f"[COLUMNS] Processing columns: {available_columns}")
    
    for idx, row in df.iterrows():
        row_terms = set()  # Use set to avoid duplicates within same row
        
        # Extract from available columns
        for col in available_columns:
            field_value = row[col]
            terms = extract_terms_from_field(field_value)
            row_terms.update(terms)
        
        # Add each unique term from this row
        for term in row_terms:
            # Get the original text for verification
            original_text = ""
            for text_col in ['Original_Text', 'Text', 'Content']:
                if text_col in df.columns and not pd.isna(row[text_col]):
                    original_text = str(row[text_col])[:200]  # Truncate for storage
                    break
            
            all_terms_data.append({
                'term': term,
                'frequency': 1,  # Will be aggregated later
                'original_text': original_text,
                'source_row': idx,
                'pos_tag': 'UNKNOWN'  # Simplified for direct processing
            })
    
    print(f"[EXTRACTED] Found {len(all_terms_data)} term instances")
    
    # Create combined data
    combined_df = pd.DataFrame(all_terms_data)
    
    # Aggregate by term
    term_aggregation = {}
    for item in all_terms_data:
        term = item['term']
        if term in term_aggregation:
            term_aggregation[term]['frequency'] += 1
            # Keep the longest original text
            if len(item['original_text']) > len(term_aggregation[term]['original_text']):
                term_aggregation[term]['original_text'] = item['original_text']
        else:
            term_aggregation[term] = item.copy()
    
    # Create cleaned data
    cleaned_data = list(term_aggregation.values())
    cleaned_df = pd.DataFrame(cleaned_data)
    
    print(f"[AGGREGATED] {len(cleaned_data)} unique terms")
    
    # Save files
    combined_file = f"{output_prefix}_Combined.csv"
    cleaned_file = f"{output_prefix}_Cleaned.csv"
    
    combined_df.to_csv(combined_file, index=False)
    cleaned_df.to_csv(cleaned_file, index=False)
    
    print(f"[SAVED] Combined data: {combined_file}")
    print(f"[SAVED] Cleaned data: {cleaned_file}")
    
    # Create simple formats
    simple_df = cleaned_df[['term', 'frequency']].copy()
    simple_file = f"{output_prefix}_Simple.csv"
    simple_df.to_csv(simple_file, index=False)
    
    summary_df = cleaned_df.head(100)  # Top 100 terms
    summary_file = f"{output_prefix}_Summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"[FORMATS] Created simple and summary formats")
    print(f"[SUCCESS] Direct processing completed!")
    
    return combined_file, cleaned_file

def integrate_direct_processor():
    """Create a patched version of the main system that uses direct processing"""
    
    patch_code = '''
def step_1_initial_term_collection_direct(self, input_file: str) -> str:
    """
    Step 1: Direct Initial Term Collection (bypasses subprocess issues)
    """
    logger.info("[LIST] STEP 1: Initial Term Collection and Verification (DIRECT)")
    logger.info("=" * 60)
    logger.info("[DIRECT] Using embedded unified processor - bypassing subprocess issues")
    
    try:
        from direct_unified_processor import process_terms_directly
        
        output_prefix = str(self.output_dir / "Step1_DirectProcessor")
        
        logger.info(f"[DIRECT] Processing {input_file} with embedded processor...")
        logger.info("[TARGET] Final_Curated_Terms and Final_Curated_Terms_Detailed columns")
        
        # Process directly
        combined_file, cleaned_file = process_terms_directly(input_file, output_prefix)
        
        # Add metadata
        self._add_step_metadata(cleaned_file, 1, "Direct Initial Term Collection")
        
        logger.info("[OK] Step 1 completed: Direct term collection successful")
        logger.info(f"[FOLDER] Cleaned data: {cleaned_file}")
        
        return cleaned_file
        
    except Exception as e:
        logger.error(f"[ERROR] Direct processing failed: {e}")
        raise RuntimeError(f"Direct unified processor failed: {e}")
'''
    
    print("[PATCH] Direct processor integration code prepared")
    return patch_code

if __name__ == "__main__":
    # Test the direct processor
    test_input = "Term_Extracted_result_sample1000.csv"
    test_output = "test_direct_output"
    
    if os.path.exists(test_input):
        print("🧪 TESTING DIRECT UNIFIED PROCESSOR")
        print("=" * 50)
        
        try:
            combined_file, cleaned_file = process_terms_directly(test_input, test_output)
            print(f"✅ Direct processing successful!")
            print(f"📁 Combined: {combined_file}")
            print(f"📁 Cleaned: {cleaned_file}")
            
            # Show some stats
            cleaned_df = pd.read_csv(cleaned_file)
            print(f"📊 Processed {len(cleaned_df)} unique terms")
            print(f"📈 Top terms by frequency:")
            top_terms = cleaned_df.nlargest(5, 'frequency')
            for _, row in top_terms.iterrows():
                print(f"   - {row['term']}: {row['frequency']}")
                
        except Exception as e:
            print(f"❌ Direct processing failed: {e}")
    else:
        print(f"⚠️ Test file not found: {test_input}")

