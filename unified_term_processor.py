#!/usr/bin/env python3
"""
Unified Term Processing System
Single comprehensive script for processing terminology CSV files
Focuses on Final_Curated_Terms and Final_Curated_Terms_Detailed columns
"""

import pandas as pd
import json
import re
import csv
import os
import nltk
import argparse
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("[OK] NLTK data ready")
except Exception as e:
    print(f"[WARNING] NLTK download issue: {e}")

class UnifiedTermProcessor:
    """Unified term processing system with configurable options"""
    
    def __init__(self):
        self.stats = {}
    
    # ========== CONFIGURATION FUNCTIONS ==========
    
    def detect_file_format(self, filename: str) -> str:
        """Detect the format of the input CSV file"""
        try:
            df = pd.read_csv(filename, nrows=5)  # Read only first 5 rows for detection
            columns = df.columns.tolist()
            
            print(f"[INFO] Detected columns: {len(columns)} columns")
            print(f"[COLUMNS] Key columns found: {[col for col in columns if any(keyword in col.lower() for keyword in ['final', 'curated', 'terms', 'original'])]}")
            
            # Check if it's a terminology extraction result file
            if 'Original_Text' in columns and ('Final_Curated_Terms' in columns or 'Final_Curated_Terms_Detailed' in columns):
                return 'extraction_result'
            elif 'Terms' in columns and 'pos_tags' in columns and 'Original Text' in columns:
                return 'combined_format'
            elif 'source' in columns:
                return 'simple_input'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"[ERROR] Error detecting file format: {e}")
            return 'unknown'
    
    def get_user_config(self):
        """Get configuration from user input"""
        print("[CONFIG] CONFIGURATION SETUP")
        print("=" * 50)
        
        # Show available files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if not csv_files:
            print("[ERROR] No CSV files found in current directory")
            return None
        
        print("[FILES] Available CSV files:")
        for i, filename in enumerate(csv_files, 1):
            # Get file size
            try:
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                size_str = f"({size_mb:.1f} MB)" if size_mb >= 1 else f"({os.path.getsize(filename)} bytes)"
            except:
                size_str = ""
            print(f"   {i:2d}. {filename} {size_str}")
        
        # Get file choice
        print()
        while True:
            try:
                choice = input("Select file number (or press Enter to type filename): ").strip()
                
                if not choice:
                    # User wants to type filename
                    filename = input("Enter CSV filename: ").strip()
                    if filename and os.path.exists(filename):
                        break
                    elif filename:
                        print(f"[ERROR] File not found: {filename}")
                        continue
                    else:
                        print("[ERROR] Please enter a filename")
                        continue
                else:
                    # User selected by number
                    file_num = int(choice)
                    if 1 <= file_num <= len(csv_files):
                        filename = csv_files[file_num - 1]
                        break
                    else:
                        print(f"[ERROR] Please enter a number between 1 and {len(csv_files)}")
                        continue
            except ValueError:
                print("[ERROR] Please enter a valid number")
                continue
        
        # Detect file format
        file_format = self.detect_file_format(filename)
        print(f"[FORMAT] Detected format: {file_format}")
        
        # Get output prefix
        print()
        default_prefix = os.path.splitext(filename)[0] + "_Processed"
        output_prefix = input(f"Output prefix (default: {default_prefix}): ").strip()
        if not output_prefix:
            output_prefix = default_prefix
        
        # Get processing options
        print()
        print("[OPTIONS] Processing Options:")
        
        verify_terms = input("Verify terms appear in texts? (Y/n): ").strip().lower()
        verify_terms = verify_terms != 'n'
        
        create_json = input("Create JSON format? (Y/n): ").strip().lower()
        create_json = create_json != 'n'
        
        create_clean_csv = input("Create clean CSV formats? (Y/n): ").strip().lower()
        create_clean_csv = create_clean_csv != 'n'
        
        config = {
            'input_file': filename,
            'file_format': file_format,
            'output_prefix': output_prefix,
            'verify_terms': verify_terms,
            'create_json': create_json,
            'create_clean_csv': create_clean_csv
        }
        
        print(f"\n[CONFIG] Configuration set:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        return config
    
    # ========== ENHANCED CLEANING FUNCTIONS ==========
    
    def clean_term(self, term):
        """Clean and normalize a term - handles various data types"""
        # Handle different data types
        if pd.isna(term):
            return None
        
        # If it's a dict or other complex object, skip it
        if isinstance(term, (dict, list)):
            return None
        
        # Convert to string and check if empty
        term_str = str(term).strip()
        if not term_str or term_str.lower() in ['nan', 'none', 'null']:
            return None
        
        # Remove extra whitespace
        term_str = re.sub(r'\s+', ' ', term_str)
        
        # Remove quotes if they wrap the entire term
        if term_str.startswith('"') and term_str.endswith('"'):
            term_str = term_str[1:-1]
        if term_str.startswith("'") and term_str.endswith("'"):
            term_str = term_str[1:-1]
        
        # Skip very short or very long terms
        if len(term_str) < 2 or len(term_str) > 100:
            return None
        
        # Skip terms that are mostly numbers or special characters
        if re.match(r'^[0-9\.\-\+\s]+$', term_str):
            return None
        
        return term_str.lower()

    def extract_final_curated_terms(self, field_value):
        """Extract terms specifically from Final_Curated_Terms or Final_Curated_Terms_Detailed"""
        if pd.isna(field_value) or not str(field_value).strip():
            return []
        
        field_str = str(field_value).strip()
        if not field_str or field_str.lower() in ['nan', 'none', 'null']:
            return []
        
        terms = []
        
        print(f"[PROCESS] Processing field value: {field_str[:200]}..." if len(field_str) > 200 else f"[PROCESS] Processing field value: {field_str}")
        
        # Try to parse as JSON array first (for Final_Curated_Terms_Detailed)
        try:
            if field_str.startswith('[') and field_str.endswith(']'):
                # Handle complex JSON with nested objects (Final_Curated_Terms_Detailed)
                if '"term"' in field_str and '"confidence"' in field_str:
                    print("   [JSON] Detected detailed JSON format with confidence scores")
                    # This is a detailed JSON format - extract just the terms
                    parsed = json.loads(field_str)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and 'term' in item:
                                clean_term = self.clean_term(item['term'])
                                if clean_term:
                                    terms.append(clean_term)
                                    print(f"   [TERM] Extracted term: {clean_term}")
                else:
                    print("   [JSON] Detected simple JSON array format")
                    # Simple JSON array
                    parsed = json.loads(field_str)
                    if isinstance(parsed, list):
                        for t in parsed:
                            clean_term = self.clean_term(t)
                            if clean_term:
                                terms.append(clean_term)
                                print(f"   [TERM] Extracted term: {clean_term}")
            else:
                print("   [TEXT] Detected pipe-separated or simple text format")
                # Split by common separators (Final_Curated_Terms format)
                separators = [' | ', '|', ',', ';', '\n']
                current_terms = [field_str]
                
                for sep in separators:
                    new_terms = []
                    for term in current_terms:
                        new_terms.extend(term.split(sep))
                    current_terms = new_terms
                
                for t in current_terms:
                    clean_term = self.clean_term(t)
                    if clean_term:
                        terms.append(clean_term)
                        print(f"   [TERM] Extracted term: {clean_term}")
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   [WARNING] JSON parsing failed: {e}")
            # Fallback: treat as pipe-separated or single term
            if ' | ' in field_str:
                for t in field_str.split(' | '):
                    clean_term = self.clean_term(t)
                    if clean_term:
                        terms.append(clean_term)
                        print(f"   [TERM] Extracted term (fallback): {clean_term}")
            elif '|' in field_str:
                for t in field_str.split('|'):
                    clean_term = self.clean_term(t)
                    if clean_term:
                        terms.append(clean_term)
                        print(f"   [TERM] Extracted term (fallback): {clean_term}")
            else:
                clean_term = self.clean_term(field_str)
                if clean_term:
                    terms.append(clean_term)
                    print(f"   [TERM] Extracted term (fallback): {clean_term}")
        
        # Filter out duplicates
        unique_terms = list(set(terms))
        print(f"   [STATS] Total unique terms extracted: {len(unique_terms)}")
        return unique_terms

    def get_pos_tag(self, term):
        """Get POS tag for a term using NLTK"""
        try:
            tokens = word_tokenize(term)
            pos_tags = pos_tag(tokens)
            if pos_tags:
                return pos_tags[0][1]
            else:
                return 'UNKNOWN'
        except Exception:
            return 'UNKNOWN'

    # ========== FORMAT-SPECIFIC PROCESSING ==========
    
    def process_extraction_result_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process files from terminology extraction system - FOCUS ON FINAL CURATED TERMS ONLY"""
        print("[PROCESS] Processing extraction result format...")
        print("[FOCUS] FOCUSING ON FINAL CURATED TERMS ONLY")
        
        combined_data = []
        processed_terms = set()
        
        # Determine which column to use for final curated terms
        final_terms_column = None
        if 'Final_Curated_Terms_Detailed' in df.columns:
            final_terms_column = 'Final_Curated_Terms_Detailed'
            print(f"   [COLUMN] Using column: {final_terms_column} (with confidence scores)")
        elif 'Final_Curated_Terms' in df.columns:
            final_terms_column = 'Final_Curated_Terms'
            print(f"   [COLUMN] Using column: {final_terms_column} (simple format)")
        else:
            print("   [ERROR] No Final_Curated_Terms or Final_Curated_Terms_Detailed column found!")
            return pd.DataFrame()
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   [PROGRESS] Processed {idx:,}/{len(df):,} rows...")
            
            original_text = str(row.get('Original_Text', '')).strip()
            if not original_text or original_text == 'nan':
                continue
            
            print(f"\n[ROW] Processing row {idx + 1}: {original_text[:100]}...")
            
            # Extract terms ONLY from the final curated column
            final_terms = self.extract_final_curated_terms(row[final_terms_column])
            
            # Process each unique term
            for term in final_terms:
                if term:
                    unique_key = f"{term}||{original_text}"
                    
                    if unique_key not in processed_terms:
                        processed_terms.add(unique_key)
                        
                        pos_tag_result = self.get_pos_tag(term)
                        
                        combined_data.append({
                            'Terms': term,
                            'pos_tags': pos_tag_result,
                            'Original Text': original_text
                        })
                        print(f"   [ADD] Added: {term} ({pos_tag_result})")
        
        print(f"\n[RESULT] Generated {len(combined_data):,} term entries from {len(df):,} source rows")
        print(f"[COLUMN] Used column: {final_terms_column}")
        return pd.DataFrame(combined_data)
    
    def process_combined_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process files already in combined format"""
        print("[PROCESS] Processing combined format...")
        print(f"[OK] File already in combined format: {len(df):,} rows")
        return df
    
    def process_simple_input_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process simple input format (like sample_input.csv)"""
        print("[PROCESS] Processing simple input format...")
        
        combined_data = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"   [PROGRESS] Processed {idx:,}/{len(df):,} rows...")
            
            source_text = str(row.get('source', '')).strip()
            if not source_text:
                continue
            
            # If there are ground truth terms, use them
            gt_terms = []
            if 'terminology_found' in row:
                gt_terms = self.extract_final_curated_terms(row['terminology_found'])
            
            # For simple format, we just use the ground truth terms
            for term in gt_terms:
                if term:
                    pos_tag_result = self.get_pos_tag(term)
                    
                    combined_data.append({
                        'Terms': term,
                        'pos_tags': pos_tag_result,
                        'Original Text': source_text
                    })
        
        print(f"[OK] Generated {len(combined_data):,} term entries")
        return pd.DataFrame(combined_data)
    
    # ========== VERIFICATION FUNCTIONS ==========
    
    def check_term_in_text(self, term: str, text: str, case_sensitive: bool = False) -> bool:
        """Check if a term appears in the text"""
        if not term or not text:
            return False
        
        term = str(term).strip()
        text = str(text).strip()
        
        if not term or not text:
            return False
        
        if case_sensitive:
            return term in text
        else:
            return term.lower() in text.lower()

    def check_term_variations(self, term: str, text: str) -> bool:
        """Check if a term or its variations appear in text"""
        if not term or not text:
            return False
        
        term = str(term).strip()
        text = str(text).strip()
        
        if not term or not text:
            return False
        
        term_lower = term.lower()
        text_lower = text.lower()
        
        # Check exact match
        if term_lower in text_lower:
            return True
        
        # Check word boundaries
        word_pattern = r'\b' + re.escape(term_lower) + r'\b'
        if re.search(word_pattern, text_lower):
            return True
        
        # Check for common variations
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

    def verify_and_clean_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify terms match their texts and return cleaned DataFrame"""
        
        print("[CLEAN] VERIFYING TERMS APPEAR IN TEXTS")
        print("=" * 50)
        
        original_count = len(df)
        print(f"[STATS] Original entries: {original_count:,}")
        
        valid_rows = []
        removed_count = 0
        
        print("[SEARCH] Filtering valid rows...")
        
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
            
            # Check if term appears in text
            if self.check_term_in_text(term, text) or self.check_term_variations(term, text):
                valid_rows.append(row)
            else:
                removed_count += 1
                print(f"   [ERROR] Removed: '{term}' not found in '{text[:50]}...'")
        
        if valid_rows:
            cleaned_df = pd.DataFrame(valid_rows)
        else:
            print("[WARNING] No valid rows found!")
            return pd.DataFrame()
        
        cleaned_count = len(cleaned_df)
        removal_pct = (removed_count / original_count) * 100 if original_count > 0 else 0
        
        print(f"[OK] Verification completed!")
        print(f"[STATS] Results:")
        print(f"   Original rows: {original_count:,}")
        print(f"   Valid rows: {cleaned_count:,}")
        print(f"   Removed rows: {removed_count:,} ({removal_pct:.1f}%)")
        
        return cleaned_df
    
    # ========== OUTPUT GENERATION ==========
    
    def create_clean_formats(self, df: pd.DataFrame, prefix: str):
        """Create clean output formats"""
        print(f"[STATS] CREATING CLEAN FORMATS")
        print("=" * 50)
        
        # Aggregate data by unique terms
        term_data = defaultdict(lambda: {
            'pos_tags': set(),
            'original_texts': set(),
            'frequency': 0
        })
        
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
        
        # Create simple format
        output_data = []
        for term, data in term_data.items():
            pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
            original_texts_list = list(data['original_texts'])
            
            pos_tags_str = " | ".join(pos_tags_list)
            
            texts_count = len(original_texts_list)
            if texts_count <= 3:
                texts_str = " | ".join(original_texts_list)
            else:
                sample_texts = original_texts_list[:3]
                texts_str = " | ".join(sample_texts) + f" | [... and {texts_count - 3} more texts]"
            
            output_data.append({
                'Terms': term,
                'POS_Tags': pos_tags_str,
                'Sample_Texts': texts_str,
                'Text_Count': texts_count,
                'Frequency': data['frequency']
            })
        
        # Sort by frequency
        output_df = pd.DataFrame(output_data)
        output_df = output_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
        
        # Save files
        simple_file = f"{prefix}_Simple.csv"
        summary_file = f"{prefix}_Summary.csv"
        
        output_df.to_csv(simple_file, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        
        # Create summary
        summary_df = output_df[['Terms', 'Text_Count', 'Frequency']].copy()
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print(f"[OK] Created {simple_file} with {len(output_df)} terms")
        print(f"[OK] Created {summary_file} with {len(summary_df)} terms")
        
        return term_data
    
    def create_json_format(self, term_data: dict, prefix: str):
        """Create JSON format"""
        print(f"[FILE] CREATING JSON FORMAT")
        print("=" * 50)
        
        json_file = f"{prefix}_Complete.json"
        
        # Create JSON structure
        complete_data = {
            "metadata": {
                "total_unique_terms": len(term_data),
                "description": "Complete processed terms data extracted from Final_Curated_Terms/Final_Curated_Terms_Detailed columns only",
                "format_version": "2.0",
                "extraction_source": "Final_Curated_Terms and Final_Curated_Terms_Detailed columns",
                "created": datetime.now().isoformat()
            },
            "terms": []
        }
        
        total_pos_variations = 0
        total_texts = 0
        
        for term, data in term_data.items():
            pos_tags_list = sorted(list(data['pos_tags'])) if data['pos_tags'] else []
            original_texts_list = list(data['original_texts'])
            
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
                    "texts": original_texts_list
                }
            }
            
            complete_data["terms"].append(term_entry)
        
        # Sort terms by frequency
        complete_data["terms"].sort(key=lambda x: x["frequency"], reverse=True)
        
        # Update metadata
        complete_data["metadata"]["total_pos_variations"] = total_pos_variations
        complete_data["metadata"]["total_original_texts"] = total_texts
        if complete_data["terms"]:
            complete_data["metadata"]["most_frequent_term"] = complete_data["terms"][0]["term"]
            complete_data["metadata"]["highest_frequency"] = complete_data["terms"][0]["frequency"]
        
        # Save JSON file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(json_file) / (1024 * 1024)
        print(f"[OK] Created {json_file} ({file_size_mb:.1f} MB)")
        print(f"   - {len(complete_data['terms'])} unique terms")
        print(f"   - {total_pos_variations:,} POS tag variations")
        print(f"   - {total_texts:,} original texts")
    
    # ========== MAIN PROCESSING ==========
    
    def process_file(self, config: dict):
        """Process file according to configuration"""
        
        print(f"\n[START] PROCESSING {config['input_file']}")
        print("=" * 80)
        
        # Load input file
        try:
            df = pd.read_csv(config['input_file'])
            print(f"[OK] Loaded {len(df):,} rows")
        except Exception as e:
            print(f"[ERROR] Error loading file: {e}")
            return False
        
        # Process according to format
        if config['file_format'] == 'extraction_result':
            combined_df = self.process_extraction_result_format(df)
        elif config['file_format'] == 'combined_format':
            combined_df = self.process_combined_format(df)
        elif config['file_format'] == 'simple_input':
            combined_df = self.process_simple_input_format(df)
        else:
            print(f"[ERROR] Unknown file format: {config['file_format']}")
            return False
        
        if combined_df.empty:
            print("[ERROR] No data to process")
            return False
        
        # Save combined format
        combined_file = f"{config['output_prefix']}_Combined.csv"
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"[OK] Saved combined format: {combined_file}")
        
        # Verify terms if requested
        if config['verify_terms']:
            cleaned_df = self.verify_and_clean_csv(combined_df)
            if not cleaned_df.empty:
                cleaned_file = f"{config['output_prefix']}_Cleaned.csv"
                cleaned_df.to_csv(cleaned_file, index=False, encoding='utf-8')
                print(f"[OK] Saved cleaned format: {cleaned_file}")
                final_df = cleaned_df
            else:
                print("[WARNING] Using original data (no valid cleaned data)")
                final_df = combined_df
        else:
            final_df = combined_df
        
        # Create output formats
        if config['create_clean_csv'] or config['create_json']:
            term_data = None
            
            if config['create_clean_csv']:
                term_data = self.create_clean_formats(final_df, config['output_prefix'])
            
            if config['create_json']:
                if term_data is None:
                    # Need to create term_data for JSON
                    term_data = defaultdict(lambda: {
                        'pos_tags': set(),
                        'original_texts': set(),
                        'frequency': 0
                    })
                    
                    for _, row in final_df.iterrows():
                        term = str(row['Terms']) if pd.notna(row['Terms']) else 'UNKNOWN'
                        pos_tag = str(row['pos_tags']) if pd.notna(row['pos_tags']) else ''
                        original_text = str(row['Original Text']) if pd.notna(row['Original Text']) else ''
                        
                        if pos_tag.strip():
                            term_data[term]['pos_tags'].add(pos_tag)
                        if original_text.strip():
                            term_data[term]['original_texts'].add(original_text)
                        term_data[term]['frequency'] += 1
                
                self.create_json_format(term_data, config['output_prefix'])
        
        print(f"\n[OK] PROCESSING COMPLETE!")
        print(f"[FOLDER] Output files created with prefix: {config['output_prefix']}")
        
        return True
    
    def show_results(self, success: bool, config: dict):
        """Show processing results"""
        print()
        if success:
            print("[SUCCESS] PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"[FOLDER] Input file: {config['input_file']}")
            print(f"[OUTPUT] Output prefix: {config['output_prefix']}")
            print()
            print("[FILES] Generated files:")
            
            # List generated files
            possible_files = [
                f"{config['output_prefix']}_Combined.csv",
                f"{config['output_prefix']}_Cleaned.csv",
                f"{config['output_prefix']}_Simple.csv",
                f"{config['output_prefix']}_Summary.csv",
                f"{config['output_prefix']}_Complete.json"
            ]
            
            for filename in possible_files:
                if os.path.exists(filename):
                    try:
                        size_mb = os.path.getsize(filename) / (1024 * 1024)
                        size_str = f"({size_mb:.1f} MB)" if size_mb >= 1 else f"({os.path.getsize(filename)} bytes)"
                    except:
                        size_str = ""
                    print(f"   [OK] {filename} {size_str}")
            
            print()
            print("[TIPS] Tips:")
            print("   - Use *_Simple.csv for easy viewing in Excel")
            print("   - Use *_Complete.json for programmatic access")
            print("   - *_Cleaned.csv contains only verified terms")
            print("   - Terms extracted from Final_Curated_Terms/Final_Curated_Terms_Detailed only")
            
        else:
            print("[ERROR] PROCESSING FAILED!")
            print("Please check the error messages above and try again.")

def main():
    """Main function with command line and interactive support"""
    parser = argparse.ArgumentParser(description='Unified Term Processing System')
    parser.add_argument('--input', '-i', help='Input CSV file')
    parser.add_argument('--output-prefix', '-o', default='Processed', help='Output filename prefix')
    parser.add_argument('--no-verify', action='store_true', help='Skip term verification')
    parser.add_argument('--no-json', action='store_true', help='Skip JSON output')
    parser.add_argument('--no-clean-csv', action='store_true', help='Skip clean CSV output')
    parser.add_argument('--interactive', '-I', action='store_true', help='Interactive configuration mode')
    
    args = parser.parse_args()
    
    processor = UnifiedTermProcessor()
    
    print("[TARGET] UNIFIED TERM PROCESSING SYSTEM")
    print("=" * 80)
    print("[FOCUS] FOCUSES ON FINAL_CURATED_TERMS ONLY - EXTRACTS CORRECT TERMS!")
    print()
    
    if args.interactive or not args.input:
        # Interactive mode
        config = processor.get_user_config()
        if not config:
            print("[ERROR] No configuration provided")
            return 1
    else:
        # Command line mode
        if not os.path.exists(args.input):
            print(f"[ERROR] Input file not found: {args.input}")
            return 1
        
        file_format = processor.detect_file_format(args.input)
        
        config = {
            'input_file': args.input,
            'file_format': file_format,
            'output_prefix': args.output_prefix,
            'create_json': not args.no_json,
            'create_clean_csv': not args.no_clean_csv,
            'verify_terms': not args.no_verify
        }
        
        print(f"[FOLDER] Processing: {args.input}")
        print(f"[SEARCH] Format: {file_format}")
    
    # Confirm processing in interactive mode
    if args.interactive or not args.input:
        print(f"\n[FILES] Processing Summary:")
        print(f"   Input: {config['input_file']}")
        print(f"   Format: {config['file_format']}")
        print(f"   Output prefix: {config['output_prefix']}")
        print(f"   Verify terms: {'Yes' if config['verify_terms'] else 'No'}")
        print(f"   Create JSON: {'Yes' if config['create_json'] else 'No'}")
        print(f"   Create clean CSV: {'Yes' if config['create_clean_csv'] else 'No'}")
        
        print()
        confirm = input("Proceed with processing? (Y/n): ").strip().lower()
        if confirm == 'n':
            print("[ERROR] Processing cancelled by user")
            return 1
    
    # Process the file
    print("\n[START] Starting processing...")
    success = processor.process_file(config)
    
    # Show results
    processor.show_results(success, config)
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n[WARNING] Processing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        exit(1)
