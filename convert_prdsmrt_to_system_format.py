#!/usr/bin/env python3
"""
PRDSMRT CSV to Multi-Agent Terminology Validation System Converter
================================================================

Converts PRDSMRT_doc_merged_results_Processed_Simple.csv format to 
High_Frequency_Terms.json format for direct Step 4 integration.

Input Format: Terms,POS_Tags,Sample_Texts,Text_Count,Frequency
Output Format: High_Frequency_Terms.json (Step 4 compatible)
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_prdsmrt_to_high_frequency_format(input_csv: str, output_dir: str = None, min_frequency: int = 2):
    """
    Convert PRDSMRT CSV to High_Frequency_Terms.json format
    
    Args:
        input_csv: Path to PRDSMRT_doc_merged_results_Processed_Simple.csv
        output_dir: Output directory (default: current directory)
        min_frequency: Minimum frequency threshold (default: 2)
    """
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    if output_dir is None:
        output_dir = os.path.dirname(input_csv) or "."
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"[CONVERT] Converting PRDSMRT CSV to High_Frequency_Terms.json format")
    logger.info(f"[INPUT] Source: {input_csv}")
    logger.info(f"[OUTPUT] Directory: {output_dir}")
    logger.info(f"[FILTER] Minimum frequency: {min_frequency}")
    
    high_frequency_terms = []
    low_frequency_terms = []
    total_terms = 0
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_terms += 1
                
                # Extract data from PRDSMRT format
                term = row.get('Terms', '').strip()
                pos_tags = row.get('POS_Tags', 'UNKNOWN').strip()
                sample_texts = row.get('Sample_Texts', '').strip()
                text_count = int(row.get('Text_Count', 0))
                frequency = int(row.get('Frequency', 0))
                
                if not term:
                    logger.warning(f"[SKIP] Empty term at row {total_terms}")
                    continue
                
                # Parse sample texts (split by ' | ')
                original_texts = []
                if sample_texts:
                    # Split by ' | ' and clean up
                    texts = sample_texts.split(' | ')
                    for text in texts:
                        text = text.strip()
                        if text and not text.startswith('[...') and not text.endswith('more texts]'):
                            original_texts.append(text)
                
                # Create term data structure compatible with Step 4 output
                term_data = {
                    'term': term,
                    'frequency': frequency,
                    'original_texts': {
                        'texts': original_texts[:10]  # Limit to first 10 examples
                    },
                    'pos_tag_variations': {
                        'tags': [pos_tags] if pos_tags != 'UNKNOWN' else ['NOUN']  # Default to NOUN
                    },
                    'glossary_analysis': f'PRDSMRT extracted term with {frequency} occurrences',
                    'dictionary_analysis': {
                        'in_dictionary': None,  # Will be determined in Step 3 if needed
                        'method': 'prdsmrt_conversion',
                        'confidence': 'high' if frequency >= 10 else 'medium',
                        'reason': f'Extracted from PRDSMRT documentation with {frequency} occurrences'
                    },
                    'source': 'PRDSMRT_doc_merged_results_Processed_Simple.csv',
                    'conversion_metadata': {
                        'original_text_count': text_count,
                        'conversion_timestamp': datetime.now().isoformat(),
                        'pos_tag_original': pos_tags
                    }
                }
                
                # Filter by frequency
                if frequency >= min_frequency:
                    high_frequency_terms.append(term_data)
                else:
                    low_frequency_terms.append(term_data)
                
                if total_terms % 1000 == 0:
                    logger.info(f"[PROGRESS] Processed {total_terms:,} terms...")
        
        logger.info(f"[STATS] Total terms processed: {total_terms:,}")
        logger.info(f"[STATS] High frequency terms (>={min_frequency}): {len(high_frequency_terms):,}")
        logger.info(f"[STATS] Low frequency terms (<{min_frequency}): {len(low_frequency_terms):,}")
        
        # Create High_Frequency_Terms.json (Step 4 compatible)
        high_freq_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_terms': len(high_frequency_terms),
                'min_frequency': min_frequency,
                'source': 'PRDSMRT_doc_merged_results_Processed_Simple.csv',
                'conversion_method': 'prdsmrt_to_high_frequency_converter',
                'original_total_terms': total_terms,
                'filtered_out_terms': len(low_frequency_terms)
            },
            'terms': high_frequency_terms
        }
        
        # Save High_Frequency_Terms.json
        high_freq_file = output_dir / "High_Frequency_Terms.json"
        with open(high_freq_file, 'w', encoding='utf-8') as f:
            json.dump(high_freq_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVED] High_Frequency_Terms.json: {high_freq_file}")
        
        # Also save low frequency terms for completeness
        if low_frequency_terms:
            low_freq_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_terms': len(low_frequency_terms),
                    'frequency_range': f'1 to {min_frequency-1}',
                    'source': 'PRDSMRT_doc_merged_results_Processed_Simple.csv'
                },
                'terms': low_frequency_terms
            }
            
            low_freq_file = output_dir / "Low_Frequency_Terms.json"
            with open(low_freq_file, 'w', encoding='utf-8') as f:
                json.dump(low_freq_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SAVED] Low_Frequency_Terms.json: {low_freq_file}")
        
        # Create metadata file
        metadata = {
            'conversion_info': {
                'source_file': os.path.basename(input_csv),
                'conversion_timestamp': datetime.now().isoformat(),
                'converter_version': '1.0.0',
                'total_input_terms': total_terms,
                'high_frequency_terms': len(high_frequency_terms),
                'low_frequency_terms': len(low_frequency_terms),
                'frequency_threshold': min_frequency
            },
            'system_integration': {
                'compatible_with': 'Multi-Agent Terminology Validation System v1.0',
                'entry_point': 'Step 4 (Frequency Analysis) - Output Compatible',
                'next_step': 'Step 5 (Translation Process)',
                'skip_steps': ['Step 1', 'Step 2', 'Step 3'],
                'usage': f'python agentic_terminology_validation_system.py --start-from-step 5 --high-freq-file {high_freq_file.name}'
            },
            'data_quality': {
                'terms_with_contexts': len([t for t in high_frequency_terms if t['original_texts']['texts']]),
                'terms_without_contexts': len([t for t in high_frequency_terms if not t['original_texts']['texts']]),
                'average_frequency': sum(t['frequency'] for t in high_frequency_terms) / len(high_frequency_terms) if high_frequency_terms else 0,
                'max_frequency': max(t['frequency'] for t in high_frequency_terms) if high_frequency_terms else 0,
                'min_frequency': min(t['frequency'] for t in high_frequency_terms) if high_frequency_terms else 0
            }
        }
        
        metadata_file = output_dir / "PRDSMRT_Conversion_Metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVED] Conversion metadata: {metadata_file}")
        
        # Create integration instructions
        instructions = f"""# PRDSMRT Integration with Multi-Agent Terminology Validation System

## Conversion Results

✅ **Successfully converted PRDSMRT data to system-compatible format**

### Files Created:
- `High_Frequency_Terms.json` - {len(high_frequency_terms):,} terms ready for Step 5
- `Low_Frequency_Terms.json` - {len(low_frequency_terms):,} terms for reference
- `PRDSMRT_Conversion_Metadata.json` - Conversion details

### Integration Options:

#### Option 1: Start from Step 5 (Recommended)
```bash
# Copy High_Frequency_Terms.json to your output directory
cp High_Frequency_Terms.json agentic_validation_output_YYYYMMDD_HHMMSS/

# Run system starting from Step 5
python agentic_terminology_validation_system.py --resume-from agentic_validation_output_YYYYMMDD_HHMMSS
```

#### Option 2: Create Custom Entry Point
```python
# Modify agentic_terminology_validation_system.py to accept PRDSMRT format
system = AgenticTerminologyValidationSystem()
system.start_from_high_frequency_terms("High_Frequency_Terms.json")
```

### Data Quality Summary:
- **Total Terms**: {total_terms:,}
- **High Frequency (≥{min_frequency})**: {len(high_frequency_terms):,} terms
- **Terms with Context**: {len([t for t in high_frequency_terms if t['original_texts']['texts']]):,} terms
- **Average Frequency**: {sum(t['frequency'] for t in high_frequency_terms) / len(high_frequency_terms) if high_frequency_terms else 0:.1f}
- **Ready for Translation**: ✅ Yes

### Next Steps:
1. Review the generated High_Frequency_Terms.json
2. Integrate with the validation system
3. Run Steps 5-9 (Translation → CSV Export)
4. Get your final approved terminology CSV!
"""
        
        instructions_file = output_dir / "Integration_Instructions.md"
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        logger.info(f"[SAVED] Integration instructions: {instructions_file}")
        
        logger.info(f"[SUCCESS] Conversion completed successfully!")
        logger.info(f"[NEXT] Review {high_freq_file} and integrate with the validation system")
        
        return {
            'high_frequency_file': str(high_freq_file),
            'low_frequency_file': str(low_freq_file) if low_frequency_terms else None,
            'metadata_file': str(metadata_file),
            'instructions_file': str(instructions_file),
            'stats': {
                'total_terms': total_terms,
                'high_frequency_terms': len(high_frequency_terms),
                'low_frequency_terms': len(low_frequency_terms),
                'terms_with_contexts': len([t for t in high_frequency_terms if t['original_texts']['texts']])
            }
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Conversion failed: {e}")
        raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert PRDSMRT CSV to Multi-Agent Terminology Validation System format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_prdsmrt_to_system_format.py PRDSMRT_doc_merged_results_Processed_Simple.csv
  
  # Custom output directory and frequency threshold
  python convert_prdsmrt_to_system_format.py PRDSMRT_doc_merged_results_Processed_Simple.csv --output-dir ./converted --min-frequency 5
        """
    )
    
    parser.add_argument("input_csv", help="Path to PRDSMRT_doc_merged_results_Processed_Simple.csv")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum frequency threshold (default: 2)")
    
    args = parser.parse_args()
    
    try:
        result = convert_prdsmrt_to_high_frequency_format(
            args.input_csv, 
            args.output_dir, 
            args.min_frequency
        )
        
        print(f"\n✅ SUCCESS: Conversion completed!")
        print(f"📊 Stats: {result['stats']['high_frequency_terms']:,} high-frequency terms ready for validation")
        print(f"📁 Files: {result['high_frequency_file']}")
        print(f"📖 Instructions: {result['instructions_file']}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
