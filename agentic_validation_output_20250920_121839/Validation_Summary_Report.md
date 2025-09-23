# Agentic Terminology Validation System - Summary Report

**Session ID:** 20250922_215302
**Generated:** 2025-09-23 04:52:50

## Process Overview

This report summarizes the complete terminology validation workflow:
1. Initial Term Collection and Verification
2. Glossary Validation
3. New Terminology Processing
4. Frequency Analysis and Filtering
5. Translation Process
6. Language Verification
7. Final Review and Decision
8. Timestamp + Term Data Recording

## Statistics

- **Total Terms Input:** 0
- **Terms After Cleaning:** 0
- **Frequency = 1 Terms (Stored):** 0
- **Frequency > 2 Terms (Processed):** 0
- **Terms Translated:** 13,325
- **Terms Approved:** 135
- **Terms Rejected:** 4

## Output Files

- **Combined Terms:** `agentic_validation_output_20250920_121839\Combined_Terms_Data.csv`
- **Cleaned Terms:** `agentic_validation_output_20250920_121839\Cleaned_Terms_Data.csv`
- **New Terms Candidates:** `agentic_validation_output_20250920_121839\New_Terms_Candidates.json`
- **High Frequency Terms:** `agentic_validation_output_20250920_121839\High_Frequency_Terms.json`
- **Frequency Storage Export:** `agentic_validation_output_20250920_121839\Frequency_Storage_Export.json`
- **Translation Results:** `agentic_validation_output_20250920_121839\Translation_Results.json`
- **Verified Results:** `agentic_validation_output_20250920_121839\Verified_Translation_Results.json`
- **Final Decisions:** `agentic_validation_output_20250920_121839\Final_Terminology_Decisions.json`
- **Audit Record:** `agentic_validation_output_20250920_121839\Complete_Audit_Record.json`

## System Configuration

```json
{
  "glossary_folder": "glossary",
  "terminology_model": "gpt-4.1",
  "validation_model": "gpt-4.1",
  "translation_model_size": "1.3B",
  "gpu_workers": 2,
  "cpu_workers": 16
}
```

## Process Completion

[OK] All 8 steps of the terminology validation process completed successfully.
[STATS] Processing session: 20250922_215302
[FOLDER] All outputs saved to: `agentic_validation_output_20250920_121839`
