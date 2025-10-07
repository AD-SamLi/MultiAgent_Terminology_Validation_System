#!/bin/bash

# Cleanup Script for Multi-Agent Terminology Validation System
# Date: October 7, 2025
# Purpose: Remove unnecessary test files and organize old files

echo "=================================="
echo "Terminology System Cleanup Script"
echo "=================================="
echo ""

# Safety check
read -p "This will remove test files and archive old files. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleanup cancelled."
    exit 1
fi

echo "Starting cleanup..."
echo ""

# 1. Remove test files
echo "[1/5] Removing test files..."
rm -f test_batch_processing.py
rm -f test_direct_batch.py
rm -f test_step7_fix.py
rm -f validate_fixes.py
rm -f test_step7_output.log
echo "  ✓ Test files removed"

# 2. Remove temporary env files
echo "[2/5] Removing temporary env files..."
rm -f current_session.env
rm -f session_vars.env
rm -f current_full_session.env
echo "  ✓ Temporary env files removed"

# 3. Archive analysis files
echo "[3/5] Archiving analysis files..."
mkdir -p archive/analysis_reports

mv BATCH_PROCESSING_FIX_SUMMARY.md archive/analysis_reports/ 2>/dev/null
mv CRITICAL_FIX_BATCH_DATA_PRESERVATION.md archive/analysis_reports/ 2>/dev/null
mv FIXES_APPLIED_REPORT.md archive/analysis_reports/ 2>/dev/null
mv ISSUE_ANALYSIS_REPORT.md archive/analysis_reports/ 2>/dev/null
mv SCORING_SYSTEM_STANDARDIZATION_CHECK.md archive/analysis_reports/ 2>/dev/null
mv SESSION_COMPARISON_ANALYSIS.md archive/analysis_reports/ 2>/dev/null
mv STEP7_*.md archive/analysis_reports/ 2>/dev/null
mv Term_Overlap_Analysis_Report.txt archive/analysis_reports/ 2>/dev/null

echo "  ✓ Analysis files archived to archive/analysis_reports/"

# 4. Archive log files
echo "[4/5] Archiving log files..."
mkdir -p archive/logs

mv *.log archive/logs/ 2>/dev/null
mv *.log.gz archive/logs/ 2>/dev/null

echo "  ✓ Log files archived to archive/logs/"

# 5. List old output folders (don't auto-delete, let user decide)
echo "[5/5] Checking old output folders..."
echo ""
echo "Old output folders found (NOT automatically deleted):"
ls -d agentic_validation_output_*/ 2>/dev/null | head -n -1
ls -d prdsmrt_full_validation_output_*/ 2>/dev/null | head -n -1
echo ""
echo "To archive old outputs manually, run:"
echo "  mkdir -p archive/old_outputs"
echo "  mv <old_folder_name> archive/old_outputs/"
echo ""

echo "=================================="
echo "Cleanup Complete!"
echo "=================================="
echo ""
echo "Summary:"
echo "  - Test files: Removed"
echo "  - Analysis files: Archived to archive/analysis_reports/"
echo "  - Log files: Archived to archive/logs/"
echo "  - Output folders: Review manually (listed above)"
echo ""
echo "System is now cleaner and ready to use!"

