#!/bin/bash
# Convenience script to run the fixed pipeline

echo "=================================="
echo "BUBBLE DETECTION - FIXED PIPELINE"
echo "=================================="
echo ""
echo "Running bubble_detection_fixed.py..."
echo ""

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the fixed pipeline
python bubble_detection_fixed.py

echo ""
echo "=================================="
echo "PIPELINE COMPLETE"
echo "=================================="
echo ""
echo "Outputs saved to:"
echo "  - outputs/feature_importance.png"
echo ""
echo "Next steps:"
echo "  1. Read QUICK_START.md for immediate actions"
echo "  2. Review FIXES_AND_IMPROVEMENTS.md for details"
echo "  3. Check SUMMARY.md for comprehensive overview"
echo ""
echo "Current performance: PR-AUC = 0.03 (weak)"
echo "Reason: Insufficient training data (only 2016-2026)"
echo "Action: Download data from 2000 onwards, add P/E + FII/DII"
echo ""
