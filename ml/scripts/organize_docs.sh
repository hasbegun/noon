#!/bin/bash
# Script to organize markdown documentation
# This moves old scattered MD files to archive and installs new organized docs

set -e

echo "============================================================"
echo "Documentation Organization Script"
echo "============================================================"
echo ""

# Backup timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="docs/archive/original_$TIMESTAMP"

echo "Creating archive directory: $ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

echo ""
echo "Moving old markdown files to archive..."
echo ""

# List of root-level MD files to archive (excluding README.md and new docs)
OLD_MD_FILES=(
    "AGGRESSIVE_MEMORY_OPTIMIZATION.md"
    "ARCHITECTURE_CHANGE_SUMMARY.md"
    "CONFIGURATION_REVIEW.md"
    "DATASET_EMPTY_MASKS_FIX.md"
    "DATASETS_AND_INCREMENTAL_TRAINING.md"
    "DEPLOYMENT_GUIDE.md"
    "FILE_DESCRIPTOR_FIX.md"
    "HIGH_QUALITY_TRAINING_STRATEGY.md"
    "MEMORY_FIX_SUMMARY.md"
    "MODEL_TESTING_PLAN.md"
    "OPTION_B_IMPLEMENTATION_COMPLETE.md"
    "PROMPTING_CHEAT_SHEET.md"
    "PROMPTING_LESSONS_LEARNED.md"
    "RECOGNITION_ARCHITECTURE.md"
    "REFACTORING_SUMMARY.md"
    "TESTING_QUICKSTART.md"
    "TRAINING_FIX_SUMMARY.md"
    "TRAINING_FIX.md"
    "TRAINING_IMPROVEMENTS_SUMMARY.md"
    "TRAINING_PROCEDURE.md"
    "TRAINING_QUICKSTART.md"
)

# Move root-level MD files
for file in "${OLD_MD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ Archiving $file"
        mv "$file" "$ARCHIVE_DIR/"
    fi
done

# Archive old docs/ content (except archive folder itself)
if [ -d "docs" ]; then
    echo ""
    echo "Archiving old docs/ contents..."
    for item in docs/*; do
        if [ "$(basename "$item")" != "archive" ]; then
            echo "  ✓ Archiving $(basename "$item")"
            mv "$item" "$ARCHIVE_DIR/"
        fi
    done
fi

echo ""
echo "Installing new organized documentation..."
echo ""

# Copy new docs from docs_new to docs/
if [ -d "docs_new" ]; then
    cp -r docs_new/* docs/
    echo "  ✓ Installed new organized docs to docs/"

    # Clean up docs_new
    rm -rf docs_new
    echo "  ✓ Cleaned up docs_new/"
else
    echo "  ⚠  Warning: docs_new/ not found"
fi

echo ""
echo "============================================================"
echo "Documentation Organization Complete!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  - Archived ${#OLD_MD_FILES[@]} root-level MD files"
echo "  - Archived old docs/ contents"
echo "  - Installed new organized documentation"
echo ""
echo "Old files location: $ARCHIVE_DIR"
echo "New docs location: docs/"
echo ""
echo "Next steps:"
echo "  1. Read docs/README.md for navigation"
echo "  2. Start with docs/01-SETUP.md if setting up"
echo "  3. See docs/02-TRAINING.md for training"
echo ""
echo "============================================================"
