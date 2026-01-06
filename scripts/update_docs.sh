#!/bin/bash
# Convenience script to regenerate documentation for all utility modules

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find all Python utility modules
UTILS_DIR="$PROJECT_ROOT/src/python/core"
OUTPUT_DIR="$PROJECT_ROOT/docs/doc/code"

echo "Generating documentation for Python utility modules..."
echo "Source directory: $UTILS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Find all .py files in the utils directory (excluding __init__.py and private modules)
PYTHON_FILES=$(find "$UTILS_DIR" -name "*.py" ! -name "__init__.py" ! -name "_*.py")

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python utility files found in $UTILS_DIR"
    exit 1
fi

# Generate documentation
python "$SCRIPT_DIR/generate_docs.py" $PYTHON_FILES --output-dir "$OUTPUT_DIR" --nav-order 50

echo ""
echo "Documentation generation complete!"
echo "Generated files are in: $OUTPUT_DIR"
