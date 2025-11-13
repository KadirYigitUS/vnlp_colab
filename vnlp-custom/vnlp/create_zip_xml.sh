#!/bin/bash
# file_loc: create_zip_xml.sh
# version_number: 4.1
# description: Scans the LinguaStrata project and packs all necessary source and documentation files into a single, well-formed XML file for AI consumption.
# known_issues_addressed: Updated IGNORE_LIST to include root-level .md and .txt files, which are now part of the project's core documentation set.

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
shopt -s nullglob # Expands to nothing if no matches are found

SCAN_DIR="${1:-.}"
# Sanitize the directory name for use in the filename.
SANITIZED_NAME=$(basename "$(realpath "$SCAN_DIR")" | sed 's/[^a-zA-Z0-9]/-/g')
# Get current datetime in YYYY-MM-DD_HH.MM format
DATETIME=$(date +"%Y-%m-%d_%H.%M")
OUTPUT_FILE="AI-input-${SANITIZED_NAME}_${DATETIME}.zip.xml"
# The root XML tag is dynamically derived from the output filename.
ROOT_TAG_NAME=$(basename "$OUTPUT_FILE")

# --- IGNORE LOGIC (v4.1) ---
# Define patterns to ignore. Directories should not have trailing slashes.
# NOTE: *.md and *.txt are intentionally NOT ignored to include project documentation.
IGNORE_LIST=(
    # Version Control
    ".git"
    ".svn"
    ".hg"

    # Caches and Build Artifacts
    "__pycache__"
    ".ruff_cache"
    ".pytest_cache"
    ".conda"
    "build"
    "dist"
    "node_modules"
    ".env"
    "venv"
    ".venv"
    "resources"
    
    # Project-Specific Directories to Exclude
    "input"
    "output"
    "retired"
    "backup"
    "docs" # General docs, not the root .md files
    "enhanced_output" # From previous runs
    "demo_output" # From previous runs

    # File Patterns to Exclude
    "*.zip.xml" # Exclude self
    "*.zip"
    "*.log"
    "*.tmp"
    "*.swp"
    "*.docx"
    "*.xlsx"
    "*.pdf"
    "*.pptx"
    "*.csv"
    "*.md"
    "*.txt"
    "*.tar.xz"
    "*.tar.gz"
    "*.tsv"
    "*.png"

    # Specific Files
    "create_zip_xml.sh"
    "*.code-workspace"
    "*.json"
)

# --- Functions ---

generate_file_summary() {
    # This heredoc contains the static summary with zero indentation.
    cat << 'EOF'
<file_summary>
This section contains a summary of this file.
<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>
<file_format>
The content is organized as follows:
1. This summary section
2. Directory structure
3. File contents
</file_format>
<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
</usage_guidelines>
<notes>
- Some files may have been excluded based on the script's configuration.
- Binary files are not included in this packed representation.
</notes>
</file_summary>
EOF
}

generate_directory_structure() {
    echo "<directory_structure>"
    # Build the ignore pattern for the `tree` command. The `|` works as OR.
    local tree_ignore_pattern
    tree_ignore_pattern=$(printf "%s|" "${IGNORE_LIST[@]}")
    tree_ignore_pattern="${tree_ignore_pattern%|}" # Remove trailing |

    # Use `tree` for high-fidelity output.
    if command -v tree &>/dev/null; then
        # The `cd` is in a subshell to not affect the script's PWD.
        (cd "$SCAN_DIR" && tree -aF --prune -I "$tree_ignore_pattern")
    else
        echo "WARNING: 'tree' command not found. Directory structure will be omitted."
    fi
    echo "</directory_structure>"
}

# --- Main Execution ---

echo "Starting LinguaStrata packing process for directory '$SCAN_DIR'..."
echo "Output will be written to '$OUTPUT_FILE'"

# Start with a clean file and write the dynamic root element
echo "<$ROOT_TAG_NAME>" > "$OUTPUT_FILE"

# 1. Write the static file summary
generate_file_summary >> "$OUTPUT_FILE"

# 2. Write the directory structure
generate_directory_structure >> "$OUTPUT_FILE"

# 3. Write the file contents
echo "<files>" >> "$OUTPUT_FILE"
echo "This section contains the contents of the repository's files." >> "$OUTPUT_FILE"

# --- DEFINITIVE FIND LOGIC ---
# Build the find command's exclusion arguments.
prune_paths=()
prune_names=()

# Separate patterns into path-based and name-based ignores.
for item in "${IGNORE_LIST[@]}"; do
    if [[ "$item" == *"*"* ]]; then
        # This is a glob pattern for filenames, like "*.md"
        prune_names+=("-o" "-name" "$item")
    else
        # This is a directory or exact filename, like ".git" or "create_zip_xml.sh"
        prune_paths+=("-o" "-path" "*/$item")
        prune_paths+=("-o" "-path" "*/$item/*")
    fi
done

find "$SCAN_DIR" \
    `# Prune based on path matches. Start with a dummy -false to handle the first -o` \
    \( -false "${prune_paths[@]}" \) -prune \
    -o \
    `# Prune based on name matches.` \
    \( -false "${prune_names[@]}" \) -prune \
    -o \
    `# If not pruned, check if it's a file and print it.` \
    \( -type f -print \) | while IFS= read -r filepath; do
    
    # Check if the file is binary. Skip if it is.
    if [[ ! -s "$filepath" ]] || file -b --mime-type "$filepath" | grep -qvE '^(text/|application/xml|application/x-sh)'; then
        echo "  -> Skipping binary or empty file: $filepath"
        continue
    fi
    
    echo "  -> Processing text file: $filepath"

    # Add a blank line for readability.
    echo "" >> "$OUTPUT_FILE"
    
    # Write the opening file tag with the escaped path attribute.
    filepath_escaped=$(sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&apos;/g' <<< "$filepath")
    echo "  <file path=\"$filepath_escaped\">" >> "$OUTPUT_FILE"
    
    # Process content: escape XML special characters.
    sed 's/\&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' "$filepath" >> "$OUTPUT_FILE"
    
    # Write the closing file tag.
    echo "" >> "$OUTPUT_FILE"
    echo "  </file>" >> "$OUTPUT_FILE"
done

echo "</files>" >> "$OUTPUT_FILE"
echo "</$ROOT_TAG_NAME>" >> "$OUTPUT_FILE"

echo "Packing process complete. Output written to '$OUTPUT_FILE'"