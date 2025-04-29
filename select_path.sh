#!/bin/bash

# Simple script to interactively select a file or directory path
# Uses fzf if available, otherwise provides a basic find prompt.

# Check if fzf is installed
if command -v fzf > /dev/null 2>&1; then
    # Use fzf to browse files and directories, starting from the current directory
    # Adjust find command as needed (e.g., add -type f for only files)
    selected=$(find . -print | fzf --prompt="Select Path: " --height=40% --layout=reverse --border)
else
    echo "fzf not found. Using basic find. Enter path manually based on output:" >&2
    find . -maxdepth 3 # Limit depth for brevity
    read -p "Enter path: " selected
fi

# If a selection was made (fzf can return empty if Esc is pressed)
if [[ -n "$selected" ]]; then
    # Output the absolute path
    realpath "$selected"
fi

exit 0 