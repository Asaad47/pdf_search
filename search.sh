#!/bin/zsh

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source conda initialization for zsh
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
conda activate pdf

# Run the search script with all arguments passed to this script
python "$SCRIPT_DIR/search.py" "$@" -i