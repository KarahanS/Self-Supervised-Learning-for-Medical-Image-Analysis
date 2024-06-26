#!/bin/bash

# Hard-coded directory path
DIRECTORY="/graphics/scratch2/students/kargibo/checkpoints/kivanc"

# Find all checkpoint files that contain 'curr-ep-' in their name and end with '.ckpt'
CHECKPOINT_FILES=$(find "$DIRECTORY" -type f -name "*curr-ep-*.ckpt")

# Check if any checkpoint files were found
if [ -z "$CHECKPOINT_FILES" ]; then
  echo "No checkpoint files found to delete."
  exit 0
fi

# Convert the string to an array
IFS=$'\n' read -r -d '' -a CHECKPOINT_ARRAY <<< "$CHECKPOINT_FILES"

# Count the number of checkpoint files
NUM_CHECKPOINTS=${#CHECKPOINT_ARRAY[@]}

# Print the number of checkpoint files and ask for confirmation
echo "Found $NUM_CHECKPOINTS checkpoint files to delete:"
for FILE in "${CHECKPOINT_ARRAY[@]}"; do
  echo "$FILE"
done
read -p "Are you sure you want to delete these files? (y/n): " CONFIRMATION

# Delete the checkpoint files if confirmed by the user
if [ "$CONFIRMATION" = "y" ] || [ "$CONFIRMATION" = "Y" ]; then
  for FILE in "${CHECKPOINT_ARRAY[@]}"; do
    rm "$FILE"
  done
  echo "Checkpoint files deleted."
else
  echo "Deletion aborted by user."
  exit 1
fi
