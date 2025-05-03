#!/bin/bash

echo "Running build script for Jekyll"

# Create _chapters directory if it doesn't exist
mkdir -p _chapters

# Process all the chapter files
for chapter_file in chapters/chapter-*.md; do
  # Extract the chapter number and title
  filename=$(basename "$chapter_file")
  chapter_num=$(echo "$filename" | sed -n 's/chapter-0*\([0-9]\+\)-.*/\1/p')
  
  # Get the title from the file
  title=$(grep -m 1 "^# Chapter [0-9]\+:" "$chapter_file" | sed 's/^# Chapter [0-9]\+: \(.*\)/\1/')
  
  # Create a sanitized output filename
  output_file="_chapters/$(printf "%02d" $chapter_num)-$(echo $filename | sed 's/chapter-[0-9]*-//g')"
  
  # Get the description from README.md
  description=$(grep -A 5 "## Chapter $chapter_num:" README.md | grep -v "^##" | head -n 1 | sed 's/^- //')
  
  # Create the front matter and copy content
  echo "---" > "$output_file"
  echo "layout: chapter" >> "$output_file"
  echo "title: $title" >> "$output_file"
  echo "number: $chapter_num" >> "$output_file"
  echo "description: $description" >> "$output_file"
  echo "---" >> "$output_file"
  echo "" >> "$output_file"
  cat "$chapter_file" >> "$output_file"
  
  echo "Processed chapter $chapter_num: $title"
done

echo "Build script completed successfully" 