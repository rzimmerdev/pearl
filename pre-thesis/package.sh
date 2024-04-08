#!/bin/bash

# Package LaTeX files and subdirectories into a .zip file

# Set the name of the zip file
ZIP_FILE="pre_project.zip"

# Remove existing zip file if present
rm -f "$ZIP_FILE"

# Add .tex files and subdirectories to the zip file
zip -r "$ZIP_FILE" *.tex *.pdf *.cls *.bib *.png *.xcf */

echo "LaTeX files and subdirectories packaged into $ZIP_FILE."
