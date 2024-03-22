#!/bin/bash

# Clean LaTeX output files

# Remove auxiliary files
find . -type f -name "*.aux" -delete
find . -type f -name "*.bbl" -delete
find . -type f -name "*.blg" -delete
find . -type f -name "*.idx" -delete
find . -type f -name "*.lof" -delete
find . -type f -name "*.log" -delete
find . -type f -name "*.loq" -delete
find . -type f -name "*.lot" -delete

# Remove LaTeX-generated files
find . -type f -name "*.out" -delete
find . -type f -name "*.toc" -delete
find . -type f -name "*.synctex.gz" -delete

rm -f *.zip

echo "Cleaned LaTeX output files."
