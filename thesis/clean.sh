#!/bin/bash

# Clean LaTeX output files

# Remove auxiliary files
rm -f *.aux *.bbl *.blg *.idx *.lof *.log *.loq *.lot

# Remove LaTeX-generated files
rm -f *.out *.toc *.synctex.gz

rm -f *.zip *.pdf

echo "Cleaned LaTeX output files."

