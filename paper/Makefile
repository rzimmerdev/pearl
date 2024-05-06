# Makefile for compiling, compressing, and cleaning LaTeX files

# Root file
TEXFILE = thesis.tex

# List of additional directories (pre-textual, pos-textual, textual, figures, tables, etc.)
DIRS = tex images packages

PDFFILE = $(TEXFILE:.tex=.pdf)
AUXFILES = $(TEXFILE:.tex=.aux) $(TEXFILE:.tex=.log) $(TEXFILE:.tex=.out) $(TEXFILE:.tex=.toc) $(TEXFILE:.tex=.gz) $(TEXFILE:.tex=.synctex.gz) $(TEXFILE:.tex=.fdb_latexmk) $(TEXFILE:.tex=.fls) $(TEXFILE:.tex=-figure*.pdf)

# Extensions to remove (list)
COMPILE = .bbl .blg .aux .log .out .toc .gz .synctex.gz .fdb_latexmk .fls -figure*.pdf .idx .lof .loq .lot

# Targets
all: $(PDFFILE)

#$(PDFFILE): $(TEXFILE)
#    pdflatex $(TEXFILE)
#
#compress: $(PDFFILE)
#    gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(PDFFILE:.pdf=-compressed.pdf) $(PDFFILE)
#
#clean:
#    rm -f $(AUXFILES) $(PDFFILE) $(PDFFILE:.pdf=-compressed.pdf)

# run without asking for user input
$(PDFFILE): $(TEXFILE)
	xelatex $(TEXFILE) </dev/null

compress: $(PDFFILE)
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(PDFFILE:.pdf=-compressed.pdf) $(PDFFILE)

# Write to .zip and .tar.gz
package: $(PDFFILE)
	zip -r $(PDFFILE:.pdf=.zip) $(TEXFILE) $(DIRS)
	tar -czvf $(PDFFILE:.pdf=.tar.gz) $(TEXFILE) $(DIRS)

clean:
	rm -f $(AUXFILES) $(PDFFILE) $(PDFFILE:.pdf=-compressed.pdf)
	for ext in $(COMPILE); do rm -f $(TEXFILE:.tex=$$ext); done
