# Makefile for compiling, compressing, and cleaning LaTeX files

# Root file
TEXFILE = proposal.tex

# List of additional directories (pre-textual, pos-textual, textual, figures, tables, etc.)
AUXDIRS = pre-textual pos-textual textual bibliographical template figuras

PDFFILE = $(TEXFILE:.tex=.pdf)
AUXFILES = $(TEXFILE:.tex=.aux) $(TEXFILE:.tex=.log) $(TEXFILE:.tex=.out) $(TEXFILE:.tex=.toc) $(TEXFILE:.tex=.gz) $(TEXFILE:.tex=.synctex.gz) $(TEXFILE:.tex=.fdb_latexmk) $(TEXFILE:.tex=.fls) $(TEXFILE:.tex=-figure*.pdf)

# Compile files (.bbl, .blg, .aux, .log, .out, .toc, .gz, .synctex.gz, .fdb_latexmk, .fls, -figure*.pdf)
COMPILEFILES = $(TEXFILE:.tex=.bbl) $(TEXFILE:.tex=.blg) $(TEXFILE:.tex=.aux) $(TEXFILE:.tex=.log) $(TEXFILE:.tex=.out) $(TEXFILE:.tex=.toc) $(TEXFILE:.tex=.gz) $(TEXFILE:.tex=.synctex.gz) $(TEXFILE:.tex=.fdb_latexmk) $(TEXFILE:.tex=.fls) $(TEXFILE:.tex=-figure*.pdf)

# Unrelated .aux files that are not TEXFILE:.tex=.aux
COMPILEFILES += *.aux

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
$(PDFFILE): $(TEXFILE)
	xelatex $(TEXFILE)

compress: $(PDFFILE)
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$(PDFFILE:.pdf=-compressed.pdf) $(PDFFILE)

# Write to .zip and .tar.gz
package: $(PDFFILE)
	zip -r $(PDFFILE:.pdf=.zip) $(TEXFILE) $(AUXDIRS) $(COMPILEFILES) $(PDFFILE) $(PDFFILE:.pdf=-compressed.pdf)
	tar -czvf $(PDFFILE:.pdf=.tar.gz) $(TEXFILE) $(AUXDIRS) $(COMPILEFILES) $(PDFFILE) $(PDFFILE:.pdf=-compressed.pdf)

clean:
	rm -f $(AUXFILES) $(PDFFILE) $(COMPILEFILES) $(PDFFILE:.pdf=-compressed.pdf)