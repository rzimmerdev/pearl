OUTDIR = out
TEXFILES = $(shell find . -type f -name "*.tex")
SUBDIRS = $(sort $(dir $(TEXFILES)))
OUTSUBDIRS = $(patsubst ./%, $(OUTDIR)/%, $(SUBDIRS))
pdflatex = /usr/local/texlive/2024/bin/x86_64-linux/pdflatex
bibtex = /usr/local/texlive/2024/bin/x86_64-linux/bibtex

PDF = $(OUTDIR)/main.pdf

all: $(OUTSUBDIRS) $(PDF)

# Create output subdirectories
$(OUTSUBDIRS):
	mkdir -p $@

# Compile LaTeX document
$(PDF): main.tex | $(OUTSUBDIRS)
	mkdir -p $(OUTDIR)
	$(pdflatex) -output-directory=$(OUTDIR) main.tex
	$(bibtex) $(OUTDIR)/main
	$(pdflatex) -output-directory=$(OUTDIR) main.tex
	$(pdflatex) -output-directory=$(OUTDIR) main.tex

clean:
	find $(OUTDIR) -type f ! -name "main.pdf" -delete
	find $(OUTDIR) -mindepth 1 -type d -empty -delete
