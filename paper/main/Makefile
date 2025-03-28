OUTDIR = out
TEXFILES = $(shell find . -type f -name "*.tex")
SUBDIRS = $(sort $(dir $(TEXFILES)))
OUTSUBDIRS = $(patsubst ./%, $(OUTDIR)/%, $(SUBDIRS))

PDF = $(OUTDIR)/main.pdf

all: $(OUTSUBDIRS) $(PDF)

# Create output subdirectories
$(OUTSUBDIRS):
	mkdir -p $@

# Compile LaTeX document
$(PDF): main.tex | $(OUTSUBDIRS)
	pdflatex -output-directory=$(OUTDIR) main.tex
	bibtex $(OUTDIR)/main
	pdflatex -output-directory=$(OUTDIR) main.tex
	pdflatex -output-directory=$(OUTDIR) main.tex

clean:
	find $(OUTDIR) -type f ! -name "main.pdf" -delete
	find $(OUTDIR) -mindepth 1 -type d -empty -delete