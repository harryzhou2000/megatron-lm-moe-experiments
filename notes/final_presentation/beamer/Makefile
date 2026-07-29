LATEXMK ?= latexmk
MAIN := main.tex
BUILD := build
ENGINE ?= xelatex

.PHONY: all clean watch

all:
	$(LATEXMK) -$(ENGINE) -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(MAIN)

watch:
	$(LATEXMK) -pvc -$(ENGINE) -interaction=nonstopmode -outdir=$(BUILD) $(MAIN)

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN)
