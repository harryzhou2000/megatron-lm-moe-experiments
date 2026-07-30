LATEXMK ?= latexmk
MAIN := main.tex
BUILD := build
ENGINE ?= xelatex
PYTHON ?= ../../../.venv/bin/python

.PHONY: all plots clean watch

all:
	$(LATEXMK) -$(ENGINE) -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(MAIN)

plots:
	mkdir -p /tmp/moe-presentation-cache/matplotlib /tmp/moe-presentation-cache/fontconfig
	XDG_CACHE_HOME=/tmp/moe-presentation-cache MPLCONFIGDIR=/tmp/moe-presentation-cache/matplotlib $(PYTHON) scripts/plot_results.py

watch:
	$(LATEXMK) -pvc -$(ENGINE) -interaction=nonstopmode -outdir=$(BUILD) $(MAIN)

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN)
