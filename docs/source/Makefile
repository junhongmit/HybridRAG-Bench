# Minimal makefile for Sphinx documentation

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
OUTPUTDIR     = ..
DOCTREEDIR    = ../.doctrees

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html -d "$(DOCTREEDIR)" "$(SOURCEDIR)" "$(OUTPUTDIR)"
	cp -f "$(SOURCEDIR)/_static/custom.css" "$(OUTPUTDIR)/_static/custom.css"

clean:
	rm -rf "$(DOCTREEDIR)" \
	"../_images" "../_sources" "../_static" "../get_started" "../install" \
	"../genindex.html" "../index.html" "../leaderboard.html" \
	"../objects.inv" "../search.html" "../searchindex.js" "../.buildinfo"
