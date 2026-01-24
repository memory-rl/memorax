# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html livehtml

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SOURCEDIR)/api/_autosummary

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
