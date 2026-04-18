#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Force pdfTeX to use the full system font map so user-installed CJK fonts are visible.
export TEXFONTMAPS=/usr/share/texlive/texmf-dist/fonts/map/pdftex/updmap//

pdflatex -interaction=nonstopmode -halt-on-error main.tex

if grep -q '\\citation{' main.aux; then
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

pdflatex -interaction=nonstopmode -halt-on-error main.tex
