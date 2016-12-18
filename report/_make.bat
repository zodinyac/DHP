pdflatex -aux-directory=_build -output-directory=_build main
bibtex _build/main
pdflatex -aux-directory=_build -output-directory=_build main
pdflatex -aux-directory=_build -output-directory=_build main

start "" "%cd%\_build\main.pdf"
