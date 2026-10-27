# SolvedOR: the article

LaTeX source of *SolvedOR: Uma Ferramenta de Software para Aprendizagem Ativa de Pesquisa
Operacional* (in Portuguese), the undergraduate thesis article behind this app.

- **Author**: João Vitor Pinheiro da Costa (Vitor Pinheiro)
- **Advisor**: Gabriela de Mattos Veroneze
- Departamento de Engenharia de Produção, Universidade Federal do Amazonas (UFAM), 2023

The article describes the 2023 version of the app, with its earlier `p:` / `max:` / `r:` syntax.
The current syntax is in the [main README](../README.md).

## Build

The [paper workflow](../.github/workflows/paper.yml) compiles it on every change to `paper/`
and attaches the PDF to the run. Locally, with a TeX distribution installed:

```sh
cd paper
latexmk -pdf article.tex
```

## Files

- `article.tex`: the article
- `references.bib`: bibliography (ABNT style through `abntex2cite`)
- `arxiv.sty`: page style, adapted from [kourgeorge/arxiv-style](https://github.com/kourgeorge/arxiv-style)
- `images/`: figures (questionnaire results are aggregated percentages)
