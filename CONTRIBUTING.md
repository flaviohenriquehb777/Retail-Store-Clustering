# Contribuindo

Obrigado pelo interesse em contribuir com o **Classificador de Lojas Novas**.
Este documento descreve o fluxo de trabalho esperado.

## Pré-requisitos

- Python **3.11+**
- [UV](https://docs.astral.sh/uv/getting-started/installation/) (recomendado)

## Configuração do ambiente

```bash
git clone <URL DO REPOSITÓRIO>
cd Retail-Store-Clustering
uv sync --extra dev
```

## Fluxo de trabalho

1. Crie uma branch a partir de `main` com nome descritivo:
   `feat/nome-curto`, `fix/nome-curto` ou `docs/nome-curto`.
2. Faça alterações pequenas e focadas (uma responsabilidade por commit).
3. Antes de abrir o PR, valide localmente:

```bash
uv run pytest          # testes do pipeline de inferência
uv run ruff check .    # lint
```

4. Abra o Pull Request preenchendo o template
   ([.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)).
   O CI (`ruff` + `pytest`) precisa passar.

## Convenções

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/pt-br/)
  (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`), em português.
- **Idioma:** código, comentários, nomes de variáveis/colunas e documentação em **pt-BR**.
- **Estilo Python:** `black` (line-length 100) e `ruff`; type hints e `from __future__ import annotations`.
- **Notebooks:** edite os arquivos em `notebooks/clean/` (sem outputs) e regenere os
  `executed` via `gerar_todos_os_notebooks.py`. Nunca edite os `executed` à mão.
- **Reprodutibilidade:** use sempre `SEED = 42`. Não reintroduza contagens/somas de dias
  como features (buraco de dados de 184 dias) nem a flag `distancia_foi_imputada` na
  matriz X de modelagem (data leakage).

## Reportando problemas

Use os templates de Issue em `.github/ISSUE_TEMPLATE/` (bug report ou feature request).
