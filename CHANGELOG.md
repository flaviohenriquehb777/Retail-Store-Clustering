# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o
versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [1.0.1] — 2026-10-31

### Corrigido
- **Imputação da `distancia_concorrente`** — o valor da mediana global de referência
  estava gravado como `2.325` (≈ 2,3 metros) em [deploy/params_modelo.json](deploy/params_modelo.json)
  e [deploy/classificador.js](deploy/classificador.js), quando o correto é **2325 m**
  (valor usado no treino da Fase 3 e documentado no case). Loja nova com distância em
  branco era deslocada para um extremo do espaço e sinalizada indevidamente como outlier.
  As duas rotas de inferência já retornam o cluster esperado nos CSVs de exemplo.

### Adicionado
- **Suíte de testes** do pipeline de inferência em [tests/test_predict.py](tests/test_predict.py)
  (7 testes: 3 casos de exemplo, determinismo, validação de colunas e persistência da saída).
- **CI no GitHub Actions** ([.github/workflows/ci.yml](.github/workflows/ci.yml)): `ruff check` + `pytest`.
- Templates de Issue e Pull Request em `.github/`, além de [CONTRIBUTING.md](CONTRIBUTING.md).
- Padronização de editor: `.editorconfig` e `.gitattributes`.

### Alterado
- `requirements.txt` agora usa `-e .` (portátil) em vez de caminho absoluto local.
- `rules` do `ruff` e `pytest` declaradas explicitamente no [pyproject.toml](pyproject.toml).

### Removido
- `LICENSE.md` (duplicava o [LICENSE](LICENSE)), pacotes vazios em `src/` e
  linhas duplicadas no `.gitignore`.

## [1.0.0] — 2026-10-31

### Adicionado
- Ciclo CRISP-DM completo em **20 notebooks** (`notebooks/clean` + `notebooks/executed`):
  Fases 1 a 6, do entendimento de negócio ao plano de metas comercial.
- **Modelo final `k5_v2_prof6`** — KMeans `k=5`, `n_init=20`, `max_iter=500`, `seed=42`,
  com 6 features padronizadas (StandardScaler) e centróides congelados.
- **Pipeline de inferência sem retreino** — [deploy/predict.py](deploy/predict.py) em
  NumPy puro (Rota B) com paridade ARI = 1.0 vs scikit-learn (Rota A).
- **Aplicação web estática** `Classificador de Lojas Novas` (HTML/CSS/JS puro, tema
  escuro) publicada via GitHub Pages em [deploy/](deploy/).
- **Fonte única da verdade** dos parâmetros do modelo:
  [deploy/params_modelo.json](deploy/params_modelo.json).
- **Plano de metas comercial por loja** (CSV + XLSX) e documentos de entrega em
  [reports/entrega_fase6](reports/entrega_fase6).

### Métricas de destaque
- Gap de faturamento projetado: **R$ 265,8 milhões/ano** (comparação das lojas ao P75
  dos seus clusters).
- Ganho de **+30,2 p.p.** no R² médio e **+19,3 p.p.** no R² de Ticket Médio frente ao
  baseline cadastral (`tipo_loja` / `sortimento`).
- Estabilidade: Shuffle ARI **0,909** · Bootstrap **0,831** (10 repetições).

[1.0.1]: https://github.com/flaviohenriquehb777/Retail-Store-Clustering/releases/tag/v1.0.1
[1.0.0]: https://github.com/flaviohenriquehb777/Retail-Store-Clustering/releases/tag/v1.0.0
