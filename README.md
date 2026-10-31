<div align="center">
  <img src="./deploy/assets/logo-fh-data.png" alt="FH Data" height="52">
  <h1>🏬 Classificador de Lojas Novas<br>Clusterização Não-Supervisionada de 1.115 Lojas de Varejo</h1>
  <p align="center"><i>Do entendimento de negócio ao deploy em GitHub Pages — metodologia CRISP-DM 6 fases.</i></p>
  <p align="center">
    <a href="https://www.python.org/downloads/release/python-3110/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white"></a>
    <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
    <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"><img alt="KMeans k=5" src="https://img.shields.io/badge/KMeans-k%3D5-F7931E?logo=scikitlearn&logoColor=white"></a>
    <a href="https://www.datascience-pm.com/crisp-dm-2/"><img alt="Metodologia CRISP-DM" src="https://img.shields.io/badge/Metodologia-CRISP--DM-FF7F24"></a>
    <a href="https://opensource.org/license/mit"><img alt="Licença MIT" src="https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow?logo=licensezero&logoColor=white"></a>
    <a href="https://pages.github.com/"><img alt="GitHub Pages" src="https://img.shields.io/badge/GitHub%20Pages-181717?logo=githubpages&logoColor=white"></a>
    <a href="#"><img alt="Status Projeto" src="https://img.shields.io/badge/Status-Finalizado_%E2%9C%85-success?logo=cachet&logoColor=white"></a>
  </p>
</div>

<div align="center" style="margin: 28px 0 8px 0;">
  <a href="https://flaviohenriquehb777.github.io/Retail-Store-Clustering/index.html"
     target="_blank"
     rel="noopener noreferrer"
     title="Abrir Classificador de Lojas Novas — página inteira, SEM chrome do GitHub">
    <img
      src="./assets/thumbnail-classificador-lojas-novas.png"
      alt="Miniatura do Classificador de Lojas Novas — clique para abrir a aplicação em página inteira"
      width="720"
      style="
        border-radius: 14px;
        border: 1px solid #2b3270;
        box-shadow:
          0 1px 3px   rgba(99,102,241,0.10),
          0 8px 20px  rgba(99,102,241,0.18),
          0 18px 40px rgba(0,0,0,0.45);
        transition: transform .25s ease, box-shadow .25s ease;
      "
      onmouseover="this.style.transform='translateY(-2px) scale(1.006)'; this.style.boxShadow='0 2px 5px rgba(99,102,241,0.13), 0 14px 30px rgba(99,102,241,0.24), 0 26px 56px rgba(0,0,0,0.55)';"
      onmouseout="this.style.transform=''; this.style.boxShadow='';">
  </a>
  <p align="center" style="margin-top:12px; color:#8b93c7; font-size:13.5px;">
    👆 <a href="https://flaviohenriquehb777.github.io/Retail-Store-Clustering/index.html" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:#818cf8; font-weight:600;">Clique na imagem → abre a APLICAÇÃO em página inteira</a>
    (nova aba · só o classificador, sem header do GitHub).<br>
    <span style="font-size:12.5px;">⚠️ Pré-requisito: antes de clicar, habilite Pages em Settings → Pages → Branch <b>main</b> / Folder <b>/deploy</b>.</span>
  </p>
</div>

---

## 📑 Sumário

1.  [Visão Geral do Modelo](#1-visão-geral-do-modelo)
2.  [Objetivos da Análise](#2-objetivos-da-análise)
3.  [Estrutura do Modelo (5 Clusters)](#3-estrutura-do-modelo-5-clusters)
4.  [Bases de Dados](#4-bases-de-dados)
5.  [Metodologia de Análise · CRISP-DM 6 fases](#5-metodologia-de-análise--crisp-dm-6-fases)
6.  [Resultados Chave e Apresentação](#6-resultados-chave-e-apresentação)
7.  [Tecnologias Utilizadas](#7-tecnologias-utilizadas)
8.  [Instalação e Uso](#8-instalação-e-uso)
9.  [Publicação no GitHub Pages](#9-publicação-no-github-pages)
10. [Licença (MIT)](#10-licença-mit)
11. [Contato](#11-contato)

---

## 1. Visão Geral do Modelo

Uma rede varejista com **1.115 lojas** reformula a forma de definir metas e comparar desempenho. Em vez de comparar uma loja de shopping de alto fluxo diretamente com uma loja de periferia (população e perfil de demanda completamente diferentes), este projeto constrói **5 grupos de lojas PARECIDAS (clusters)** via KMeans k=5 com 6 features de comportamento operacional e geolocalização.

Com os clusters definidos, a empresa consegue 4 entregáveis acionáveis:
- **Comercial:** metas de crescimento por **grupo de pares** (1 linha / loja · média ponderada +5,0000% exato calibrado via α linear · planilha com 2 abas XLSX: **Metas por Loja** + **Resumo Executivo**);
- **Trade Marketing:** priorização de verba promocional apenas para os clusters onde o incremento em promo gera uplift real (cluster **C02 Promo-Dependent**);
- **Supply:** calibração de mix de produto e volumes de reposição por padrão de demanda sazonal (**C03 Sazonal Raro**);
- **Expansão:** enquadramento de **lojas novas em grupos SEM retreinar o modelo** (aplicativo web estático 100% cliente-side em GitHub Pages, SEM backend, classificação em <1 s).

> 📚 Documento de referência do case (escopo original, dicionário de dados e perguntas do negócio): [references/Retail Store Clustering.md](./references/Retail%20Store%20Clustering.md).
>
> 📘 Resposta às **6 perguntas do negócio** (formato .docx, com tabelas R²/riscos/planilha financeira): [reports/Perguntas do Case - Respostas Negócio.docx](./reports/Perguntas%20do%20Case%20-%20Respostas%20Neg%C3%B3cio.docx).

---

## 2. Objetivos da Análise

Os 5 critérios de sucesso do projeto (metodologia do case, todos cumpridos ✅):

| # | Critério | Como foi medido | Resultado Final |
|---|---|---|---|
| 1 | **Interpretável** | Todo cluster tem nome que um gerente de loja entende sem Dicionário de Dados | 5 perfis: C01 Alto Fluxo · C04 Standard · C00 Ticket Premium · C02 Promo-Dependent · C03 Sazonal Raro |
| 2 | **Acionável** | Cada cluster leva a uma decisão diferente por área | Tabela com 15 ações (3 áreas × 5 perfis) entregue na Fase 5 · plano de metas Excel 1 linha por loja |
| 3 | **Estável** | Re-embaralhar a base ou remover 20% das lojas NÃO redesenha os grupos | Shuffle ARI **0,909** · Random states ≥ **0,910** · Bootstrap 10× média **0,831** (1 rodada o C03 — 4,4% — é absorvido, tratado no deploy) |
| 4 | **Melhor que baseline** | Supera a segmentação cadastral atual (`tipo_loja` / `sortimento`) | **+30,2 pp de R² médio** (14,7% → 44,9%) · Ticket Médio R² sobe **+19,3 pp** (42,3% tipo_loja → 61,6% Cluster K5) |
| 5 | **Quantificado** | Entrega termina em R$, não em silhueta | **R$ 265,8 MILHÕES / ano** de gap total realizável (P75 − loja_i · 260 dias úteis · 6 premissas conservadoras) |

---

## 3. Estrutura do Modelo (5 Clusters)

**Feature Set FINAL de modelagem (6 colunas — alinhado ao professor/Sol):**
`clientes_medio_dia_log1p` · `ticket_medio` · `razao_promo` · `cv_volatilidade_diaria` · `cv_sazonalidade_mensal_log1p` · `distancia_concorrente_log1p`.

Baselines **tipo_loja** e **sortimento** NUNCA entram no modelo — usados apenas pós-fato na comparação do critério 4.

Ordenados por Venda Média Dia (maior → menor):

| Ordem | Cluster | Nome Comercial | Lojas | % rede | Clientes /dia | Ticket R$ | Razão Promo | CV Vol | CV Saz | Dist Concorr. (m) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **C01** | Alto Fluxo / Ponto de Passagem | 193 | 17,3% | 1.212 | 7,45 | 1,340 | 0,249 | 0,098 | 433 |
| 2 | **C04** | Standard / Classe Média | 357 | 32,0% | 761 | 9,09 | 1,358 | 0,239 | 0,080 | 4.339 |
| 3 | **C00** | Ticket Premium / Mix Caro | 287 | 25,7% | 553 | 12,03 | 1,341 | 0,251 | 0,097 | 8.221 |
| 4 | **C02** | Promo-Dependent | 229 | 20,5% | 630 | 9,52 | 1,669 | 0,348 | 0,090 | 6.904 |
| 5 | **C03** | Sazonal Raro / Temporada | 49 | 4,4% | 669 | 8,92 | 1,360 | 0,344 | 0,220 | 9.061 |
| | **Rede** | 1.115 lojas totais | 1.115 | 100% | 794 | 9,66 | 1,420 | 0,272 | 0,100 | 5.671 |

> 🔬 Perfil médio de referência oficial do modelo (CSV): [deploy/perfil_medio_clusters_referencia.csv](./deploy/perfil_medio_clusters_referencia.csv)
>
> 💾 Parâmetros congelados para Expansão (μ, σ, 5 centróides, mapeamento de labels e flags): [deploy/params_modelo.json](./deploy/params_modelo.json)

---

## 4. Bases de Dados

| Arquivo (em `data/01_raw/`) | Granularidade | Tamanho | Período | O que descreve |
|---|---|---:|---|---|
| **loja.csv** | 1 linha por loja | 1.115 × 5 | estático (cadastro) | cadastro da loja (id, tipo_loja a/b/c/d, sortimento basic/plus/extended, se abre no domingo, distância ao concorrente mais próximo em metros). |
| **treino.csv** | 1 linha por loja × dia | 1.017.209 × 9 | 01/01/2013 a 31/07/2015 | série temporal de demanda: venda total do dia R$, clientes atendidos no dia, flag abriu ou não, quantidade de produtos em promo, flag loja em reforma. |

### Tratativas de Data Quality (já aplicadas)
- **181 lojas com série incompleta** (180 delas com buraco comum 01/jul/2014 a 31/dez/2014). Nenhuma feature usa contagem de dias (tudo média/razão/proporção — decisão de negócio do usuário).
- **3 nulos de `distancia_concorrente`** (lojas 291, 622, 879) → imputados com **mediana GLOBAL de 2.325 m**. Flag auditoria criada, mas NÃO entra no X de modelagem (data leakage).
- Assimetrias tratadas com `log1p(x) = ln(x+1)`: `venda_media_dia`, `clientes_medio_dia`, `distancia_concorrente`, `cv_sazonalidade_mensal`.
- Todas as seeds fixadas em **42** (KMeans `random_state`, `n_init=20`, `max_iter=500`, shuffle, splits bootstrap).
- StandardScaler aplicado **apenas no início da Fase 4** (nunca na Fase 3, por decisão do usuário).

---

## 5. Metodologia de Análise · CRISP-DM 6 fases

O projeto segue a metodologia CRISP-DM tradicional — 6 fases, uma por vez:

```
┌───────────────┐
│  1. NEGÓCIO   │ ← Kick-off, reuniões, 5 critérios sucesso, árvore de decisão.
└───────┬───────┘
        v
┌───────────────┐
│  2. DADOS     │ ← Dicionário, data quality 181 lojas, baselines tipo_loja / sortimento.
└───────┬───────┘
        v
┌───────────────┐
│  3. PREPARAÇÃO│ ← 4 blocos features, log1p, imputação dist., Spearman 8→6 cols.
└───────┬───────┘
        v
┌───────────────┐
│  4. MODELAGEM │ ← KMeans k=2..10 cotovelo silhueta, Ward dendrograma, k=5 por negócio.
└───────┬───────┘
        v
┌───────────────┐
│  5. AVALIAÇÃO │ ← Estabilidade, ANOVA R² vs baseline, $, concentrações tipo_b.
└───────┬───────┘
        v
┌───────────────┐
│  6. DEPLOY    │ ← Planilha de Metas 1/loja · Predict.py · App GitHub Pages.
└───────────────┘
```

Os 20 notebooks CLEAN (código limpo) e EXECUTED (com outputs) estão na pasta [notebooks/](./notebooks/), numerados de **01_Fase_1_Entendimento_do_Negocio** até **20_Fase_6_Deploy_Plano_de_Metas_Comercial**.

---

## 6. Resultados Chave e Apresentação

### 🔢 Números que a diretoria quer ver
| Indicador | Valor | O que significa |
|---|---:|---|
| **R² médio vs tipo_loja (6 features)** | **+30,2 pp** (14,7% → 44,9%) | Cluster explica **3,1× mais** da variância das lojas do que o `tipo_loja` cadastral. |
| **R² Ticket Médio vs tipo_loja** | **+19,3 pp** (42,3% → 61,6%) | Cluster captura Premiumização de mix 49% melhor que o segmento cadastral. |
| **Estabilidade (Bootstrap 10× 80%)** | **ARI médio 0,831 · mín 0,396** | 1 única rodada absorve o C03 (Sazonal Raro = 4,4% da rede). Resolvido no deploy: classificação SEM retreino. |
| **Estabilidade Shuffle** | **ARI 0,909** | Reordenando 100% das linhas, os grupos continuam os mesmos. |
| **Lojas tipo_b (17 unidades)** | **76,5% → C01 Alto Fluxo** | Mesmo **sem nunca ver** a coluna tipo_loja no treino, o KMeans agrupa corretamente. |
| **Gap realizável rede toda** | **R$ 265,8 MILHÕES / ano** | Uplift de 9,9% ao mover todas as lojas até o P75 do próprio grupo, com 6 premissas conservadoras. |
| **Plano de metas** | **Média ponderada = +5,0000% EXATO** | Regra P75 + meta 5% base + (1 − percentil) × gap × 0,55 · piso 1,5% · teto 10,49% · α linear calibrado. |

### 📦 Artefatos finais entregues

1. **Planilha de Metas 1 linha por loja:** [reports/entrega_fase6/fase6_plano_metas_comercial_por_loja.xlsx](./reports/entrega_fase6/fase6_plano_metas_comercial_por_loja.xlsx) — 2 abas: **Metas por Loja** + **Resumo Executivo**.
2. **Standalone Python SEM retreino:** [deploy/predict.py](./deploy/predict.py) + [deploy/requirements_para_predicao.txt](./deploy/requirements_para_predicao.txt). Usa NumPy puro como rota padrão e sklearn `.pkl` como sanity check opcional.
3. **Parâmetros do modelo (Fonte Única da Verdade):** [deploy/params_modelo.json](./deploy/params_modelo.json).
4. **3 CSVs de lojas novas para Expansão testar:** [deploy/exemplos/](./deploy/exemplos/).
5. **Aplicação Web Estática GitHub Pages:** [deploy/index.html](./deploy/index.html) · [deploy/estilo.css](./deploy/estilo.css) · [deploy/classificador.js](./deploy/classificador.js). Tema escuro · responsivo · 2 abas (form 1 loja / lote CSV).
6. **Respostas docx 6 perguntas negócio:** [reports/Perguntas do Case - Respostas Negócio.docx](./reports/Perguntas%20do%20Case%20-%20Respostas%20Neg%C3%B3cio.docx).

---

## 7. Tecnologias Utilizadas

| Camada | Tecnologias |
|---|---|
| **Linguagem** | ![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white) |
| **Gerenciador de Ambiente** | ![UV](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json) + `.python-version` |
| **Manipulação de Dados** | `pandas` · `numpy` |
| **Modelagem** | `scikit-learn` (KMeans / StandardScaler / adjusted_rand_score) · `scipy` (Ward linkage / cophenet) · `kneed` (cotovelo) |
| **Visualização** | `matplotlib` · `seaborn` · `plotly` |
| **Notebooks** | `jupyter` · `nbclient` (execução headless programática de 20 notebooks CLEAN + EXECUTED via AST parse) |
| **Qualidade de Código** | `ruff` (lint) · `black` (format) · `pytest` |
| **Deploy App (GitHub Pages)** | **HTML5 + CSS3 + Vanilla JavaScript** puro (NÃO precisa de backend · NÃO usa Vercel · NÃO usa Streamlit · classificação 100% cliente-side). |
| **Docs Executáveis / Relatórios** | `python-docx` (docx 6 perguntas) · `openpyxl` (planilha metas comerciais 2 abas) |
| **Controle de Versão** | `Git` + GitHub (commits planejados: 4 meses · julho/2026 a outubro/2026). |

---

## 8. Instalação e Uso

### Pré-requisitos
- Python 3.11+;
- [UV](https://docs.astral.sh/uv/getting-started/installation/) (recomendado) ou `pip` + `venv`;
- Bases `loja.csv` e `treino.csv` (1.115 lojas · 1.017.209 linhas) — copie manualmente para `data/01_raw/` (esses arquivos são ignorados pelo `.gitignore` por serem sensíveis).

### Passo a Passo
```bash
# 1) Clonar o repositório
git clone <URL DO SEU REPOSITÓRIO>
cd Retail-Store-Clustering

# 2) Criar ambiente virtual e instalar DEPENDÊNCIAS COMPLETAS do projeto
uv sync --extra dev
# OU (sem UV):
# python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 3) Colocar as bases originais em data/01_raw/ (fora do versionamento):
#    data/01_raw/loja.csv     (1.115 linhas cadastro)
#    data/01_raw/treino.csv   (1.017.209 linhas série)

# 4) Rodar os 20 notebooks CLEAN + EXECUTED (gera tudo do zero):
uv run python gerar_todos_os_notebooks.py

# 5) (Opcional) Rodar scripts de relatório individualmente:
#    Fase 4 cotovelo / silhueta
uv run python reports/fase4_kmeans_cotovelo_silhueta.py
#    Fase 6 plano de metas
uv run python reports/fase6_plano_metas_comercial.py
```

### 🎯 Classificar lojas novas SEM retreinar o modelo (deploy diário da Expansão)
```bash
# Opção A — standalone Python (APENAS 2 dependências: numpy + pandas)
uv pip install -r deploy/requirements_para_predicao.txt
uv run python deploy/predict.py deploy/exemplos/teste3_lote8_e_outlier.csv
# Saída: CSV em deploy/predicoes_YYYYMMDD_HHMMSS.csv com 1 linha por loja nova.

# Opção B — Aplicação Web Local (sem backend):
cd deploy
uv run python -m http.server 8765
# Abra no navegador:  http://localhost:8765/index.html
# Use a aba 📥 Lote → upload do CSV com as 6 features raw.
```

---

## 9. Publicação no GitHub Pages

A aplicação de classificação de lojas novas é **100% HTML/CSS/JS estático**. **NÃO precisa de backend Python. Não custa nada. Funciona até mesmo file:// local.**

### Passo a passo da publicação (1 vez só)

1. **Commite e push** tudo da pasta `deploy/` (inclua `index.html`, `estilo.css`, `classificador.js`, `params_modelo.json`, `perfil_medio_clusters_referencia.csv`, `assets/` e `exemplos/`) para o repositório no GitHub.

2. No repositório, clique em **Settings → Pages**.

3. Em **Source**, selecione **Deploy from a branch**, depois:
   - **Branch:** `main` (ou `master`);
   - **Folder:** `/ (root)` — SE você deixar o `deploy/index.html` no subdiretório, o GitHub Pages vai servir `deploy/` como subpasta, então a URL final seria `https://seu-user.github.io/Retail-Store-Clustering/deploy/index.html`.
   - **Alternativa recomendada para URL limpa:** copie **todo conteúdo de `deploy/` para a raiz do repositório** (e ajuste os paths da logo se necessário), deixando o `index.html` na raiz — a URL fica limpa `https://seu-user.github.io/Retail-Store-Clustering/`.

4. Clique **Save** · aguarde 1~3 minutos. GitHub Pages publicará e mostrará a URL final.

5. **Teste de fumaça após publicar:**
   - Abra a URL no Chrome/Safari/Edge;
   - Clique em **📥 Baixar modelo CSV (exemplo 3 lojas)** → baixa um CSV modelo;
   - Faça upload de qualquer um dos 3 CSVs de teste em `deploy/exemplos/` e confira se os batimentos da classificação aparecem (tabelão colorido com distância aos 5 centroides, tag outlier ciano se distância imputada, tag vermelha se outlier).

> ✅ **Confirmação arquitetural:** A equipe de TI NÃO precisa provisionar Vercel, Streamlit, Flask, FastAPI, servidor Windows, containers, banco de dados nem API. O app lê o CSV com `FileReader` e gera o CSV de download via `Blob` + `URL.createObjectURL`, tudo no navegador.

---

## 10. Licença (MIT)

Distribuído sob a licença MIT. Termos completos em:

- 📜 [LICENSE.md](./LICENSE.md) — versão legível e formatada.
- 📜 [LICENSE](./LICENSE) — cópia em texto puro compatível com GitHub License detection.

---

## 11. Contato

**Flávio Henrique Barbosa**

| Canal | Link / Contato |
|---|---|
| 👤 Nome | Flávio Henrique Barbosa |
| 💼 LinkedIn | [linkedin.com/in/flávio-henrique-barbosa-38465938](https://www.linkedin.com/in/flávio-henrique-barbosa-38465938) |
| 📧 Email | [flaviohenriquehb777@outlook.com](mailto:flaviohenriquehb777@outlook.com) |

---

<div align="center">
  <i>Projeto de Case — Ciência de Dados. Conforme combinado, commits/datas/histórico do git serão montados por pedido explícito do usuário (4 meses, julho/2026 a outubro/2026).</i>
</div>
