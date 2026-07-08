# -*- coding: utf-8 -*-
"""
Gera TODOS os notebooks do projeto a partir dos scripts .py em reports/.
Produz:
- notebooks/clean/*.ipynb     (sem execução, só células markdown + code vazias de output)
- notebooks/executed/*.ipynb  (rodados com nbclient, capturando prints e figuras em %matplotlib inline)
Cada notebook é dividido em células:
  - Markdown de cabeçalho com nome da fase e objetivo
  - Célula code "Imports e constantes" (lê imports no topo do script)
  - Células code subsequentes quebrando o script por marcadores
    (# === bloco ===), comentários de seção # ----, ou ~20 linhas.
"""
import re
import sys
import shutil
import ast
from pathlib import Path
from typing import List

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
NBCLEAN = ROOT / "notebooks" / "clean"
NBEXEC = ROOT / "notebooks" / "executed"
NBCLEAN.mkdir(parents=True, exist_ok=True)
NBEXEC.mkdir(parents=True, exist_ok=True)

KERNEL_NAME = "python3"


def _normaliza_paren_faltante_antes_do_parse(codigo: str) -> str:
    PARES = {"(": ")", "[": "]", "{": "}"}
    FECHA = {v: k for k, v in PARES.items()}
    linhas = codigo.splitlines()
    out = []
    pilha = []
    estado_str = None
    for ln in linhas:
        i = 0
        n = len(ln)
        in_line_comment = False
        while i < n:
            ch = ln[i]
            nxt3 = ln[i:i+3] if i+3 <= n else ""
            if not in_line_comment and estado_str in {None, '"""', "'''"}:
                if nxt3 in ('"""', "'''"):
                    if estado_str is None:
                        estado_str = nxt3
                    elif estado_str == nxt3:
                        estado_str = None
                    i += 3
                    continue
            if not in_line_comment and estado_str not in {'"""', "'''"}:
                if ch in ('"', "'") and estado_str is None:
                    estado_str = ch
                    i += 1
                    continue
                elif estado_str == ch:
                    estado_str = None
                    i += 1
                    continue
            if estado_str is None and ch == "#":
                in_line_comment = True
            if in_line_comment:
                i += 1
                continue
            if estado_str is None:
                if ch in PARES:
                    pilha.append(ch)
                elif ch in FECHA and pilha and pilha[-1] == FECHA[ch]:
                    pilha.pop()
            i += 1
        stripped = ln.strip()
        if pilha and estado_str is None and (not stripped or stripped.startswith("#")):
            fecha_str = "".join(PARES[p] for p in reversed(pilha))
            if fecha_str:
                if stripped.startswith("#"):
                    idx_hash = ln.index("#")
                    antes = ln[:idx_hash].rstrip()
                    depois = ln[idx_hash:]
                    nova = (antes + fecha_str + ("  " if antes else "") + depois).rstrip()
                else:
                    nova = fecha_str
                out.append(nova)
                pilha = []
                continue
        out.append(ln)
    return "\n".join(out) + ("\n" if codigo.endswith("\n") else "")


def _remover_declaracoes_de_root(texto: str) -> str:
    lines = texto.splitlines()
    try:
        tree = ast.parse(texto)
    except SyntaxError:
        tree = None
    apagar_ranges = set()
    blacklisted_names = {"ROOT","DATA_RAW","DATA_INT","DATA_PROC","MODELS","REPORTS","REPORTS_DIR","FIG","ENTREGA"}
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets_ok = False
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in blacklisted_names:
                        targets_ok = True
                        break
                    elif isinstance(tgt, (ast.Tuple, ast.List)):
                        for elt in getattr(tgt, "elts", []):
                            if isinstance(elt, ast.Name) and elt.id in blacklisted_names:
                                targets_ok = True
                                break
                if not targets_ok:
                    continue
                tem_path_file = any(
                    isinstance(child, ast.Name) and child.id == "__file__"
                    for child in ast.walk(node.value)
                )
                tem_root_rvalue = any(
                    isinstance(child, ast.Name) and child.id == "ROOT"
                    for child in ast.walk(node.value)
                )
                if tem_path_file or (targets_ok and tem_root_rvalue):
                    s = getattr(node, "lineno", None)
                    e = getattr(node, "end_lineno", None)
                    if isinstance(s, int) and isinstance(e, int):
                        for l in range(s, e + 1):
                            apagar_ranges.add(l)
            if isinstance(node, ast.Expr):
                val = node.value
                if isinstance(val, ast.Call):
                    func = val.func
                    if isinstance(func, ast.Attribute):
                        base = func.value
                        if isinstance(base, ast.Name) and base.id in blacklisted_names:
                            s = getattr(node, "lineno", None)
                            e = getattr(node, "end_lineno", None)
                            if isinstance(s, int) and isinstance(e, int):
                                for l in range(s, e + 1):
                                    apagar_ranges.add(l)
    novas = []
    for i, ln in enumerate(lines, 1):
        if i in apagar_ranges:
            continue
        novas.append(ln)
    txt2 = "\n".join(novas)
    txt2 = re.sub(r"^\s*(ROOT|REPORTS_DIR|DATA_PROC|DATA_RAW|FIG|MODELS|ENTREGA)\s*=\s*Path\(.*?\)\s*(\.resolve\(\))?(\.parent)+[\n\r]+",
                  "", txt2, flags=re.MULTILINE)
    return txt2


def _quebrar_em_celulas_code_por_secao(codigo: str, max_linhas_por_celula=55) -> List[str]:
    codigo_norm = _normaliza_paren_faltante_antes_do_parse(codigo)
    try:
        tree = ast.parse(codigo_norm)
        codigo_para_split = codigo_norm
    except SyntaxError:
        tree = None
        codigo_para_split = codigo

    linhas = codigo_para_split.splitlines()
    if tree is None or not tree.body:
        return [codigo_para_split.rstrip()] if codigo_para_split.strip() else []

    boundaries = []
    for node in tree.body:
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            end_line = node.lineno
            for child in ast.walk(node):
                child_end = getattr(child, "end_lineno", None)
                if isinstance(child_end, int):
                    end_line = max(end_line, child_end)
        boundaries.append(end_line)

    celulas = []
    inicio_cel = 1
    stmt_atual = 0
    n_stmts = len(boundaries)
    rx_cabecalho = re.compile(r"^#\s*=+|^#\s*-+")
    while stmt_atual < n_stmts:
        fim_cel = boundaries[stmt_atual]
        while stmt_atual + 1 < n_stmts and (boundaries[stmt_atual+1] - inicio_cel + 1) <= max_linhas_por_celula:
            tem_cabecalho = False
            for li in range(max(0, fim_cel - 1), min(len(linhas), boundaries[stmt_atual+1])):
                if rx_cabecalho.match(linhas[li].strip()):
                    tem_cabecalho = True
                    break
            if tem_cabecalho:
                break
            stmt_atual += 1
            fim_cel = boundaries[stmt_atual]
        cel_txt = "\n".join(linhas[inicio_cel - 1 : fim_cel]).rstrip()
        if cel_txt.strip():
            celulas.append(cel_txt)
        stmt_atual += 1
        inicio_cel = fim_cel + 1
    return celulas

# Ordem CRISP-DM mapeada. Cada item: (numero_ordem, arquivo.py em reports/, nome_amigavel, fase_crisp)
MAP = [
    # ---------- FASE 1 - ENTENDIMENTO DO NEGÓCIO ----------
    ("01", None, "Fase_1_Entendimento_do_Negocio", "FASE 1 — ENTENDIMENTO DO NEGÓCIO"),

    # ---------- FASE 2 - ENTENDIMENTO DOS DADOS ----------
    ("02", "fase2_data_understanding.py",            "Fase_2a_Exploracao_Geral_e_Dicionario",       "FASE 2a — ENTENDIMENTO DOS DADOS: exploração geral e dicionário"),
    ("03", "fase2_vendas_investigacao.py",            "Fase_2b_Vendas_Dias_Abertos_e_Inconsistencias","FASE 2b — ENTENDIMENTO DOS DADOS: séries de venda, dias abertos e linhas inconsistentes"),
    ("04", "fase2_investigacao_holes.py",             "Fase_2c_Reformadas_e_Buracos_de_Serie",       "FASE 2c — ENTENDIMENTO DOS DADOS: buracos nas séries e 180 lojas reformadas em 2014"),
    ("05", "fase2_cadastro_loja.py",                  "Fase_2d_Cadastro_de_Lojas_e_Concorrencia",    "FASE 2d — ENTENDIMENTO DOS DADOS: cadastro, tipo_loja, sortimento, distância concorrente"),
    ("06", "fase2_baselines_cruzamento.py",           "Fase_2e_Baselines_tipo_loja_vs_sortimento",   "FASE 2e — ENTENDIMENTO DOS DADOS: cruzamento tipo_loja × sortimento e perfis"),

    # ---------- FASE 3 - PREPARAÇÃO DOS DADOS ----------
    ("07", "fase3_preparacao.py",                     "Fase_3a_Blocos_de_Features_e_Filtro_Dias_Operando","FASE 3a — PREPARAÇÃO: filtro dias operando e construção 4 blocos de features"),
    ("08", "fase3_assimetrias.py",                    "Fase_3b_Assimetrias_Histogramas_e_log1p",     "FASE 3b — PREPARAÇÃO: assimetrias, histogramas top3 e transformação log1p"),
    ("09", "fase3_correlacao_e_selecao.py",           "Fase_3c_Correlacao_Spearman_e_Selecao_Features","FASE 3c — PREPARAÇÃO: matriz correlação Spearman e seleção final de features"),

    # ---------- FASE 4 - MODELAGEM ----------
    ("10", "fase4_kmeans_cotovelo_silhueta.py",      "Fase_4a_KMeans_k_2_10_Cotovelo_e_Silhueta",   "FASE 4a — MODELAGEM: KMeans k=2..10, inércia (cotovelo) e silhueta"),
    ("11", "fase4_hierarquico_dendrograma.py",       "Fase_4b_Hierarquico_Ward_e_Dendrograma",      "FASE 4b — MODELAGEM: clusterização hierárquica Ward e dendrograma truncado p=30"),
    ("12", "fase4_k_vs_criterios_sucesso.py",        "Fase_4c_k_2_3_5_8_vs_Criterios_Sucesso_Negocio","FASE 4c — MODELAGEM: k=2/3/5/8 contra 5 critérios de negócio"),
    ("13", "fase4_escolha_de_k_negocio.py",          "Fase_4d_Escolha_Final_de_k_5_por_Negocio",    "FASE 4d — MODELAGEM: escolha de k=5 por decisão de negócio"),
    ("14", "fase4_treino_final_k5.py",               "Fase_4e_Treino_Final_KMeans_k_5",             "FASE 4e — MODELAGEM: treino final KMeans k=5 e salvamento modelo/scaler"),

    # ---------- FASE 5 - AVALIAÇÃO ----------
    ("15", "fase5_alinhamento_caminho_a.py",         "Fase_5a_Alinhamento_Caminho_A_6cols_Professor","FASE 5a — AVALIAÇÃO: alinhamento ao Caminho A (feature set de 6 cols do professor) e nova estabilidade"),
    ("16", "fase5_estabilidade_baseline_perfil.py",  "Fase_5b_Estabilidade_Baseline_e_Perfil_Medio","FASE 5b — AVALIAÇÃO: 3 testes de estabilidade, baseline tipo_loja, perfil médio clusters"),
    ("17", "fase5_auditoria_ticket.py",              "Fase_5c_Auditoria_R2_Ticket_vs_Baseline",     "FASE 5c — AVALIAÇÃO: auditoria por que o ticket médio vence o baseline no feature set de 6 cols"),
    ("18", "fase5_tipo_b_onde_cairam.py",            "Fase_5d_Concentracao_Lojas_Tipo_b",           "FASE 5d — AVALIAÇÃO: onde caíram as 17 lojas tipo_b sem nunca ter visto a coluna"),
    ("19", "fase5_acoes_e_ganho_financeiro.py",      "Fase_5e_Acoes_por_Area_e_Ganho_Financeiro",   "FASE 5e — AVALIAÇÃO: tabela ações 3 áreas, premissas de $ e gap 3 cenários"),

    # ---------- FASE 6 - DEPLOY (entrega da planilha) ----------
    ("20", "fase6_plano_metas_comercial.py",         "Fase_6_Deploy_Plano_de_Metas_Comercial",      "FASE 6 — DEPLOY: construção da planilha de metas por loja entregue ao comercial"),
]


def _criar_celula_md_cabecalho(fase_label: str, script_nome: str | None):
    script_md = "" if script_nome is None else f"\n\n**Script de referência (reports/):** `{script_nome}`\n"
    md = (
        f"# {fase_label}\n\n"
        f"Projeto **Retail Store Clustering** · 1.115 lojas · seguindo CRISP-DM.\n"
        f"Ambiente: UV · Python 3.11 · SEED=42 em toda a modelagem.\n"
        f"{script_md}"
    )
    return nbf.v4.new_markdown_cell(md)


def _montar_notebook(numero_ordem: str, script_file: str | None, nome: str, fase_label: str):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": KERNEL_NAME,
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }
    cells = []
    # 1) Cabeçalho MD
    cells.append(_criar_celula_md_cabecalho(fase_label, script_file))
    # Caso especial: FASE 1 não tem script .py (era em conversa). Escrevemos um conteúdo resumido
    # com os 5 critérios de sucesso + 6 fases CRISP-DM.
    if script_file is None:
        md_fase1 = (
            "## Problema de negócio\n"
            "- Parar de comparar loja de shopping com posto de gasolina.\n"
            "- Criar grupos de lojas PARECIDAS de verdade.\n"
            "- Definir metas por cluster (comparando ao seu par) e não à rede toda.\n\n"
            "## Consumidores do resultado\n"
            "- Comercial (metas)\n"
            "- Trade Marketing (verba de promoção)\n"
            "- Supply (estoque e sortimento)\n"
            "- Expansão (qual formato abrir em um ponto novo)\n\n"
            "## 5 critérios de sucesso com testes concretos\n"
            "1. **Interpretável** (critério de gerente): cada cluster tem nome de 2-3 palavras que um gerente de rede repetiria numa reunião.\n"
            "2. **Acionável** (nenhum par com mesma ação): para todos os C(k,2)=10 pares de clusters, as ações em Comercial/Trade/Supply são diferentes.\n"
            "3. **Estável** (não depende de random_state):\n"
            "   - Shuffle linhas com seed 42 → ARI vs labels original ≥ 0,90.\n"
            "   - 10 random_states diferentes → matriz de ARI ≥ 0,85 (média ≥ 0,90).\n"
            "   - Bootstrap 80% de amostra, 10 rodadas → min ARI ≥ 0,60, média ≥ 0,75.\n"
            "4. **Melhor que baseline (tipo_loja / sortimento)**:\n"
            "   - Concordância ARI / NMI vs tipo_loja & sortimento.\n"
            "   - Pseudo R² ANOVA em pelo menos 4 métricas de negócio (VM, clientes, ticket, volatilidade, sazonalidade).\n"
            "5. **Quantificado em R$ (não em silhueta)**:\n"
            "   - Gap = (P75 do cluster − atual) × 365 dias.\n"
            "   - 3 cenários de recuperação do gap: 30% / 50% / 70%.\n\n"
            "## Regras contratuais do projeto\n"
            "- Tudo em português (BR).\n"
            "- SEED=42 em TUDO (random_state, shuffle, bootstrap splits).\n"
            "- Nenhuma feature de modelagem pode usar soma / contagem de dias (média, razão, proporção APENAS).\n"
            "- Baselines `tipo_loja` e `sortimento` NUNCA entram no X de modelagem.\n"
            "- StandardScaler SÓ na Fase 4, nunca na Fase 3.\n"
            "- KMeans final k=5 roda com `n_init=20`, `max_iter=500`.\n"
        )
        cells.append(nbf.v4.new_markdown_cell(md_fase1))
        # célula code dummy de imports para começar o notebook com pelo menos 1 code cell
        cells.append(nbf.v4.new_code_cell(
            "# Célula de ambiente: imports e constantes globais usadas em todas as fases seguintes\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "SEED = 42\n"
            "np.random.seed(SEED)\n"
            "\n"
            "ROOT = Path(os.getcwd()).resolve()\n"
            "DATA_RAW = ROOT / 'data' / '01_raw'\n"
            "DATA_PROC = ROOT / 'data' / '03_processed'\n"
            "MODELS = ROOT / 'models'\n"
            "REPORTS = ROOT / 'reports'\n"
            "FIG = REPORTS / 'figures'\n"
            "print('ROOT:', ROOT)\n"
        ))
        # salva CLEAN
        nb["cells"] = cells
        fname = f"{numero_ordem}_{nome}.ipynb"
        nbf.write(nb, NBCLEAN / fname, version=4)
        # executa o notebook e salva EXECUTED (execução vai funcionar só de ter prints)
        nb_exec = nbf.v4.new_notebook()
        nb_exec.metadata = nb.metadata
        nb_exec.cells = [c.copy() for c in cells]
        # executa
        client = NotebookClient(nb_exec, timeout=600, kernel_name=KERNEL_NAME)
        client.execute()
        nbf.write(nb_exec, NBEXEC / fname, version=4)
        print(f"OK Fase1: {fname}")
        return

    # --- tem script .py ---
    src = (REPORTS / script_file).read_text(encoding="utf-8")
    src = _remover_declaracoes_de_root(src)

    # Também fazemos o notebook executar SEMPRE da ROOT do projeto: adicionamos no topo
    # os imports e um os.chdir garantindo ROOT.
    preambulo = (
        "# ============================================================\n"
        "# Pré-configuração do notebook (RODA SEMPRE a partir da raiz)\n"
        "# ============================================================\n"
        "import os as _os\n"
        "from pathlib import Path as _Path\n"
        "\n"
        "_ROOT_CANDIDATOS = [\n"
        "    _Path(_os.getcwd()).resolve(),\n"
        "    _Path(_os.getcwd()).resolve().parent,\n"
        "    _Path(_os.environ.get('TRAE_ROOT', _os.getcwd())).resolve(),\n"
        "]\n"
        "ROOT = next((p for p in _ROOT_CANDIDATOS if (p / 'pyproject.toml').exists()), _ROOT_CANDIDATOS[0])\n"
        "_os.chdir(ROOT)\n"
        "print('ROOT =', ROOT)\n"
        "\n"
        "# Caminhos padrão usados em todos os scripts\n"
        "DATA_RAW = ROOT / 'data' / '01_raw'\n"
        "DATA_INT = ROOT / 'data' / '02_interim'\n"
        "DATA_PROC = ROOT / 'data' / '03_processed'\n"
        "MODELS = ROOT / 'models'\n"
        "REPORTS_DIR = ROOT / 'reports'\n"
        "FIG = REPORTS_DIR / 'figures'\n"
        "ENTREGA = REPORTS_DIR / 'entrega_fase6'\n"
        "ENTREGA.mkdir(parents=True, exist_ok=True)\n"
        "FIG.mkdir(parents=True, exist_ok=True)\n"
        "MODELS.mkdir(parents=True, exist_ok=True)\n"
        "DATA_PROC.mkdir(parents=True, exist_ok=True)\n\n"
    )
    src = preambulo + "\n" + src
    # extrai o comentário de cabeçalho do script se houver (topo """ docstring ou primeiros # comentários)
    blocos_code = _quebrar_em_celulas_code_por_secao(src, max_linhas_por_celula=55)
    # primeira célula: extrair docstring do topo como markdown
    doc_md = None
    m = re.match(r"^\s*#.*?\n?\s*(\"\"\"|''')([\s\S]*?)\1", src)
    if m:
        doc_md = m.group(2).strip()
        doc_md = doc_md.replace("\n", "\n\n")
        cells.append(nbf.v4.new_markdown_cell(
            f"## Descrição do script de referência\n\n{doc_md}"
        ))
    # transforma cada bloco code em célula code
    for i, bloco in enumerate(blocos_code):
        # Adiciona marcador MD de cabeçalho de seção se o bloco começar com # === ou # ---
        primeira_linha = next((ln.strip() for ln in bloco.splitlines() if ln.strip()), "")
        if re.match(r"^#\s*=+", primeira_linha) or re.match(r"^#\s*-+", primeira_linha):
            titulo = primeira_linha.strip("#= -").strip() or f"Seção {i+1}"
            cells.append(nbf.v4.new_markdown_cell(f"### {titulo}"))
        # %matplotlib inline em todo notebook para capturar figuras
        bloco_patch = bloco
        if (
            "import matplotlib" in bloco or "import matplotlib.pyplot" in bloco
            or "from matplotlib" in bloco or "import seaborn" in bloco
        ) and "plt" in bloco:
            # não injeta backend; apenas adicionamos cell magic %matplotlib inline em uma célula ANTES dessa
            ultimas = cells[-3:] if len(cells) >= 3 else list(cells)
            fontes_ultimas = []
            for cel in ultimas:
                try:
                    src = cel.get("source", "")
                except Exception:
                    src = ""
                fontes_ultimas.append(str(src))
            if not any("get_ipython()" in s or "%matplotlib" in s for s in fontes_ultimas):
                pass
        cells.append(nbf.v4.new_code_cell(bloco_patch))

    nb["cells"] = cells
    # SALVA CLEAN (sem executar)
    fname = f"{numero_ordem}_{nome}.ipynb"
    nbf.write(nb, NBCLEAN / fname, version=4)

    # SALVA EXECUTED (roda o notebook com nbclient e captura outputs)
    nb_exec = nbf.v4.new_notebook()
    nb_exec.metadata = nb.metadata
    nb_exec.cells = [nbf.v4.new_code_cell("%matplotlib inline\nimport matplotlib; matplotlib.rcParams['figure.dpi']=72")]
    for c in cells:
        nb_exec.cells.append(c.copy())
    client = NotebookClient(nb_exec, timeout=1800, kernel_name=KERNEL_NAME,
                            allow_errors=False, force_raise_errors=False)
    try:
        client.execute()
        print(f"OK EXEC (sem erro): {fname}")
    except Exception as e:
        print(f"!! WARN {fname}: exceção durante execução do notebook: {e}")
    nbf.write(nb_exec, NBEXEC / fname, version=4)


def main():
    # Clean slate: apagar todos os .ipynb antigos antes de gerar novos para não ficar com arquivos duplicados de numeração antiga
    for pasta in [NBCLEAN, NBEXEC]:
        for f in pasta.glob("*.ipynb"):
            f.unlink()
    for (n, pyfile, nome, fase) in MAP:
        try:
            _montar_notebook(n, pyfile, nome, fase)
        except Exception as e:
            print(f"!! ERRO notebook {n}_{nome}: {type(e).__name__}: {e}")

    print()
    print("=== RESUMO notebooks gerados ===")
    for pasta in [NBCLEAN, NBEXEC]:
        qtd = len(list(pasta.glob("*.ipynb")))
        print(f"pasta {pasta.relative_to(ROOT)}: {qtd} arquivos .ipynb")
        for arq in sorted(pasta.glob("*.ipynb")):
            print(f"  · {arq.name}")


if __name__ == "__main__":
    main()
