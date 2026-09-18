# -*- coding: utf-8 -*-
"""
FASE 5 — Critérios 1 (Interpretável) e 2 (Acionável).
Entrega 1: Tabela de ações por cluster (1 linha por cluster, 3 áreas).
Entrega 2: Cálculo teórico de ganho $ (meta condizente com realidade → fecha gap até mediana do seu cluster).
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"

tab = pd.read_csv(DATA_PROC / "tabela_lojas_com_cluster_final_v2_prof6.csv")
assert len(tab) == 1115

# =============== NOMES DOS CLUSTERS ===============
# Escolhidos a partir dos números do perfil v2 (prof6)
NOMES = {
    1: "Alto Fluxo / Ponto de Passagem",
    4: "Standard / Classe Média",
    0: "Ticket Premium / Mix Caro",
    2: "Promo-Dependent",
    3: "Sazonal Raro / Temporada",
}
tab["nome_cluster"] = tab["cluster_final_v2"].map(NOMES)
ordem_numerica = [1, 4, 0, 2, 3]

# =============== ESTATÍSTICAS PARA DAR NOME E AÇÃO ===============
METRICAS_PERFIL = [
    ("venda_media_dia", "VM_dia_med"),
    ("clientes_medio_dia", "Cli_dia_med"),
    ("ticket_medio", "Ticket_med"),
    ("razao_promo", "Raz_promo_med"),
    ("cv_volatilidade_diaria", "Vol_med"),
    ("cv_sazonalidade_mensal", "Saz_med"),
    ("distancia_concorrente", "Dist_concorr_med"),
    ("promo_continua", "Tx_promocont_med"),
    ("tx_domingos_abertos", "Tx_domingo_med"),
]

def agg(g):
    out = {"n_lojas": g["loja"].nunique()}
    for col, nome in METRICAS_PERFIL:
        if col == "distancia_concorrente":
            out[nome] = float(g[col].median())
        elif col in ("promo_continua", "tx_domingos_abertos"):
            out[nome] = float(g[col].mean() * 100)
        elif col == "razao_promo":
            out[nome] = float((g[col].mean() - 1) * 100)  # uplift %
        else:
            out[nome] = float(g[col].mean())
    # Mediana de venda do cluster (referência para o cálculo de $ abaixo)
    out["VM_dia_mediana_cluster"] = float(g["venda_media_dia"].median())
    return pd.Series(out)

perfil = tab.groupby("cluster_final_v2").apply(agg).reset_index()
perfil["nome_cluster"] = perfil["cluster_final_v2"].map(NOMES)
perfil["%_rede"] = (perfil["n_lojas"] / len(tab) * 100).round(1)
perfil = perfil.set_index("cluster_final_v2").loc[ordem_numerica].reset_index()

# =============== TABELA 1: AÇÕES POR CLUSTER (1 linha por cluster) ===============
# Regra de escrita do texto: NOME DO CLUSTER → justificar a ação nos números do perfil.
def acao_comercial_meta(row):
    vm = row["VM_dia_med"]
    uplift = row["Raz_promo_med"]
    pct_dom = row["Tx_domingo_med"]
    cid = row["cluster_final_v2"]
    if cid == 1:
        return (f"Meta agressiva acima da média (+15%): {vm:,.0f} R$/dia referência. "
                f"Motivo: 2º maior VM, fluxo alto (1.212 cli/dia), perto concorrente (270m med). "
                f"Loja de passagem bate meta com operação diária.")
    elif cid == 4:
        return (f"Meta moderada (+5%): {vm:,.0f} R$/dia. "
                f"Motivo: 357 lojas standard, vol baixa (0,24 CV) — resultado previsível. "
                f"Acompanha inflação + crescimento da rede.")
    elif cid == 0:
        return (f"Meta de margem não só volume (+8%): {vm:,.0f} R$/dia. "
                f"Motivo: Ticket R$12,03 (o MAIOR), dist concorr 6.360m = pouco preço. "
                f"Foco em mix premium e upsell.")
    elif cid == 2:
        return (f"Meta condicionada a promo (+10% se campanha atender): {vm:,.0f} R$/dia. "
                f"Motivo: uplift de promo +{uplift:.0f}% (2× a média da rede). "
                f"Sem promo a venda cai 40% — meta sem evento de promo é irreal.")
    elif cid == 3:
        return (f"Meta dinâmica por temporada (+3% anual): {vm:,.0f} R$/dia anual médio. "
                f"Motivo: Saz CV={row['Saz_med']:.2f} (2,3× a rede). "
                f"Acompanha picos de temporada — reunião bimestral de revisão, não anual.")

def acao_trade_verba(row):
    cid = row["cluster_final_v2"]
    if cid == 1:
        return "Verba de CONTRA-ATAQUE a concorrente (40% do budget). Dist média de 270m. Promoção pontual de preço em itens de preço conhecido (PPC) para não perder cliente pro vizinho."
    elif cid == 4:
        return "Verba de MANUTENÇÃO / frequência (20%). Cluster standard com 32% da rede. Verba pequena, campanha massiva genérica da rede."
    elif cid == 0:
        return "Verba de BRANDING / mix premium (15%). R$12 ticket alto, distante concorrente. Não faz desconto de preço. Faz evento de lançamento de produto premium, experiência em loja."
    elif cid == 2:
        return "TODA a verba que sobrar (~25%). Uplift de promo +67% é o ROI mais alto da rede. Toda liberação de verba começa aqui. Campanha de desconto agressivo com fabricante, 2ª unidade, leve+3 pague 2 etc."
    elif cid == 3:
        return "Verba SAZONAL programada (~10%). Não é verba mensal. Budget pré-aprovado para o pico de temporada correspondente (férias, verão, datas regionais). Fora da temporada: verba zero."

def acao_supply(row):
    cid = row["cluster_final_v2"]
    if cid == 1:
        return "Alto giro, sortimento enxuto, reposição diária. Ponto crítico: ruptura de estoque mata venda pois cliente passa e vai pro vizinho (270m). Stock de segurança 2× a média no PPC."
    elif cid == 4:
        return "Mix padrão da rede. Política global de estoque, reposição semanal. 357 lojas permitem escala de compra."
    elif cid == 0:
        return "Mix expandido de categorias premium e lançamentos. SKUs de maior valor agregado. Stock de segurança maior (dist concorr 6,3 km: cliente não tem alternativa se faltar). Menor giro."
    elif cid == 2:
        return "Política de estoque VOLÁTIL. Planejar reposição extra semana de promo. Sortimento de oferta. Precisa de planejamento integrado com Trade Calendar da semana."
    elif cid == 3:
        return "Saiba quando é a temporada dessa loja — estoque programado 30 dias antes do pico, 60% de giro só no pico. Data-driven por loja, não por mês. Fora da temporada, reduz sortimento para não morrer em estoque encalhado."

perfil["Comercial · Meta"] = perfil.apply(acao_comercial_meta, axis=1)
perfil["Trade Marketing · Verba Promo"] = perfil.apply(acao_trade_verba, axis=1)
perfil["Supply · Estoque e Sortimento"] = perfil.apply(acao_supply, axis=1)

# ========== IMPRIMIR TABELA 1 ==========
print("=" * 140)
print("TABELA AÇÕES POR CLUSTER (1 linha por cluster)")
print("=" * 140)
cols_out = ["cluster_final_v2", "nome_cluster", "n_lojas", "%_rede",
            "Comercial · Meta", "Trade Marketing · Verba Promo", "Supply · Estoque e Sortimento"]
for _, r in perfil[cols_out].iterrows():
    cid = int(r["cluster_final_v2"])
    print(f"\n{'-' * 140}")
    print(f"CLUSTER C{cid:02d} — {r['nome_cluster']}  (n={int(r['n_lojas'])} lojas · {r['%_rede']}% da rede)")
    print(f"{'-' * 140}")
    for area in ["Comercial · Meta", "Trade Marketing · Verba Promo", "Supply · Estoque e Sortimento"]:
        print(f"\n  ▸ {area}:")
        txt = str(r[area])
        # Quebra de linha a cada 120 chars
        words = txt.split()
        line = ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 > 120:
                lines.append(line)
                line = w
            else:
                line = w if line == "" else f"{line} {w}"
        if line:
            lines.append(line)
        for ln in lines:
            print(f"    {ln}")

# =============== TABELA 2: CÁLCULO TEÓRICO DE $ GANHO ===============
# Premissas (documentadas na saída):
# P1. Referência = VENDA MÉDIA DIÁRIA da loja (valor em data/03_processed tabela final).
# P2. Lojas ABAIXO DA MEDIANA DE VENDA do seu próprio cluster = as que recebem meta condizente.
# P3. Gap = (Mediana do seu cluster) − (VM_dia da loja). Nunca negativo.
# P4. Recuperação parcial (não 100% do gap, isso é irreal): 30% conservador, 50% moderado, 70% otimista.
# P5. Período = 365 dias de operação por ano. Ignora fechamentos para não inflar números pequenos.
# P6. Não inflaciona: assume VM constante em termos reais. Focado só no fechamento do gap intra-cluster.

rows_ganho = []
for cid in ordem_numerica:
    sub = tab[tab["cluster_final_v2"] == cid].copy()
    n = len(sub)
    vm_median_cluster = float(sub["venda_media_dia"].median())
    abaixo = sub[sub["venda_media_dia"] < vm_median_cluster]
    n_abaixo = len(abaixo)
    pct_abaixo = n_abaixo / n * 100
    # Gap por loja (R$ / dia)
    gap_unitario = vm_median_cluster - abaixo["venda_media_dia"].values
    gap_unitario_medio = float(gap_unitario.mean()) if len(gap_unitario) > 0 else 0
    gap_total_ano_RS = float((gap_unitario * 365).sum())  # 100% do gap
    row = {
        "Cluster": f"C{cid:02d} · {NOMES[cid]}",
        "n_lojas": n,
        "VM_dia_mediana_cluster_R$": vm_median_cluster,
        "n_lojas_abaixo_mediana_cluster": n_abaixo,
        "%_abaixo_mediana_cluster": pct_abaixo,
        "gap_unitario_medio_R$_por_dia": gap_unitario_medio,
        "GAP_100_RECUPERADO_R$_por_ano_total": gap_total_ano_RS,
        "CENARIO_C_30_RS_ano": gap_total_ano_RS * 0.30,
        "CENARIO_M_50_RS_ano": gap_total_ano_RS * 0.50,
        "CENARIO_O_70_RS_ano": gap_total_ano_RS * 0.70,
    }
    rows_ganho.append(row)

tbl_ganho = pd.DataFrame(rows_ganho)
total_linha = {
    "Cluster": "TOTAL REDE",
    "n_lojas": tbl_ganho["n_lojas"].sum(),
    "VM_dia_mediana_cluster_R$": None,
    "n_lojas_abaixo_mediana_cluster": tbl_ganho["n_lojas_abaixo_mediana_cluster"].sum(),
    "%_abaixo_mediana_cluster": tbl_ganho["n_lojas_abaixo_mediana_cluster"].sum() / tbl_ganho["n_lojas"].sum() * 100,
    "gap_unitario_medio_R$_por_dia": tbl_ganho["gap_unitario_medio_R$_por_dia"].mean(),
    "GAP_100_RECUPERADO_R$_por_ano_total": tbl_ganho["GAP_100_RECUPERADO_R$_por_ano_total"].sum(),
    "CENARIO_C_30_RS_ano": tbl_ganho["CENARIO_C_30_RS_ano"].sum(),
    "CENARIO_M_50_RS_ano": tbl_ganho["CENARIO_M_50_RS_ano"].sum(),
    "CENARIO_O_70_RS_ano": tbl_ganho["CENARIO_O_70_RS_ano"].sum(),
}
tbl_ganho = pd.concat([tbl_ganho, pd.DataFrame([total_linha])], ignore_index=True)

def money(v):
    return "—" if v is None or pd.isna(v) else f"R$ {v:>15,.0f}"

def pct(v):
    return f"{v:>5.1f}%" if v is not None and not pd.isna(v) else "—"

print("\n\n" + "=" * 160)
print("CÁLCULO TEÓRICO DE GANHO COM METAS CONDIZENTES.")
print("=" * 160)
print("\nPREMISSAS (6):")
print("  P1. Referência de venda = 'venda_media_dia' (média só sobre dias em que a loja operou; não usa somas).")
print("  P2. Definição de 'loja mal performante' = abaixo da MEDIANA de venda diária DO SEU PRÓPRIO CLUSTER.")
print("      (não comparada à rede toda).")
print("  P3. Gap unitário = (Mediana do cluster) − (VM diária da loja). Gap truncado em ≥ 0.")
print("  P4. NÃO recuperamos 100% do gap (fantasia). 3 cenários: CONSERVADOR 30%, MODERADO 50%, OTIMISTA 70%.")
print("  P5. Período de projeção = 365 dias de operação / ano (não infla por fechamentos, números ficam menores).")
print("  P6. Cenário real (sem crescimento macro, sem inflação). O ganho vem EXCLUSIVAMENTE do fechamento do gap")
print("      intra-cluster.")
print()

cabecalhos = ["Cluster", "n_lojas", "VM med. clus (R$/dia)", "n Abaixo", "% abaixo",
              "Gap médio/dia (R$)", "Gap total ano (100%)",
              "Cenário C (30%)", "Cenário M (50%)", "Cenário O (70%)"]
print(" | ".join(f"{h:>30s}" for h in cabecalhos))
print("-" * 335)
for _, r in tbl_ganho.iterrows():
    cols = [
        f"{r['Cluster']:<40s}",
        f"{r['n_lojas']:>6.0f}",
        money(r["VM_dia_mediana_cluster_R$"]).replace("R$ ",""),
        f"{r['n_lojas_abaixo_mediana_cluster']:>8.0f}",
        pct(r["%_abaixo_mediana_cluster"]),
        money(r["gap_unitario_medio_R$_por_dia"]).replace("R$ ",""),
        money(r["GAP_100_RECUPERADO_R$_por_ano_total"]),
        money(r["CENARIO_C_30_RS_ano"]),
        money(r["CENARIO_M_50_RS_ano"]),
        money(r["CENARIO_O_70_RS_ano"]),
    ]
    print(" | ".join(c if c.endswith(")") or c.startswith("TOTAL") or len(c) <= 8 else f"{c:>30}" for c in cols))

# Salvando tabelas
cols_t1 = cols_out + ["VM_dia_med", "Cli_dia_med", "Ticket_med", "Raz_promo_med",
                      "Vol_med", "Saz_med", "Dist_concorr_med", "Tx_promocont_med", "Tx_domingo_med"]
perfil[cols_t1].to_csv(DATA_PROC / "fase5_tabela_acoes_por_cluster.csv", index=False, float_format="%.2f")
tbl_ganho.to_csv(DATA_PROC / "fase5_ganho_financeiro_por_cluster.csv", index=False, float_format="%.2f")
print("\n> Salvo: fase5_tabela_acoes_por_cluster.csv")
print("> Salvo: fase5_ganho_financeiro_por_cluster.csv")
