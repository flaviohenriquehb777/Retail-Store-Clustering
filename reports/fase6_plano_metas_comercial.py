# -*- coding: utf-8 -*-
"""
FASE 6 - DEPLOYMENT
Entrega 1: Plano de metas para o Comercial.
Regra de negócio:
- Meta percentual de cada loja sai da POSIÇÃO DENTRO DO SEU CLUSTER (não % linear).
- Benchmark do grupo = P75 da venda_media_dia dentro do cluster (referência de top-quartil).
- Gap% = (Benchmark − atual) / atual.
- Meta% pura = base 5% + participação proporcional do gap, com teto.
- Prioridade: Q1 = ação, Q2-Q3 = acompanhamento, Q4 = manutenção.
- Depois: checagem de MÉDIA GLOBAL DAS METAS tem que dar ~+5% da diretoria.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
REPORTS_DIR = ROOT / "reports"
ENTREGA = ROOT / "reports" / "entrega_fase6"
ENTREGA.mkdir(exist_ok=True, parents=True)

NOMES_CLUSTER = {
    1: "C01 · Alto Fluxo / Ponto de Passagem",
    4: "C04 · Standard / Classe Média",
    0: "C00 · Ticket Premium / Mix Caro",
    2: "C02 · Promo-Dependent",
    3: "C03 · Sazonal Raro / Temporada",
}

tab = pd.read_csv(DATA_PROC / "tabela_lojas_com_cluster_final_v2_prof6.csv")
assert len(tab) == 1115, len(tab)

df = tab[["loja", "cluster_final_v2", "tipo_loja", "sortimento",
          "venda_media_dia", "clientes_medio_dia", "ticket_medio"]].copy()
df = df.rename(columns={"cluster_final_v2": "cluster"})
df["nome_cluster"] = df["cluster"].map(NOMES_CLUSTER)

# ============================ PASSO 1: BENCHMARK por cluster ============================
# Benchmark do grupo = P75 da venda media diaria DENTRO DO CLUSTER (loja top 25% do grupo).
bench = df.groupby("cluster")["venda_media_dia"].agg(
    benchmark_grupo=lambda g: float(np.quantile(g, 0.75)),
    mediana_grupo="median",
    p25_grupo=lambda g: float(np.quantile(g, 0.25)),
    media_grupo="mean",
    n_lojas="count",
).reset_index()
df = df.merge(bench, on="cluster", how="left")
df["gap_pct"] = (df["benchmark_grupo"] - df["venda_media_dia"]) / df["venda_media_dia"]

# ============================ PASSO 2: RANK DENTRO DO CLUSTER ============================
df["rank_no_cluster"] = df.groupby("cluster")["venda_media_dia"].rank(ascending=False, method="min")
df["percentil_cluster"] = df.groupby("cluster")["venda_media_dia"].rank(pct=True, ascending=True)  # 0=ruim,1=bom

# ============================ PASSO 3: REGRA DE META POR LOJA (AJUSTE RELATIVO) ============================
# Fórmula meta%:
# Meta base = 5% (compromisso diretoria)
# Ajuste de performance: se percentil no cluster = P10 (pior), recebe +XX% extra;
# se é P90 (melhor), recebe 0 extra (meta de manutenção).
# Extra = (1 - percentil_cluster) * multiplicador_pondera * gap_pct * k_ganho.
# TETO = max 15% de meta (ninguém recebe mais de +15%, irreal).
# PISO = min 2% (ninguém recebe menos de +2%, mantém crescimento da rede).
mult_extra = 0.55  # metade do gap em % como contribuidor de meta extra (o resto fica como melhoria possível futura)
df["ajuste_extra"] = (1 - df["percentil_cluster"]) * df["gap_pct"] * mult_extra
df["ajuste_extra"] = df["ajuste_extra"].clip(lower=0, upper=0.10)   # max +10% extra além da base
df["meta_pct_bruta"] = 0.05 + df["ajuste_extra"]
# Teto e piso
df["meta_pct_bruta"] = df["meta_pct_bruta"].clip(lower=0.02, upper=0.15)

# ============================ PASSO 4: CHECAGEM DA MÉDIA GLOBAL ============================
# Depois do cálculo bruto: Média ponderada por venda diária tem que = 5% aproximadamente (compromisso).
# Para ser realista, a média das metas % por loja NÃO ponderada por tamanho é o que o comercial recebe.
# Mas a diretoria prometeu crescimento total da rede = +5%, então vamos checar a média PONDERADA por venda.
media_simples = df["meta_pct_bruta"].mean() * 100
media_ponderada_vm = (df["meta_pct_bruta"] * df["venda_media_dia"]).sum() / df["venda_media_dia"].sum() * 100
print("===== CHECAGEM ANTES DO AJUSTE =====")
print(f"Média SIMPLES das metas % (1 por loja)  = +{media_simples:.2f}%")
print(f"Média PONDERADA por venda da rede (meta total crescimento) = +{media_ponderada_vm:.2f}%")
print(f"Compromisso diretoria = +5,00%")
print(f"Delta da média ponderada vs meta: {media_ponderada_vm - 5.00:+.2f} pts")
print()

# Alinhamento linear para que a média ponderada fique EXATAMENTE em 5.0%.
# alpha deslocamento para bater exato
objetivo = 0.05
ponderacao = df["venda_media_dia"].values / df["venda_media_dia"].sum()
# Resolvendo para alpha:  sum( (x + alpha) * w ) = objetivo
# => alpha = (objetivo - sum(x*w)) / sum(w) ; sum(w)=1
delta_linear = (objetivo - (df["meta_pct_bruta"].values * ponderacao).sum()) / 1.0
df["meta_pct_final"] = (df["meta_pct_bruta"] + delta_linear).round(6)
# Reaplicar tetos e pisos depois do deslocamento (se alguém passou)
df["meta_pct_final"] = df["meta_pct_final"].clip(lower=0.015, upper=0.16)
# Re-ajustar uma segunda vez (clipping pode ter tirado do centro)
delta2 = (objetivo - (df["meta_pct_final"].values * ponderacao).sum()) / 1.0
df["meta_pct_final"] = (df["meta_pct_final"] + delta2).round(6)

# Valores monetários
df["venda_meta_diaria_R$"] = df["venda_media_dia"] * (1 + df["meta_pct_final"])
df["crescimento_esperado_anual_R$"] = df["venda_meta_diaria_R$"] * 365 - df["venda_media_dia"] * 365

# ============================ PASSO 5: MARCAÇÃO DE PRIORIDADE ============================
# Regra:
# P1 = AÇÃO URGENTE: abaixo do P25 do SEU cluster (quadrimestre pior do grupo).
# P2 = AÇÃO PLANEJADA: entre P25 e P50, e gap_pct > 20%.
# P3 = ACOMPANHAMENTO: entre P25 e P75 do grupo (pessoa comum do meio).
# P4 = MANUTENÇÃO / REFERÊNCIA: acima do P75 do SEU cluster.
def prioridade(row):
    p = row["percentil_cluster"]
    gap = row["gap_pct"]
    if p <= 0.25:
        return "P1 · AÇÃO URGENTE"
    elif p <= 0.5 and gap > 0.20:
        return "P2 · AÇÃO PLANEJADA"
    elif p >= 0.75:
        return "P4 · MANUTENÇÃO / REFERÊNCIA"
    else:
        return "P3 · ACOMPANHAMENTO"

df["prioridade"] = df.apply(prioridade, axis=1)

# ============================ PASSO 6: ESTATÍSTICAS FINAIS ============================
print("===== CHECAGEM DEPOIS DO AJUSTE LINEAR (exato em 5% ponderado) =====")
m_simples_f = df["meta_pct_final"].mean() * 100
m_pond_f = (df["meta_pct_final"] * df["venda_media_dia"]).sum() / df["venda_media_dia"].sum() * 100
print(f"Média SIMPLES das metas % (por loja): +{m_simples_f:.2f}%")
print(f"Média PONDERADA por venda (total da rede): +{m_pond_f:.4f}%")
print(f"Compromisso diretoria: +5,0000%")
print(f"Delta final do compromisso (ponderado): {m_pond_f - 5.0:+.4f} pts")
print()

print("===== DISTRIBUIÇÃO DAS METAS % (histograma por faixa) =====")
bins = [0, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.10, 1.00]
labels_b = ["< +2,5%", "+2,5% a +3,5%", "+3,5% a +4,5%", "+4,5% a +5,5%",
            "+5,5% a +6,5%", "+6,5% a +7,5%", "+7,5% a +8,5%", "+8,5% a +10%", "> +10%"]
df["faixa_meta"] = pd.cut(df["meta_pct_final"], bins=bins, labels=labels_b, include_lowest=True, right=True)
hist = (df.groupby("faixa_meta", observed=False)["loja"]
          .count().rename("n_lojas").to_frame())
hist["%_da_rede"] = (hist["n_lojas"] / len(df) * 100).round(1)
print(hist.to_string())
print()

print("===== ESTATÍSTICAS DAS METAS % POR CLUSTER =====")
c_agg = df.groupby("cluster")["meta_pct_final"].agg(
    n_lojas="count", media="mean", P25=lambda s: s.quantile(0.25),
    P50="median", P75=lambda s: s.quantile(0.75), min="min", max="max"
).round(4) * 100
c_agg.insert(0, "nome_cluster", [NOMES_CLUSTER[c] for c in c_agg.index])
print(c_agg.to_string())
print()

print("===== DISTRIBUIÇÃO DE PRIORIDADES =====")
pri = df.groupby("prioridade")["loja"].count().rename("n_lojas").to_frame()
pri["%_da_rede"] = (pri["n_lojas"] / len(df) * 100).round(1)
pri["crescimento_esperado_anual_total_R$"] = df.groupby("prioridade")["crescimento_esperado_anual_R$"].sum().round(0)
pri = pri.sort_index()
print(pri.to_string())
print()

print("===== 10 EXEMPLOS (primeiras 10 lojas ordenadas por gap decrescente) =====")
cols_show = ["loja", "cluster", "nome_cluster", "venda_media_dia", "benchmark_grupo",
             "gap_pct", "percentil_cluster", "meta_pct_final", "prioridade"]
ex = df.sort_values("gap_pct", ascending=False).head(10)[cols_show].copy()
ex["gap_pct"] = (ex["gap_pct"] * 100).round(1)
ex["percentil_cluster"] = (ex["percentil_cluster"] * 100).round(1)
ex["meta_pct_final"] = (ex["meta_pct_final"] * 100).round(2)
print(ex.to_string(index=False, float_format=lambda v: f"{v:,.0f}" if isinstance(v, (int,float)) and v > 1000 else f"{v}"))

# ============================ SALVAR ENTREGA ============================
# Colunas limpas para o comercial
colunas_final = [
    "loja",
    "cluster",
    "nome_cluster",
    "tipo_loja",
    "sortimento",
    "venda_media_dia",
    "mediana_grupo",
    "benchmark_grupo",    # P75 do grupo
    "p25_grupo",
    "gap_pct",
    "rank_no_cluster",
    "percentil_cluster",
    "meta_pct_final",
    "venda_meta_diaria_R$",
    "crescimento_esperado_anual_R$",
    "prioridade",
]
out = df[colunas_final].sort_values(["cluster", "venda_media_dia"], ascending=[True, False]).reset_index(drop=True)
out.to_csv(ENTREGA / "fase6_plano_metas_comercial_por_loja.csv", index=False, float_format="%.5f")

# -------- Aba 1 "Metas por Loja" (colunas alinhadas ao padrão pedido) --------
aba1 = out.copy()
aba1 = aba1.rename(columns={
    "loja": "Loja",
    "cluster": "Cluster ID",
    "nome_cluster": "Perfil Comercial",
    "venda_media_dia": "Venda Média Diária (R$)",
    "benchmark_grupo": "Benchmark Grupo (R$)",
    "gap_pct": "Gap Relativo (%)",
    "meta_pct_final": "Meta de Crescimento (%)",
    "prioridade": "Prioridade de Gestão",
})
colunas_aba1 = ["Loja","Cluster ID","Perfil Comercial","Venda Média Diária (R$)",
                "mediana_grupo","Benchmark Grupo (R$)","p25_grupo","Gap Relativo (%)",
                "rank_no_cluster","percentil_cluster","Meta de Crescimento (%)",
                "venda_meta_diaria_R$","crescimento_esperado_anual_R$","Prioridade de Gestão"]
aba1 = aba1[colunas_aba1]

# -------- Aba 2 "Resumo Executivo" --------
# Colunas como na planilha do professor, 1 linha por cluster.
re = df.groupby(["cluster","nome_cluster"]).agg(**{
    "Cluster ID": ("cluster","first"),
    "Perfil Comercial": ("nome_cluster","first"),
    "Nº de Lojas": ("loja","count"),
    "% da Rede": ("loja", lambda s: len(s) / len(df)),
    "Venda Média (R$)": ("venda_media_dia","mean"),
    "Benchmark Mediana (R$)": ("mediana_grupo","first"),
    "Benchmark P75 (R$)": ("benchmark_grupo","first"),
    "Meta Média (%)": ("meta_pct_final","mean"),
    "Lojas para Ação (P1+P2)": ("prioridade",
        lambda s: int(s.isin(["P1 · AÇÃO URGENTE","P2 · AÇÃO PLANEJADA"]).sum())),
    "Crescimento Anual Total (R$)": ("crescimento_esperado_anual_R$","sum"),
}).reset_index(drop=True).sort_values("Cluster ID").reset_index(drop=True)
# Linha TOTAL DA REDE no final
total_row = pd.DataFrame([{
    "Cluster ID": 99,
    "Perfil Comercial": "TOTAL REDE",
    "Nº de Lojas": len(df),
    "% da Rede": 1.0,
    "Venda Média (R$)": df["venda_media_dia"].mean(),
    "Benchmark Mediana (R$)": None,
    "Benchmark P75 (R$)": None,
    "Meta Média (%)": (df["meta_pct_final"]*df["venda_media_dia"]).sum() / df["venda_media_dia"].sum(),
    "Lojas para Ação (P1+P2)": int(df["prioridade"].isin(["P1 · AÇÃO URGENTE","P2 · AÇÃO PLANEJADA"]).sum()),
    "Crescimento Anual Total (R$)": df["crescimento_esperado_anual_R$"].sum(),
}])
re = pd.concat([re, total_row], ignore_index=True)

# Também salva cópia NA RAIZ do projeto com os 2 sheets exatos pedidos
arq_raiz = ROOT / "fase6_plano_metas_comercial_entrega.xlsx"
arq_entrega = ENTREGA / "fase6_plano_metas_comercial_por_loja.xlsx"
try:
    with pd.ExcelWriter(arq_entrega) as xw:
        aba1.to_excel(xw, sheet_name="Metas por Loja", index=False)
        re.to_excel(xw, sheet_name="Resumo Executivo", index=False)
    print(f"> Excel (entrega_fase6) salvo em: {arq_entrega}")
    # cópia idêntica na raiz
    import shutil
    shutil.copy2(arq_entrega, arq_raiz)
    print(f"> Cópia idêntica NA RAIZ salva em: {arq_raiz}")
except Exception as e:
    print(f"> Excel não gerado: {e}")

print(f"> CSV plano de metas salvo em: {ENTREGA / 'fase6_plano_metas_comercial_por_loja.csv'}")
print(f"> Crescimento total anual da rede (projeta meta × 365 dias): "
      f"R$ {df['crescimento_esperado_anual_R$'].sum():,.0f}")
