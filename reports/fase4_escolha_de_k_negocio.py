# -*- coding: utf-8 -*-
"""
FASE 4 - Análise de perfil por cluster p/ decisão de k de negócio.
Para k = 2,3,4,5,6,7:
- tamanho dos clusters (n e %)
- perfil médio em escala ORIGINAL (não padronizada) p/ interpretação
- teste de negócio: distribuição por tipo_loja e sortimento (baselines)
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"

FEATURES_X = [
    "venda_media_dia_log1p",
    "clientes_medio_dia_log1p",
    "ticket_medio",
    "razao_promo",
    "promo_continua",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal_log1p",
    "distancia_concorrente_log1p",
]

FEATURES_INTERPRET = [
    "venda_media_dia",
    "clientes_medio_dia",
    "ticket_medio",
    "razao_promo",
    "uplift_promo_pct",
    "promo_continua",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal",
    "tx_domingos_abertos",
    "distancia_concorrente",
]

# Carregar tabela com features + baselines
tab = pd.read_csv(DATA_PROC / "tabela_por_loja_fase3_com_features.csv")
labels = pd.read_csv(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.csv")
df = tab.merge(labels, on="loja", how="inner")
assert len(df) == 1115

cols_baselines = ["tipo_loja", "sortimento"]

# =========================
# 1) Critério 4: ganho sobre baseline (1-way ANOVA pseudo R²)
# Função: SS_between / SS_total
def r2_oneway(y, cat):
    g = y.groupby(cat)
    means = g.mean()
    overall = y.mean()
    ss_bet = (g.count() * (means - overall) ** 2).sum()
    ss_tot = ((y - overall) ** 2).sum()
    return ss_bet / ss_tot if ss_tot > 0 else np.nan

alvos_anova = ["venda_media_dia", "ticket_medio", "clientes_medio_dia",
               "razao_promo", "cv_volatilidade_diaria", "cv_sazonalidade_mensal",
               "distancia_concorrente"]

anova_rows = []
for baseline_name, col_cat in [("baseline tipo_loja", "tipo_loja"),
                               ("baseline sortimento", "sortimento")]:
    for ycol in alvos_anova:
        anova_rows.append({
            "segmentacao": baseline_name,
            "alvo": ycol,
            "R²_1way": r2_oneway(df[ycol], df[col_cat]),
        })

for k in [2, 3, 4, 5, 6, 7]:
    col_cluster = f"cluster_k{k}"
    for ycol in alvos_anova:
        anova_rows.append({
            "segmentacao": f"KMeans k={k}",
            "alvo": ycol,
            "R²_1way": r2_oneway(df[ycol], df[col_cluster]),
        })

anova = pd.DataFrame(anova_rows).pivot(index="segmentacao", columns="alvo", values="R²_1way")
anova["R²_medio"] = anova.mean(axis=1)
anova = anova.sort_values("R²_medio", ascending=False)
print("===== R² da segmentação (1-way ANOVA por variável alvo). R² maior = melhor =====")
print((anova * 100).round(2).to_string())  # %
print()

# =========================
# 2) Estabilidade pura entre k vizinhos: % de lojas que caem em um cluster de tamanho >1
# e Adjusted Rand Index entre k e k+1 (diz se as partições são consistentes entre si)
from sklearn.metrics import adjusted_rand_score
print("===== Consistência entre k vizinhos (ARI) =====")
for k in [2,3,4,5,6]:
    ari = adjusted_rand_score(df[f"cluster_k{k}"], df[f"cluster_k{k+1}"])
    print(f"  ARI(k={k} vs k={k+1}) = {ari:.3f}")
print()

# =========================
# 3) Perfil interpretável de cada cluster, k escolhidos
for k in [4,5,6]:
    col = f"cluster_k{k}"
    print(f"\n========= PERFIL DOS CLUSTERS K={k} (escala de negócio original) =========")
    # Tamanho
    size = (df.groupby(col)["loja"].agg(["count"])
              .rename(columns={"count": "n_lojas"}))
    size["%_da_rede"] = (size["n_lojas"] / len(df) * 100).round(1)
    # Perfil médio
    prof = df.groupby(col)[FEATURES_INTERPRET].mean().round(2)
    # Baselines (% de cada tipo_loja e sortimento DENTRO do cluster)
    tl = pd.crosstab(df[col], df["tipo_loja"], normalize="index").round(3) * 100
    tl.columns = [f"%tipo_{c}" for c in tl.columns]
    so = pd.crosstab(df[col], df["sortimento"], normalize="index").round(3) * 100
    so.columns = [f"%sort_{c}" for c in so.columns]
    merged = pd.concat([size, prof, tl, so], axis=1)
    print(merged.to_string())

# =========================
# 4) Sanity check: % das tipo_loja b (17 lojas) que estão NOS MESMOS clusters
# Esperado para um modelo "melhor que baseline": o grupo raro b ficar concentrado.
print("\n===== Captura do baseline tipo_loja b (17 lojas raras) =====")
for k in [2,3,4,5,6,7]:
    col = f"cluster_k{k}"
    aux = df[df["tipo_loja"]=="b"].groupby(col)["loja"].count().sort_values(ascending=False)
    # concentração = share do maior cluster dentro do tipo b
    conc = aux.iloc[0] / aux.sum()
    # quantos clusters distintos tem tipo b
    qtd_clusters_b = len(aux)
    # % tipo b no maior cluster (recall do baseline)
    share = aux.iloc[0] / df.loc[df[col]==aux.index[0], "loja"].count() * 100
    print(f"  k={k} | tipo_b espalhado em {qtd_clusters_b} clusters | "
          f"maior concentração tem {aux.iloc[0]}/17 tipo b = "
          f"{conc*100:.0f}% do grupo | "
          f"esse cluster é {share:.1f}% tipo_b (pureza)")

# Salvar tabela anova para reuso
anova.reset_index().to_csv(DATA_PROC / "fase4_anova_baseline_vs_kmeans.csv", index=False,
                           float_format="%.4f")
print(f"\n> R² ANOVA salvo em fase4_anova_baseline_vs_kmeans.csv")
