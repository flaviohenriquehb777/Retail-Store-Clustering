# -*- coding: utf-8 -*-
"""
FASE 5 - AVALIAÇÃO
Parte 1: Estabilidade, Baseline (tipo_loja/sortimento), Perfil Médio dos clusters.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
FIG = ROOT / "reports" / "figures"

X = np.load(DATA_PROC / "X_scaled.npy")
tab = pd.read_csv(DATA_PROC / "tabela_lojas_com_cluster_final.csv")
assert len(tab) == 1115
k_final = 5
l_ref = tab["cluster_final"].values


def km(k, X, rs=SEED):
    return KMeans(n_clusters=k, init="k-means++", n_init=20,
                  max_iter=500, random_state=rs).fit_predict(X)


# ===================== ITEM 1: ESTABILIDADE =====================
print("=" * 80)
print("ITEM 1 — ESTABILIDADE DO KMeans k=5")
print("=" * 80)
print("\nExperimento 1a) Reexecutando 10 vezes com random_states diferentes:")
print(f"  (comparando cada run com a referência SEED={SEED})")
rs_list = [42, 123, 456, 789, 1024, 2048, 7, 91, 314, 2718]
res_a = []
for rs in rs_list:
    y = km(k_final, X, rs=rs)
    res_a.append({"rs": rs,
                  "ARI_vs_ref": adjusted_rand_score(l_ref, y),
                  "AMI_vs_ref": adjusted_mutual_info_score(l_ref, y)})
df_a = pd.DataFrame(res_a)
print(df_a.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média  ARI = {df_a['ARI_vs_ref'].mean():.4f}  min = {df_a['ARI_vs_ref'].min():.4f}  max = {df_a['ARI_vs_ref'].max():.4f}")
print(f"  ▸ Média  AMI = {df_a['AMI_vs_ref'].mean():.4f}")

print("\nExperimento 1b) 10 amostras bootstrap SEM reposição de 80% das lojas (holdout):")
print("  Treino KMeans na amostra, predição NO MESMO conjunto. Medo: clusters que só existem por causa das 180 reformadas.")
res_b = []
for i in range(10):
    idx = resample(np.arange(len(X)), replace=False, n_samples=int(0.8*len(X)), random_state=SEED+i)
    l_in_ref = l_ref[idx]
    l_in_new = km(k_final, X[idx], rs=SEED+i)
    res_b.append({"seed": SEED+i,
                  "n_amostra": len(idx),
                  "ARI_interno (ref vs new na amostra)": adjusted_rand_score(l_in_ref, l_in_new),
                  "AMI_interno": adjusted_mutual_info_score(l_in_ref, l_in_new)})
df_b = pd.DataFrame(res_b)
print(df_b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média  ARI bootstrap = {df_b.iloc[:,2].mean():.4f}  min = {df_b.iloc[:,2].min():.4f}")

print("\nExperimento 1c) 5 shuffles de ordem das linhas (o que mais tem potencial de enviesar KMeans):")
rng = np.random.default_rng(SEED)
res_c = []
for s in range(5):
    p = rng.permutation(len(X))
    Xs = X[p]
    ys = km(k_final, Xs, rs=SEED+s)
    yr = np.empty_like(ys)
    yr[p] = ys
    res_c.append({"seed_passo": s,
                  "ARI_vs_ref": adjusted_rand_score(l_ref, yr)})
df_c = pd.DataFrame(res_c)
print(df_c.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média  ARI shuffle = {df_c['ARI_vs_ref'].mean():.4f}")

# ================================================================
# ITEM 2A — Concordância nossos clusters vs tipo_loja / sortimento
# ================================================================
print("\n" + "=" * 80)
print("ITEM 2A — CONCORDÂNCIA COM BASELINES (tipo_loja e sortimento)")
print("=" * 80)
print("ARI/AMI perto de 0 = pouca concordância (bom p/ 'não é o mesmo agrupamento'). Perto de 1 = idênticos.")
for bl, col in [("tipo_loja", "tipo_loja"), ("sortimento", "sortimento")]:
    ari = adjusted_rand_score(tab[col], tab["cluster_final"])
    ami = adjusted_mutual_info_score(tab[col], tab["cluster_final"])
    # Normalized Mutual Information (interpretável diretamente como % de info compartilhada)
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(tab[col], tab["cluster_final"])
    print(f"▸ cluster_final vs {bl:15s}:  ARI = {ari:.4f}   AMI = {ami:.4f}   NMI = {nmi:.4f}")
print()
print("Tabela cruzada (%) cluster × tipo_loja (linha soma 100%):")
ct_tl = pd.crosstab(tab["cluster_final"], tab["tipo_loja"], normalize="index").round(3) * 100
ct_tl["n_lojas"] = tab["cluster_final"].value_counts().sort_index()
print(ct_tl.to_string(float_format=lambda v: f"{v:.0f}%"))
print()
print("Tabela cruzada (%) cluster × sortimento:")
ct_so = pd.crosstab(tab["cluster_final"], tab["sortimento"], normalize="index").round(3) * 100
ct_so["n_lojas"] = tab["cluster_final"].value_counts().sort_index()
print(ct_so.to_string(float_format=lambda v: f"{v:.0f}%"))

# ================================================================
# ITEM 2B — 1-way ANOVA pseudo-R² (SSentre/SStotal) em 4 métricas de negócio
# ================================================================
print("\n" + "=" * 80)
print("ITEM 2B — EXPLICAÇÃO DO NEGÓCIO vs BASELINE (pseudo R² ANOVA 1 via)")
print("=" * 80)
print("Métrica escolhida: R² de uma análise de variância de 1 via (1-way ANOVA).")
print("Fórmula: R² = SS_entre_grupos / SS_total.")
print("Por quê essa métrica aqui (2 frases):")
print("  1) Ela responde exatamente 'quanta % da variação da métrica de negócio é explicada pelo agrupamento'.")
print("  2) É a única medida comparável DIRETAMENTE entre agrupamentos de k diferente (tipo_loja k=4 vs sortimento k=3 vs nossos k=5).")
print()


def r2_oneway(y, cat):
    g = y.groupby(cat)
    m = g.mean()
    gm = y.mean()
    ss_entre = (g.count() * (m - gm) ** 2).sum()
    ss_total = ((y - gm) ** 2).sum()
    return ss_entre / ss_total if ss_total > 0 else np.nan


alvos = [
    ("venda_media_dia", "Venda média / dia aberto (R$)"),
    ("clientes_medio_dia", "Clientes médios / dia (pessoas)"),
    ("ticket_medio", "Ticket médio (R$)"),
    ("cv_volatilidade_diaria", "Volatilidade diária (CV)"),
]
segs = [
    ("Baseline tipo_loja (k=4)", "tipo_loja"),
    ("Baseline sortimento (k=3)", "sortimento"),
    ("Nossos clusters KMeans (k=5)", "cluster_final"),
]
rows = []
for nome, col in segs:
    row = {"Segmentação": nome}
    for alvo, alvo_nome in alvos:
        r2 = r2_oneway(tab[alvo], tab[col])
        row[alvo_nome] = r2
        rows.append({
            "Segmentação": nome,
            "Métrica de Negócio": alvo_nome,
            "R²_ANOVA": r2,
            "R²_ANOVA_pct": r2 * 100,
        })
# Tabela pivot: linhas=segmentação, colunas=métricas
tbl_detalhe = pd.DataFrame(rows).drop_duplicates()
pivot = tbl_detalhe.pivot_table(index="Segmentação", columns="Métrica de Negócio", values="R²_ANOVA_pct")
pivot = pivot[[a[1] for a in alvos]]
pivot["R² MÉDIO das 4 métricas"] = pivot.mean(axis=1)
# Ordenar por média final
pivot = pivot.sort_values("R² MÉDIO das 4 métricas", ascending=False)
print("Tabela em % (maior = melhor, explica mais o negócio):")
print(pivot.round(1).to_string())
print()
# Calcular ganho ABSOLUTO e RELATIVO de nossos clusters sobre os baselines
bl_medio = {
    "tipo_loja": r2_oneway(tab[alvos[0][0]], tab["tipo_loja"]) if False else None
}
nossos = "Nossos clusters KMeans (k=5)"
base_tl = "Baseline tipo_loja (k=4)"
base_so = "Baseline sortimento (k=3)"
print("▸ GANHO ABSOLUTO de R² dos nossos clusters sobre cada baseline (pts percentuais) por métrica:")
ganho_rows = []
for alvo, alvo_nome in alvos:
    g_tl = pivot.loc[nossos, alvo_nome] - pivot.loc[base_tl, alvo_nome]
    g_so = pivot.loc[nossos, alvo_nome] - pivot.loc[base_so, alvo_nome]
    # R² de % por baseline como referência
    ref_tl = pivot.loc[base_tl, alvo_nome]
    ref_so = pivot.loc[base_so, alvo_nome]
    ganho_rel_tl = (g_tl / ref_tl * 100) if ref_tl > 0 else np.nan
    ganho_rel_so = (g_so / ref_so * 100) if ref_so > 0 else np.nan
    ganho_rows.append({
        "Métrica": alvo_nome,
        "R² baseline tipo": f"{ref_tl:.1f}%",
        "R² baseline sort.": f"{ref_so:.1f}%",
        "R² nossos": f"{pivot.loc[nossos, alvo_nome]:.1f}%",
        "Ganho vs tipo (pts)": f"+{g_tl:.1f} pp",
        "Ganho vs tipo (×)": f"{(1 + g_tl/100)/(ref_tl/100):.1f}×" if ref_tl > 0 else "—",
        "Ganho vs sort. (pts)": f"+{g_so:.1f} pp",
    })
tbl_ganho = pd.DataFrame(ganho_rows)
print(tbl_ganho.to_string(index=False))
medio_n = pivot.loc[nossos, "R² MÉDIO das 4 métricas"]
medio_t = pivot.loc[base_tl, "R² MÉDIO das 4 métricas"]
medio_s = pivot.loc[base_so, "R² MÉDIO das 4 métricas"]
print(f"\nResumo consolidado das 4 métricas:")
print(f"  · Média R² baseline tipo_loja      = {medio_t:.1f}%")
print(f"  · Média R² baseline sortimento     = {medio_s:.1f}%")
print(f"  · Média R² NOSSOS CLUSTERS k=5     = {medio_n:.1f}%")
print(f"  · GANHO ABSOLUTO vs tipo_loja      = +{(medio_n - medio_t):.1f} pontos percentuais  ({(medio_n / medio_t):.1f}×)")
print(f"  · GANHO ABSOLUTO vs sortimento     = +{(medio_n - medio_s):.1f} pontos percentuais  ({(medio_n / medio_s):.1f}×)")

# ================================================================
# ITEM 3 — Perfil médio de cada cluster
# ================================================================
print("\n" + "=" * 80)
print("ITEM 3 — PERFIL MÉDIO POR CLUSTER k=5")
print("=" * 80)
cols_profile = [
    "loja",
    "venda_media_dia",
    "clientes_medio_dia",
    "ticket_medio",
    "razao_promo",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal",
    "distancia_concorrente",
    "promo_continua",
    "tx_domingos_abertos",
]

def agg_profile(group):
    out = {}
    out["n_lojas"] = group["loja"].nunique()
    out["venda_media_dia"] = group["venda_media_dia"].mean()
    out["clientes_medio_dia"] = group["clientes_medio_dia"].mean()
    out["ticket_medio"] = group["ticket_medio"].mean()
    out["uplift_promo_x"] = group["razao_promo"].mean()
    out["sensibilidade_promo_pct"] = (group["razao_promo"].mean() - 1) * 100
    out["cv_volatilidade"] = group["cv_volatilidade_diaria"].mean()
    out["cv_sazonalidade"] = group["cv_sazonalidade_mensal"].mean()
    out["mediana_dist_concorrente"] = group["distancia_concorrente"].median()
    out["tx_promo_continua_pct"] = group["promo_continua"].mean() * 100
    out["tx_domingos_abertos_pct"] = group["tx_domingos_abertos"].mean() * 100
    return pd.Series(out)

prof = tab.groupby("cluster_final").apply(agg_profile).reset_index()
prof["n_lojas"] = prof["n_lojas"].astype(int)
prof["cluster_final"] = prof["cluster_final"].astype(int)
prof["% da rede"] = (prof["n_lojas"] / len(tab) * 100).round(1)
# Ordem de colunas
ord_cols = [
    "cluster_final", "n_lojas", "% da rede",
    "venda_media_dia", "clientes_medio_dia", "ticket_medio",
    "sensibilidade_promo_pct", "cv_volatilidade", "cv_sazonalidade",
    "mediana_dist_concorrente", "tx_promo_continua_pct", "tx_domingos_abertos_pct",
]
prof = prof[ord_cols].sort_values("venda_media_dia", ascending=False)
fmt_pct_1 = lambda c: lambda v: f"{v:.1f}%" if "pct" in c.lower() or c in {"% da rede"} else (
    f"{v:.0f}" if c in {"n_lojas", "mediana_dist_concorrente"} else (
        f"{v:.2f}" if c in {"ticket_medio", "cv_volatilidade", "cv_sazonalidade"} else
        f"{v:.0f}"
    )
)
# Print manual com formatação decente
def formatar_linha(row):
    return (
        f"C{row['cluster_final']:02d}  "
        f"{row['n_lojas']:>4d} ({row['% da rede']:>4.1f}%)  "
        f"VM R${row['venda_media_dia']:>6,.0f}  "
        f"Cli:{row['clientes_medio_dia']:>5,.0f}  "
        f"Ticket R${row['ticket_medio']:.2f}  "
        f"Uplift promo +{row['sensibilidade_promo_pct']:>5.1f}%  "
        f"Vol={row['cv_volatilidade']:.2f}  "
        f"Saz={row['cv_sazonalidade']:.3f}  "
        f"Dist concorr med: {row['mediana_dist_concorrente']:>5.0f} m  "
        f"Promoc.cont: {row['tx_promo_continua_pct']:>5.1f}%  "
        f"Abre dom: {row['tx_domingos_abertos_pct']:>4.1f}%"
    )

print("(Ordenado por Venda Média decrescente)")
for _, r in prof.iterrows():
    c = int(r["cluster_final"])
    n = int(r["n_lojas"])
    print((
        f"C{c:02d}  "
        f"{n:>4d} ({r['% da rede']:>4.1f}%)  "
        f"VM R${r['venda_media_dia']:>6,.0f}  "
        f"Cli:{r['clientes_medio_dia']:>5,.0f}  "
        f"Ticket R${r['ticket_medio']:.2f}  "
        f"Uplift promo +{r['sensibilidade_promo_pct']:>5.1f}%  "
        f"Vol={r['cv_volatilidade']:.2f}  "
        f"Saz={r['cv_sazonalidade']:.3f}  "
        f"Dist concorr med: {r['mediana_dist_concorrente']:>5.0f} m  "
        f"Promoc.cont: {r['tx_promo_continua_pct']:>5.1f}%  "
        f"Abre dom: {r['tx_domingos_abertos_pct']:>4.1f}%"
    ))
print()
# Salvar tabela de perfil
prof.to_csv(DATA_PROC / "fase5_perfil_medio_clusters_k5.csv", index=False, float_format="%.4f")
print("> Perfil salvo em fase5_perfil_medio_clusters_k5.csv")
