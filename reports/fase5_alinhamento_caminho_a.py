# -*- coding: utf-8 -*-
"""
FASE 5 - Alinhamento ao Caminho A.
Feature set do professor/Sol (6 cols):
  clientes_medio_dia_log1p
  ticket_medio
  dependencia_promo  (== razao_promo)
  volatilidade_cv    (== cv_volatilidade_diaria)
  sazonalidade_log1p (== cv_sazonalidade_mensal_log1p)
  distancia_concorrente_log1p

Saídas:
- modelo kmeans_final_k5_v2.pkl (feature set alinhado)
- scaler_final_v2.pkl
- tabela_lojas_com_cluster_final_v2.csv (labels novos)
- fase5_*_novos.csv para comparação e perfis
"""
import os
import sys
import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True, parents=True)

# ================ FEATURE SET DO PROFESSOR/SOL (6 cols) ===================
FEATURES_PROF = {
    "clientes_medio_dia_log1p": "clientes_medio_dia_log1p",
    "ticket_medio": "ticket_medio",
    "razao_promo": "razao_promo",          # = dependencia_promo
    "cv_volatilidade_diaria": "cv_volatilidade_diaria", # = volatilidade_cv
    "cv_sazonalidade_mensal_log1p": "cv_sazonalidade_mensal_log1p",
    "distancia_concorrente_log1p": "distancia_concorrente_log1p",
}
FS = list(FEATURES_PROF.keys())

tab = pd.read_csv(DATA_PROC / "tabela_por_loja_fase3_com_features.csv")
assert len(tab) == 1115

# Padronizar (StandardScaler novo, treinado apenas nessas 6 features, igual ao professor)
scaler2 = StandardScaler()
X_v2 = scaler2.fit_transform(tab[FS].values)
np.save(DATA_PROC / "X_scaled_prof6.npy", X_v2)
with open(MODELS / "scaler_final_v2_prof6.pkl", "wb") as f:
    pickle.dump(scaler2, f)
print(f"> X_v2 shape = {X_v2.shape}")
print(f"> Média pós scaler: {X_v2.mean(0).round(4)}")
print(f"> Dp pós scaler   : {X_v2.std(0).round(4)}")

# ================ KMeans k=5, n_init=20, seed=42 (final v2) ===================
km2 = KMeans(n_clusters=5, init="k-means++", n_init=20, max_iter=500, random_state=SEED)
tab["cluster_final_v2"] = km2.fit_predict(X_v2).astype(int)
print(f"\n> KMeans v2 (prof6) treinado. Inércia final = {km2.inertia_:,.2f}")
with open(MODELS / "kmeans_final_k5_v2_prof6.pkl", "wb") as f:
    pickle.dump(km2, f)

# ========================= TAMANHOS =========================
print("\n" + "=" * 80)
print("TAMANHOS DOS CLUSTERS (k=5) - Feature set do professor")
print("=" * 80)
sz = (tab.groupby("cluster_final_v2")["loja"].agg(n_lojas="count").sort_values("n_lojas", ascending=False))
sz["pct_rede"] = (sz["n_lojas"] / len(tab) * 100).round(2)
sz["pct_rede_acum"] = sz["pct_rede"].cumsum().round(2)
print(sz.to_string())

# ========================= ESTABILIDADE (mesmo teste anterior) =========================
def km_fit(X_in, k=5, rs=SEED):
    return KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=rs).fit_predict(X_in)

ref = tab["cluster_final_v2"].values

print("\n" + "=" * 80)
print("ESTABILIDADE (v2 / feature set do professor)")
print("=" * 80)
print("Experimento 1a: 10 random_states diferentes:")
rs_list = [42, 123, 456, 789, 1024, 2048, 7, 91, 314, 2718]
out = []
for rs in rs_list:
    y = km_fit(X_v2, k=5, rs=rs)
    out.append({"rs": rs,
                "ARI_vs_ref": adjusted_rand_score(ref, y),
                "AMI_vs_ref": adjusted_mutual_info_score(ref, y)})
df_a = pd.DataFrame(out)
print(df_a.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média ARI = {df_a.ARI_vs_ref.mean():.4f}  min = {df_a.ARI_vs_ref.min():.4f}")

print("\nExperimento 1b: 10 bootstrap 80% SEM reposição:")
out = []
for i in range(10):
    idx = resample(np.arange(len(X_v2)), replace=False, n_samples=int(0.8*len(X_v2)), random_state=SEED+i)
    out.append({"seed": SEED+i,
                "ARI_interno": adjusted_rand_score(ref[idx], km_fit(X_v2[idx], k=5, rs=SEED+i))})
df_b = pd.DataFrame(out)
print(df_b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média ARI bootstrap = {df_b.ARI_interno.mean():.4f}  min = {df_b.ARI_interno.min():.4f}")

print("\nExperimento 1c: 5 shuffles ordem das linhas:")
rng = np.random.default_rng(SEED)
out = []
for s in range(5):
    p = rng.permutation(len(X_v2))
    ys = km_fit(X_v2[p], k=5, rs=SEED+s)
    yr = np.empty_like(ys); yr[p] = ys
    out.append({"seed_passo": s, "ARI_vs_ref": adjusted_rand_score(ref, yr)})
df_c = pd.DataFrame(out)
print(df_c.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  ▸ Média ARI shuffle = {df_c.ARI_vs_ref.mean():.4f}")

# ========================= 2A CONCORDÂNCIA COM BASELINES =========================
print("\n" + "=" * 80)
print("2A CONCORDÂNCIA COM BASELINES (v2)")
print("=" * 80)
for nome, col in [("tipo_loja", "tipo_loja"), ("sortimento", "sortimento")]:
    ari = adjusted_rand_score(tab[col], tab["cluster_final_v2"])
    ami = adjusted_mutual_info_score(tab[col], tab["cluster_final_v2"])
    nmi = normalized_mutual_info_score(tab[col], tab["cluster_final_v2"])
    print(f"· Nossos clusters v2 vs {nome:15s}:  ARI = {ari:.4f}   AMI = {ami:.4f}   NMI = {nmi:.4f}")

# ========================= 2B R² ANOVA 4+ MÉTRICAS DE NEGÓCIO =========================
print("\n" + "=" * 80)
print("2B - R² ANOVA (melhor que baseline) V2")
print("=" * 80)
def r2_oneway(y, cat):
    g = y.groupby(cat)
    m = g.mean(); gm = y.mean()
    ss_b = (g.count() * (m - gm) ** 2).sum()
    ss_t = ((y - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan

alvos = [
    ("venda_media_dia", "Venda média / dia (R$)"),
    ("clientes_medio_dia", "Clientes médios / dia"),
    ("ticket_medio", "Ticket médio (R$)"),
    ("cv_volatilidade_diaria", "Volatilidade diária (CV)"),
    ("cv_sazonalidade_mensal", "Sazonalidade mensal (CV)"),
]
segs = [("Baseline tipo_loja", "tipo_loja"),
        ("Baseline sortimento", "sortimento"),
        ("NOSSOS CLUSTERS v2 (prof6 · k=5)", "cluster_final_v2")]

rows = []
for seg, col in segs:
    for alvo, nome in alvos:
        rows.append({"Segmentação": seg, "Métrica": nome, "R²_pct": r2_oneway(tab[alvo], tab[col]) * 100})
tbl_comp = pd.DataFrame(rows).pivot(index="Segmentação", columns="Métrica", values="R²_pct")
tbl_comp = tbl_comp[[a[1] for a in alvos]]
tbl_comp["R² MÉDIO"] = tbl_comp.mean(axis=1)
tbl_comp = tbl_comp.sort_values("R² MÉDIO", ascending=False)
print(tbl_comp.round(1).to_string())

print("\nGanho vs tipo_loja (nossos v2 - tipo_loja) em pontos percentuais:")
for alvo, nome in alvos:
    delta = tbl_comp.loc["NOSSOS CLUSTERS v2 (prof6 · k=5)", nome] - tbl_comp.loc["Baseline tipo_loja", nome]
    bl_val = tbl_comp.loc["Baseline tipo_loja", nome]
    mult = (tbl_comp.loc["NOSSOS CLUSTERS v2 (prof6 · k=5)", nome] / bl_val) if bl_val > 0 else np.inf
    print(f"· {nome:35s}: +{delta:.1f} pp  ({mult:.1f}× o baseline)")

# ========================= 3 PERFIL MÉDIO POR CLUSTER =========================
print("\n" + "=" * 80)
print("3 - PERFIL MÉDIO POR CLUSTER (v2) — feature set do professor")
print("=" * 80)
cols_profile = [
    "loja", "venda_media_dia", "clientes_medio_dia", "ticket_medio",
    "razao_promo", "cv_volatilidade_diaria", "cv_sazonalidade_mensal",
    "distancia_concorrente", "promo_continua", "tx_domingos_abertos",
]

def agg_profile(g):
    out = {}
    out["n_lojas"] = g["loja"].nunique()
    out["venda_media_dia"] = g["venda_media_dia"].mean()
    out["clientes_medio_dia"] = g["clientes_medio_dia"].mean()
    out["ticket_medio"] = g["ticket_medio"].mean()
    out["uplift_promo_pct"] = (g["razao_promo"].mean() - 1) * 100
    out["cv_volatilidade"] = g["cv_volatilidade_diaria"].mean()
    out["cv_sazonalidade"] = g["cv_sazonalidade_mensal"].mean()
    out["mediana_dist_concorrente"] = g["distancia_concorrente"].median()
    out["pct_promo_continua"] = g["promo_continua"].mean() * 100
    out["pct_abrem_domingo"] = (g["tx_domingos_abertos"] >= 0.5).mean() * 100
    out["tipo_b"] = (g["tipo_loja"] == "b").mean() * 100
    return pd.Series(out)

prof = tab.groupby("cluster_final_v2").apply(agg_profile).reset_index()
prof["cluster_final_v2"] = prof["cluster_final_v2"].astype(int)
prof["n_lojas"] = prof["n_lojas"].astype(int)
prof["pct_rede"] = (prof["n_lojas"] / len(tab) * 100).round(1)
prof = prof.sort_values("venda_media_dia", ascending=False)
# Imprimir legível
for _, r in prof.iterrows():
    c = int(r["cluster_final_v2"])
    n = int(r["n_lojas"])
    print(f"C{c:02d}  n={n:>4d} ({r['pct_rede']:>4.1f}%)"
          f"  VM R${r['venda_media_dia']:>6,.0f}  Cli:{r['clientes_medio_dia']:>5,.0f}"
          f"  Tk R${r['ticket_medio']:.2f}  Promo +{r['uplift_promo_pct']:>5.1f}%"
          f"  Vol={r['cv_volatilidade']:.2f}  Saz={r['cv_sazonalidade']:.3f}"
          f"  DistMed:{r['mediana_dist_concorrente']:>5.0f}m"
          f"  PromCont:{r['pct_promo_continua']:>5.1f}%"
          f"  AbreDom:{r['pct_abrem_domingo']:>4.1f}%"
          f"  TipoB:{r['tipo_b']:>4.1f}%")

# Salvar tudo
tab.to_csv(DATA_PROC / "tabela_lojas_com_cluster_final_v2_prof6.csv", index=False, float_format="%.4f")
prof.to_csv(DATA_PROC / "fase5_perfil_medio_clusters_k5_v2_prof6.csv", index=False, float_format="%.4f")
tbl_comp.reset_index().to_csv(DATA_PROC / "fase5_comparativo_r2_baseline_vs_v2_prof6.csv", index=False, float_format="%.3f")
print(f"\nSalvo: tabela_lojas_com_cluster_final_v2_prof6.csv")
print(f"Salvo: fase5_perfil_medio_clusters_k5_v2_prof6.csv")
print(f"Salvo: fase5_comparativo_r2_baseline_vs_v2_prof6.csv")
