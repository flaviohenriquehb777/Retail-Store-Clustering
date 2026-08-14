# -*- coding: utf-8 -*-
"""
FASE 4 - Modelagem - Passo 1
Aplicar StandardScaler nas 8 features fechadas.
Rodar KMeans k=2..10 com n_init=20, SEED=42.
Montar tabela inercia + silhueta.
Plotar cotovelo e silhueta lado a lado.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
np.random.seed(SEED)

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "venda_media_dia_log1p",
    "clientes_medio_dia_log1p",
    "ticket_medio",
    "razao_promo",
    "promo_continua",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal_log1p",
    "distancia_concorrente_log1p",
]

# 1) Carregar a tabela com as features
path_csv = DATA_PROC / "tabela_por_loja_fase3_com_features.csv"
df = pd.read_csv(path_csv)
assert df["loja"].nunique() == 1115
assert len(df) == 1115

print(f"> Carregado {len(df)} lojas de {path_csv.name}")
print(f"> Features: {len(FEATURES)}")

# 2) Padronizar com StandardScaler
X_raw = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

stats = pd.DataFrame({
    "feature": FEATURES,
    "media_original": scaler.mean_.round(5),
    "std_original": scaler.scale_.round(5),
    "media_escalonada": X_scaled.mean(axis=0).round(5),
    "std_escalonada": X_scaled.std(axis=0).round(5),
})
print("\n=== Conferência do StandardScaler (todas devem ficar média ~= 0, std ~= 1) ===")
print(stats.to_string(index=False))

# Salvar artefatos
np.save(DATA_PROC / "X_scaled.npy", X_scaled)
np.savetxt(DATA_PROC / "X_scaled.csv", X_scaled, delimiter=",",
           header=",".join(FEATURES), comments="")
df_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=df["loja"])
df_scaled.reset_index().to_parquet(DATA_PROC / "X_scaled.parquet", index=False)
with open(DATA_PROC / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"\n> Artefatos salvos em {DATA_PROC}: X_scaled, scaler.pkl")

# 3) Rodar KMeans k=2..10, n_init=20, random_state=SEED
KS = list(range(2, 11))
results = []
labels_store = {}

for k in KS:
    model = KMeans(n_clusters=k, n_init=20, random_state=SEED, max_iter=500)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, metric="euclidean", random_state=SEED)
    results.append({
        "k": k,
        "inercia": round(model.inertia_, 2),
        "silhueta": round(sil, 3),
    })
    labels_store[k] = labels
    print(f"> k={k} | inercia={model.inertia_:12,.2f} | silhueta={sil:.3f}")

tbl = pd.DataFrame(results)
tbl.to_csv(DATA_PROC / "fase4_kmeans_metricas.csv", index=False)
print("\n=== TABELA FINAL ===")
print(tbl.to_string(index=False))

# Salvar labels (para uso posterior)
df_labels = pd.DataFrame({"loja": df["loja"]})
for k in KS:
    df_labels[f"cluster_k{k}"] = labels_store[k]
df_labels.to_csv(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.csv", index=False)
df_labels.to_parquet(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.parquet", index=False)
print(f"> Labels salvos em fase4_kmeans_labels_k2_a_k10.csv")

# 4) Cotovelo e Silhueta lado a lado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cotovelo
axes[0].plot(tbl["k"], tbl["inercia"], marker="o", color="#2166ac", linewidth=2)
axes[0].set_title("Método do Cotovelo — KMeans")
axes[0].set_xlabel("Número de Clusters (k)")
axes[0].set_ylabel("Inércia (Soma dos Quadrados Intra-cluster)")
axes[0].set_xticks(KS)
for _, row in tbl.iterrows():
    axes[0].annotate(f"{row['inercia']:,.0f}",
                     (row["k"], row["inercia"]),
                     textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="#2166ac")

# Silhueta
axes[1].plot(tbl["k"], tbl["silhueta"], marker="s", color="#b2182b", linewidth=2)
best = tbl["silhueta"].idxmax()
axes[1].scatter(tbl.loc[best, "k"], tbl.loc[best, "silhueta"],
                s=150, facecolors="none", edgecolors="#b2182b", linewidths=2, zorder=5,
                label=f"Máximo k={tbl.loc[best, 'k']} sil={tbl.loc[best, 'silhueta']:.3f}")
axes[1].legend(fontsize=9, loc="upper right")
axes[1].set_title("Coeficiente de Silhueta Médio — KMeans")
axes[1].set_xlabel("Número de Clusters (k)")
axes[1].set_ylabel("Silhueta Média (euclidiana)")
axes[1].set_xticks(KS)
axes[1].set_ylim(0.165, 0.200)
axes[1].set_yticks(np.arange(0.165, 0.200 + 1e-9, 0.005))
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
axes[1].grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.7, color="#888888")
for _, row in tbl.iterrows():
    axes[1].annotate(f"{row['silhueta']:.3f}",
                     (row["k"], row["silhueta"]),
                     textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="#b2182b")

fig.suptitle("Fase 4 — KMeans k=2..10 | n_init=20 | random_state=42 | 8 features padronizadas",
             fontsize=14, fontweight="bold")
plt.tight_layout()
png = FIG / "fase4_cotovelo_silhueta_kmeans.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
print(f"\n> Gráfico salvo: {png}")
plt.close(fig)
