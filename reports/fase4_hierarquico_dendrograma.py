# -*- coding: utf-8 -*-
"""
FASE 4 - Modelagem - Passo 2 - Hierárquico
Clusterização hierárquica sobre X_scaled (8 features padronizadas).
Dendrograma truncado nos últimos 30 grupos.
Critério de linkage: Ward (minimiza variância intra).
Métrica: euclidiana.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

SEED = 42
plt.rcParams["font.family"] = "Segoe UI"
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_PROC / "X_scaled.npy")
assert X.shape == (1115, 8), X.shape

# ========== ITEM 1: linkage + dendrograma truncado últimos 30 grupos ==========
Z = linkage(X, method="ward", metric="euclidean")
# Coeficiente cofenético: 0-1; mede preservação de distâncias originais pela hierarquia
c, coph_dists = cophenet(Z, pdist(X))
print(f"> Coeficiente cofenético (Ward) = {c:.3f}  (0=ruim, 1=ótimo)")

# 30 últimas distâncias (merge do passo 1085..1114)
zs = pd.DataFrame(Z, columns=["i", "j", "d", "n"])
zs["merge_step"] = zs.index + 1  # 1..1114
zs_tail = zs.tail(40).copy()
zs_tail["gap"] = zs_tail["d"].diff()
print("\n===== Últimos 15 merges (maior altura) =====")
print(zs_tail[["merge_step", "d", "n", "gap"]].tail(15).to_string(index=False,
                                                                 float_format=lambda v: f"{v:.2f}"))

fig, ax = plt.subplots(figsize=(14, 6))
ddata = dendrogram(
    Z,
    ax=ax,
    truncate_mode="lastp",     # últimos p grupos
    p=30,
    leaf_rotation=0,
    leaf_font_size=9,
    show_contracted=True,
    above_threshold_color="#555555",
    color_threshold=None,
)
# Anotar as maiores "lacunas" (alturas de merge grandes)
alturas = np.sort(Z[:, 2])
top_jumps = np.argsort(np.diff(alturas))[-6:][::-1]  # maiores saltos (índices no vetor ordenado)
print("\n===== Saltos de altura na distância ordenada (propostas de corte) =====")
for idx in top_jumps:
    k1 = len(Z) - idx            # n clusters ANTES do salto
    k2 = len(Z) - idx - 1        # n clusters DEPOIS do salto
    d_de = alturas[idx]
    d_ate = alturas[idx + 1]
    salto = d_ate - d_de
    print(f"  k={k1:3d} → k={k2:3d}  | corte em altura entre {d_de:.2f} e {d_ate:.2f}  | salto={salto:.2f}")

ax.set_title("Dendrograma Hierárquico (Ward) — Truncado nos últimos 30 grupos\n"
             f"X_scaled (8 features) · n=1.115 lojas · cophenet={c:.3f}",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Altura (distância de ligação - Ward)")
ax.set_xlabel("Tamanho do grupo (entre parênteses)")
# Marcador sugerido de corte
ax.axhline(alturas[-6], linestyle="--", color="#b2182b", linewidth=1.5, alpha=0.8,
           label=f"Corte propositura ~{alturas[-6]:.1f} altura")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
png = FIG / "fase4_dendrograma_ward_truncado_30.png"
fig.savefig(png, dpi=200, bbox_inches="tight")
print(f"\n> Figura salva: {png}")
plt.close(fig)

# ========== ITEM 3: Consistência com KMeans (k=2..10) via ARI ==========
km_labels = pd.read_csv(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.csv")
ari_rows = []
for k in range(2, 11):
    lh = fcluster(Z, t=k, criterion="maxclust")
    lkm = km_labels[f"cluster_k{k}"].values
    ari = adjusted_rand_score(lh, lkm)
    ari_rows.append({"k": k, "ARI_hierarq_vs_kmeans": round(ari, 3)})
tab_ari = pd.DataFrame(ari_rows)
print("\n===== ARI (partição hierárquica vs KMeans) por k =====")
print(tab_ari.to_string(index=False))
tab_ari.to_csv(DATA_PROC / "fase4_ari_hierarquico_vs_kmeans.csv", index=False)
