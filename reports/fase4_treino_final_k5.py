# -*- coding: utf-8 -*-
"""
FASE 4 - Passo 3 - Treinamento FINAL KMeans k=5.
Salva modelo final, tabela de lojas com rótulos, tamanhos de cluster, e análise das variáveis retiradas (binárias/categóricas: tipo_loja, sortimento, abre_domingo = tx_domingos_abertos≥50%, e razao_pico_sabado).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"
DATA_MOD = ROOT / "models"
DATA_MOD.mkdir(parents=True, exist_ok=True)

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

# ========== CARREGAR ==========
X = np.load(DATA_PROC / "X_scaled.npy")
tab = pd.read_csv(DATA_PROC / "tabela_por_loja_fase3_com_features.csv")
assert len(tab) == 1115 and len(X) == 1115

# ========== TREINAR MODELO FINAL ==========
km = KMeans(n_clusters=5, init="k-means++", n_init=20,
            max_iter=500, random_state=SEED, verbose=0)
tab["cluster_final"] = km.fit_predict(X).astype(int)
inercia = float(km.inertia_)
print(f"> KMeans k=5 treinado. Inércia final = {inercia:,.2f}")

# Salvar modelo final + mover scaler para pasta models (mantendo backup do caminho original)
import shutil
scaler_origem = DATA_PROC / "scaler.pkl"
scaler_destino = DATA_MOD / "scaler_final.pkl"
if scaler_origem.exists():
    shutil.copy(scaler_origem, scaler_destino)
    print(f"> Scaler copiado p/ models: {scaler_destino}")
else:
    print(f"> Aviso: scaler original não encontrado em {scaler_origem}")
with open(DATA_MOD / "kmeans_final_k5.pkl", "wb") as f:
    pickle.dump(km, f)
print(f"> Modelo salvo em: {DATA_MOD / 'kmeans_final_k5.pkl'}")

# ========== TAMANHOS DOS CLUSTERS ==========
print("\n===== TAMANHO DOS CLUSTERS (k=5) =====")
sz = (tab.groupby("cluster_final")["loja"]
         .agg(n_lojas="count")
         .sort_values("n_lojas", ascending=False))
sz["pct_rede"] = (sz["n_lojas"] / len(tab) * 100).round(2)
sz["pct_rede_acum"] = sz["pct_rede"].cumsum().round(2)
print(sz.to_string())

# ========== VARIÁVEIS RETIRADAS (BINÁRIAS/CATEGÓRICAS) vs CLUSTER ==========
# Cria flag explicitas
tab["bin_abre_domingo"] = (tab["tx_domingos_abertos"] >= 0.5).astype(int)
# Pico de sábado em 3 faixas: < 0.9 (sabado ruim), 0.9-1.1 (medio), >1.1 (pico sabado)
tab["faixa_pico_sabado"] = pd.cut(tab["razao_pico_sabado"],
                                  bins=[-np.inf, 0.9, 1.1, np.inf],
                                  labels=["Sab_pior_que_segsex", "Sab_parecido", "Sab_pico_semana"])

print("\n===== DISTRIBUIÇÃO TIPO_LOJA DENTRO DE CADA CLUSTER (em % do cluster) =====")
tl = pd.crosstab(tab["cluster_final"], tab["tipo_loja"], normalize="index").round(3)*100
tl.columns = [f"tipo_{c} [%]" for c in tl.columns]
tl["n"] = sz["n_lojas"]
print(tl.to_string())

print("\n===== DISTRIBUIÇÃO SORTIMENTO DENTRO DE CADA CLUSTER (em % do cluster) =====")
so = pd.crosstab(tab["cluster_final"], tab["sortimento"], normalize="index").round(3)*100
so.columns = [f"sort_{c} [%]" for c in so.columns]
so["n"] = sz["n_lojas"]
print(so.to_string())

print("\n===== % DE LOJAS QUE ABRE DOMINGO (binária, t≥50%) POR CLUSTER =====")
dom = (tab.groupby("cluster_final")["bin_abre_domingo"]
          .agg(n_lojas="count",
               n_abrem_domingo="sum",
               pct_abrem_domingo=lambda s: s.mean()*100))
dom["pct_abrem_domingo"] = dom["pct_abrem_domingo"].round(1)
print(dom.to_string())

print("\n===== % DE LOJAS COM PICO EM SÁBADO / MÉDIO / RUIM POR CLUSTER =====")
ps = pd.crosstab(tab["cluster_final"], tab["faixa_pico_sabado"], normalize="index").round(3)*100
ps["n"] = sz["n_lojas"]
print(ps.to_string())

# ========== SALVAR TABELA FINAL ==========
tab.to_csv(DATA_PROC / "tabela_lojas_com_cluster_final.csv", index=False,
           float_format="%.4f")
tab.to_parquet(DATA_PROC / "tabela_lojas_com_cluster_final.parquet", index=False)
print(f"\n> Tabela final salva (n={len(tab)}): data/03_processed/tabela_lojas_com_cluster_final.*")

# Cross-check: o cluster_final é igual ao cluster_k5 do grid?
labels_df = pd.read_csv(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.csv")
chk = tab[["loja", "cluster_final"]].merge(labels_df, on="loja")
ari = adjusted_rand_score(chk["cluster_final"], chk["cluster_k5"])
print(f"> Sanity check ARI(cluster_final vs grid cluster_k5) = {ari:.6f} (esperado 1.0)")
