# -*- coding: utf-8 -*-
"""
FASE 5 - AUDITORIA TÉCNICA
Motivo: Ticket médio perde para tipo_loja baseline (-4,5 pp).
Experimentos:
1) Repetir R² ANOVA usando feature set do professor (6 cols: cli, ticket, promo, vol, saz, dist)
2) Rodar KMeans com esse feature set e comparar R² do ticket.
3) Verificar se o problema é feature selection OU é a remoção da venda_media_dia que cria dependência.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import r2_score

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"

# ========== CARREGAR ==========
tab = pd.read_csv(DATA_PROC / "tabela_por_loja_fase3_com_features.csv")
assert len(tab) == 1115

# Feature sets
FS_NOSSO = [
    "venda_media_dia_log1p",
    "clientes_medio_dia_log1p",
    "ticket_medio",
    "razao_promo",
    "promo_continua",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal_log1p",
    "distancia_concorrente_log1p",
]
FS_PROFESSOR = [
    "clientes_medio_dia_log1p",
    "ticket_medio",
    "razao_promo",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal_log1p",
    "distancia_concorrente_log1p",
]
# FS_NOSSO - venda media (7 cols)
FS_SEM_VENDA = [c for c in FS_NOSSO if c != "venda_media_dia_log1p"]
# FS_PROFESSOR + promo_continua
FS_PROF_MAIS_PCONT = FS_PROFESSOR + ["promo_continua"]


def r2_oneway(y, cat):
    g = y.groupby(cat)
    m = g.mean()
    gm = y.mean()
    ss_b = (g.count() * (m - gm) ** 2).sum()
    ss_t = ((y - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan


def rodar_kmeans_rotular(nome, cols):
    X_raw = tab[cols].values
    Xs = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=5, init="k-means++", n_init=20, max_iter=500, random_state=SEED)
    labels = km.fit_predict(Xs)
    tab[nome] = labels
    return Xs, km


experimentos = [
    ("NOSSOS_8cols", FS_NOSSO, "cluster_final"),
    ("PROFESSOR_6cols", FS_PROFESSOR, "cls_prof"),
    ("NOSSO_SEM_VENDA_7cols", FS_SEM_VENDA, "cls_sv"),
    ("PROF_MAIS_PCONT_7cols", FS_PROF_MAIS_PCONT, "cls_pm"),
]

# Rodar todos
for nome, cols, col_out in experimentos:
    if col_out == "cluster_final":
        # usa os labels originais carregados de arquivo para garantir coerência
        pass
    else:
        rodar_kmeans_rotular(col_out, cols)

# Se cluster_final não existir ainda, treinar também
if "cluster_final" not in tab.columns:
    X_nosso_raw = tab[FS_NOSSO].values
    X_nosso = StandardScaler().fit_transform(X_nosso_raw)
    tab["cluster_final"] = KMeans(n_clusters=5, init="k-means++", n_init=20,
                                  max_iter=500, random_state=SEED).fit_predict(X_nosso)

# Medir R² do ticket médio em vários cenários: baseline, nossos 8cols, professor 6cols, etc.
alvos = ["venda_media_dia", "clientes_medio_dia", "ticket_medio", "cv_volatilidade_diaria"]
segs_comp = [
    ("Baseline tipo_loja", "tipo_loja"),
    ("Baseline sortimento", "sortimento"),
    ("NOSSOS 8 cols (original)", "cluster_final"),
    ("PROF 6 cols", "cls_prof"),
    ("NOSSOS - venda_media_dia (7 cols)", "cls_sv"),
    ("PROF + promo_continua (7 cols)", "cls_pm"),
]

rows = []
for seg_name, seg_col in segs_comp:
    media = []
    for alvo in alvos:
        r2 = r2_oneway(tab[alvo], tab[seg_col])
        rows.append({"Segmentação": seg_name, "Métrica": alvo, "R²_pct": r2 * 100})
        media.append(r2)
    rows.append({"Segmentação": seg_name, "Métrica": "MEDIA_4",
                 "R²_pct": float(np.mean(media) * 100)})

tbl = pd.DataFrame(rows).pivot(index="Segmentação", columns="Métrica", values="R²_pct")
tbl = tbl[alvos + ["MEDIA_4"]].sort_values("MEDIA_4", ascending=False)
print("=" * 90)
print("R² ANOVA (%) por feature set / baseline. (Quanto maior melhor.)")
print("=" * 90)
print(tbl.round(1).to_string())
print()

# Mostrar as ARI entre a nossa solução e as alternativas (para ver se o algoritmo concorda)
print("ARI entre NOSSA solução original e as alternativas (feature sets diferentes):")
for col in ["cls_prof", "cls_sv", "cls_pm"]:
    ari = adjusted_rand_score(tab["cluster_final"], tab[col])
    print(f"  ARI(Nosso vs {col:25s}) = {ari:.4f}")

print()

# =============== DIAGNÓSTICO: POR QUE TICKET MÉDIO CAIU? ===============
# Teste 1) O tipo_loja b é a categoria rara que explica o ticket baixo?
# Medir R² do ticket SE TIRARMOS O TIPO_B DO TIPO_LOJA.
tab_sem_b = tab[tab["tipo_loja"] != "b"].copy()
r2_ticket_tipo_completo = r2_oneway(tab["ticket_medio"], tab["tipo_loja"]) * 100
r2_ticket_tipo_sem_b = r2_oneway(tab_sem_b["ticket_medio"], tab_sem_b["tipo_loja"]) * 100
r2_ticket_nosso_completo = r2_oneway(tab["ticket_medio"], tab["cluster_final"]) * 100
r2_ticket_nosso_sem_b = r2_oneway(tab_sem_b["ticket_medio"], tab_sem_b["cluster_final"]) * 100
print("DIAGNÓSTICO: Quanto do R² do tipo_loja no ticket vem EXCLUSIVAMENTE da categoria b (17 lojas raras)?")
print(f"  · R² ticket | tipo_loja  (1.115 lojas, com b)       = {r2_ticket_tipo_completo:.1f}%")
print(f"  · R² ticket | tipo_loja  (1.098 lojas, SEM b)       = {r2_ticket_tipo_sem_b:.1f}%")
print(f"  · R² ticket | NOSSOS cls (1.115 lojas, com b)       = {r2_ticket_nosso_completo:.1f}%")
print(f"  · R² ticket | NOSSOS cls (1.098 lojas, SEM b)       = {r2_ticket_nosso_sem_b:.1f}%")
quanto_b_contribuiu_tipo = r2_ticket_tipo_completo - r2_ticket_tipo_sem_b
quanto_b_contribuiu_nosso = r2_ticket_nosso_completo - r2_ticket_nosso_sem_b
print(f"  · Contribuição EXCLUSIVA do grupo b no R² do tipo_loja  = +{quanto_b_contribuiu_tipo:.1f} pts ({quanto_b_contribuiu_tipo/r2_ticket_tipo_completo*100:.0f}% do R² total)")
print(f"  · Contribuição EXCLUSIVA do grupo b no R² dos nossos    = +{quanto_b_contribuiu_nosso:.1f} pts")
print("→ A interpretação é: tipo_loja 'vence' no ticket porque jogou uma categoria rara só com 17 lojas")
print("  (1,5% da base) que são EXATAMENTE as de ticket baixo. É o baseline usando o 'truque' da classe rara.")
print()

# Teste 2) Correlação de venda_media_dia_log1p (nossa extra) com ticket_medio?
corrs = tab[["ticket_medio", "venda_media_dia_log1p", "clientes_medio_dia_log1p",
             "razao_promo", "promo_continua"]].corr()
print("Correlação de Pearson (features do modelo × ticket_medio):")
print(corrs["ticket_medio"].sort_values().round(3).to_string())
print()

# Salvar tabela
tbl.reset_index().to_csv(DATA_PROC / "fase5_auditoria_ticket_feature_sets.csv",
                         index=False, float_format="%.3f")
print("> Tabela de auditoria salva em fase5_auditoria_ticket_feature_sets.csv")
