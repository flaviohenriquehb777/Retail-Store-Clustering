# -*- coding: utf-8 -*-
"""
FASE 4 - Comparativo k=2,3,5,8 vs 5 CRITÉRIOS DE SUCESSO da Fase 1.
Gera tabela de evidências numéricas + relato por critério.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.utils import resample

SEED = 42
KS = [2, 3, 5, 8]
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"

# Carregar
X = np.load(DATA_PROC / "X_scaled.npy")
tab = pd.read_csv(DATA_PROC / "tabela_por_loja_fase3_com_features.csv")
labels_df = pd.read_csv(DATA_PROC / "fase4_kmeans_labels_k2_a_k10.csv")
df = tab.merge(labels_df, on="loja", how="inner")
assert len(df) == 1115

# =====================
# CRITÉRIO 1: Interpretável
# Medida = Pureza média das features (R² médio de uma ANOVA de 1 via usando o cluster como fator).
# Quanto >, mais separadas as variáveis por cluster → mais fácil dar nome.
def r2_1way(y, cat):
    g = y.groupby(cat)
    m = g.mean(); gm = y.mean()
    ss_bet = (g.count() * (m-gm)**2).sum()
    ss_tot = ((y-gm)**2).sum()
    return ss_bet / ss_tot if ss_tot > 0 else np.nan

FEATURES_INTERPRET = ["venda_media_dia","clientes_medio_dia","ticket_medio","razao_promo",
                      "promo_continua","cv_volatilidade_diaria","cv_sazonalidade_mensal",
                      "distancia_concorrente"]

# =====================
# CRITÉRIO 2: Acionável
# Medida objetiva = % de pares de clusters com score de ação distinto >= TH.
# Score de ação é um vetor de 4 dimensões (Comercial, Trade, Supply, Expansão)
# convertido numericamente a partir de características do cluster (ordinal 0..4).
# Se a distância euclidiana média do vetor de ação entre 2 clusters > limiar, contam como ações diferentes.
# Não queremos uma classificação literal de ação (que é subjetiva), queremos uma medida proxy de "distinção de perfil".
def perfil_acao_por_cluster(sub):
    """Retorna vetor [comercial_porte, trade_promo, supply_vol, expansao_saz] normalizado 0..1"""
    vm = sub["venda_media_dia"].mean()
    uplift = sub["razao_promo"].mean()
    p_continua = sub["promo_continua"].mean()
    cv_vol = sub["cv_volatilidade_diaria"].mean()
    cv_saz = sub["cv_sazonalidade_mensal"].mean()
    cli = sub["clientes_medio_dia"].mean()
    ticket = sub["ticket_medio"].mean()
    return np.array([
        (vm / 15000),                  # comercial: tamanho do resultado
        (uplift * p_continua) / 1.6,   # trade: intensidade de promoção
        (cv_vol / 0.35),               # supply: oscilação → estoque
        (cv_saz / 0.15),               # expansão: necessidade de sazonalizar
    ])

# =====================
# CRITÉRIO 3: Estável
# 3 experimentos:
# (a) 5 shuffles da base
# (b) 10 bootstrap 80% (SEED 42..51)
# (c) 5 random_states no KMeans
from sklearn.cluster import KMeans

def kmeans_labels(X_in, k, random_state=SEED, n_init=20):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                  max_iter=500, random_state=random_state).fit_predict(X_in)

# ================================================
# RODAR TUDO
from itertools import combinations

results = []
for k in KS:
    col = f"cluster_k{k}"
    l_ref = df[col].values
    row = {"k": k}

    # -------- CRITÉRIO 1: Interpretável --------
    r2s = [r2_1way(df[col], df[fv]) for fv in FEATURES_INTERPRET if fv != col]
    r2s2 = []
    for fv in FEATURES_INTERPRET:
        r2s2.append(r2_1way(df[fv], df[col]))
    r2s2 = np.array(r2s2)
    row["r2_interpret_medio"] = r2s2.mean()

    # -------- CRITÉRIO 2: Acionável --------
    acoes = []
    cluster_ids = sorted(df[col].unique())
    for cid in cluster_ids:
        sub = df[df[col] == cid]
        acoes.append(perfil_acao_por_cluster(sub))
    acoes = np.array(acoes)
    # todas as combinações 2 a 2
    dx = []
    for (i,j) in combinations(range(len(cluster_ids)),2):
        d = float(np.linalg.norm(acoes[i] - acoes[j]))
        dx.append(d)
    dx = np.array(dx)
    # limiar para "ações diferentes": dist > 0,30 (empírico, ~30% do range total dos perfis)
    threshold = 0.30
    row["pct_pares_acoes_diferentes"] = float((dx > threshold).sum()) / len(dx)
    row["dist_acao_media"] = float(dx.mean())

    # -------- CRITÉRIO 3: Estável --------
    # (a) shuffled rows
    rng = np.random.default_rng(SEED)
    shuf = []
    for s in range(5):
        idx = rng.permutation(len(X))
        l_s = kmeans_labels(X[idx], k, random_state=SEED+s)
        # reindexar de volta pro índice original
        l_s_orig = np.empty_like(l_s)
        l_s_orig[idx] = l_s
        shuf.append(adjusted_rand_score(l_ref, l_s_orig))
    row["ari_shuffle_medio"] = float(np.mean(shuf))
    row["ari_shuffle_min"] = float(np.min(shuf))

    # (b) bootstrap 80%
    boots = []
    for s in range(10):
        idx_sample = resample(np.arange(len(X)), replace=False,
                              n_samples=int(0.8*len(X)), random_state=SEED+s)
        Xb = X[idx_sample]
        lb_ref = l_ref[idx_sample]
        lb_new = kmeans_labels(Xb, k, random_state=SEED+s)
        boots.append(adjusted_rand_score(lb_ref, lb_new))
    row["ari_bootstrap_medio"] = float(np.mean(boots))
    row["ari_bootstrap_min"] = float(np.min(boots))

    # (c) random_states variados
    rs_list = [42, 123, 456, 789, 1024]
    rands = []
    for rs in rs_list:
        l_alt = kmeans_labels(X, k, random_state=rs)
        rands.append(adjusted_rand_score(l_ref, l_alt))
    row["ari_randstate_medio"] = float(np.mean(rands))
    row["ari_randstate_min"] = float(np.min(rands))

    # agregado
    row["estavel_nota_media"] = float(np.mean([
        row["ari_shuffle_medio"],
        row["ari_bootstrap_medio"],
        row["ari_randstate_medio"]
    ]))

    # -------- CRITÉRIO 4: Melhor que baseline --------
    # R² ANOVA médio da segmentação vs baselines tipo_loja e sortimento.
    # "Vitória" = R² modelo > R² baseline
    anova_alvos = ["venda_media_dia","ticket_medio","clientes_medio_dia",
                   "razao_promo","cv_volatilidade_diaria","cv_sazonalidade_mensal",
                   "distancia_concorrente"]
    model_r2 = np.array([r2_1way(df[y], df[col]) for y in anova_alvos])
    bl1_r2 = np.array([r2_1way(df[y], df["tipo_loja"]) for y in anova_alvos])
    bl2_r2 = np.array([r2_1way(df[y], df["sortimento"]) for y in anova_alvos])
    row["r2_modelo_medio"] = float(model_r2.mean())
    row["r2_tipo_loja_medio"] = float(bl1_r2.mean())
    row["r2_sortimento_medio"] = float(bl2_r2.mean())
    row["delta_vs_tipo_loja"] = float((model_r2 - bl1_r2).mean())
    row["pct_vence_tipo_loja"] = float((model_r2 > bl1_r2).mean())
    row["pct_vence_sortimento"] = float((model_r2 > bl2_r2).mean())

    # Captura do tipo_loja b: qual % das 17 lojas b cai no mesmo cluster dominante?
    tb = df[df["tipo_loja"]=="b"][col].value_counts(normalize=True).iloc[0]
    row["captura_b_concentracao"] = float(tb)

    # -------- CRITÉRIO 5: Quantificado --------
    # Gap total em R$ (oportunidade) dentro de cada cluster.
    # MÉTODO: para cada cluster, defina o "limiar do par" = P25 do cluster (25% piores do grupo).
    # Para toda loja abaixo do limiar no seu cluster, some (limiar - venda_media_dia) * 365 dias.
    # Total em R$/ano.
    gap_total = 0.0
    for cid in cluster_ids:
        sub = df[df[col] == cid]["venda_media_dia"].values
        p25 = float(np.quantile(sub, 0.25))
        below = sub[sub < p25]
        gap_total += float(np.sum(p25 - below) * 365)
    row["oportunidade_ano_R$"] = gap_total
    row["oportunidade_por_loja_R$"] = gap_total / len(df)

    # -------- EXTRA: tamanho de menor cluster (restringe governança) --------
    sizes = df[col].value_counts()
    row["menor_cluster_n"] = int(sizes.min())
    row["menor_cluster_pct"] = float(sizes.min() / len(df) * 100)

    results.append(row)

tab_final = pd.DataFrame(results)
print(tab_final.to_string(index=False, float_format=lambda v: f"{v:,.2f}" if v >= 100 else f"{v:.4f}"))
tab_final.to_csv(DATA_PROC / "fase4_k_vs_5_criterios_sucesso.csv", index=False, float_format="%.4f")
print(f"\n> Tabela salva em: fase4_k_vs_5_criterios_sucesso.csv")

# Tabela transposta comparativa (linhas = critérios, colunas = k)
cols_display = {
    "k": "k",
    "r2_interpret_medio": "C1 Interpre·R² médio",
    "pct_pares_acoes_diferentes": "C2 Acionável·%pares dist.",
    "dist_acao_media": "C2 Dist·ação média",
    "estavel_nota_media": "C3 Estável·ARI médio",
    "pct_vence_tipo_loja": "C4 %vence tipo_loja",
    "delta_vs_tipo_loja": "C4 ΔR² médio vs tipo",
    "captura_b_concentracao": "C4 captura tipo_b",
    "oportunidade_ano_R$": "C5 Oportunidade R$/ano",
    "menor_cluster_pct": "menor_cluster %",
}
disp = tab_final[list(cols_display)].rename(columns=cols_display)
print("\n==== TABELA COMPARATIVA (linhas=k; colunas=critérios) ====")
print(disp.to_string(index=False))
