# -*- coding: utf-8 -*-
"""
FASE 5 - Onde caíram as 17 lojas tipo_loja b?
Análise de concentração, pureza, perfil.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "03_processed"

tab = pd.read_csv(DATA_PROC / "tabela_lojas_com_cluster_final_v2_prof6.csv")
assert len(tab) == 1115

# Filtrar as 17 lojas tipo_b
tb = tab[tab["tipo_loja"] == "b"].copy()
print(f"> Total de lojas tipo_loja b = {len(tb)}")
print()

# ================== DISTRIBUIÇÃO POR CLUSTER ==================
print("===== 1. Onde as 17 lojas tipo_b caíram (contagem e %) =====")
dist = (tb.groupby("cluster_final_v2")["loja"]
          .agg(n_tipo_b="count")
          .sort_values("n_tipo_b", ascending=False))
dist["% dentro_do_tipo_b"] = (dist["n_tipo_b"] / len(tb) * 100).round(1)
# Incluir %_do_cluster que é tipo_b (pureza)
total_por_cluster = tab.groupby("cluster_final_v2")["loja"].count().rename("n_total_cluster")
dist = dist.join(total_por_cluster)
dist["%_do_cluster_que_é_tipo_b"] = (dist["n_tipo_b"] / dist["n_total_cluster"] * 100).round(1)
print(dist.to_string())
print()

# Concentração medida: qual a proporção do tipo_b que cai no MESMO cluster (o maior)
maior_cluster_puro = dist.iloc[0]
print(f"> Concentração top-1 cluster: {maior_cluster_puro['% dentro_do_tipo_b']:.1f}% "
      f"({maior_cluster_puro['n_tipo_b']} de 17)")
print(f"> Concentração top-2 clusters: "
      f"{dist['% dentro_do_tipo_b'].head(2).sum():.1f}% "
      f"({dist['n_tipo_b'].head(2).sum()} de 17)")
print(f"> Quantos clusters distintos receberam tipo_b? {len(dist)}")
print()

# ================== NÚMERO DE LOJAS POR CLUSTER E % TIPO_B (VISÃO INVERTIDA) ==================
print("===== 2. Visão por cluster inteiro (todas as 1.115 lojas) =====")
print(f"{'Cluster':<7} {'n_lojas':>7} {'% rede':>6} {'n_tipo_b':>10} {'% tipo_b no cluster':>20} {'Rótulo sugerido':<25}")
rotulos = {
    1: "Alto Fluxo / Perto concorr.",
    0: "Ticket Premium / Distante",
    4: "Standard / Média",
    2: "Promo-Dependent",
    3: "Sazonal Raro",
}
for cid in sorted(tab["cluster_final_v2"].unique()):
    sub = tab[tab["cluster_final_v2"] == cid]
    n = len(sub)
    nb = (sub["tipo_loja"] == "b").sum()
    pct_b = nb / n * 100
    print(f"C{cid:<6} {n:>7} {n/len(tab)*100:>6.2f}% {nb:>10} {pct_b:>20.1f}% {rotulos.get(int(cid), str(cid)):<25}")
print()

# ================== PERFIL INDIVIDUAL DAS 17 LOJAS TIPO_B ==================
cols_show = [
    "loja",
    "cluster_final_v2",
    "tipo_loja",
    "sortimento",
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
print("===== 3. Lista completa das 17 lojas tipo_b com seus valores =====")
tb_show = tb[cols_show].sort_values(["cluster_final_v2", "venda_media_dia"], ascending=[True, False])
print(tb_show.to_string(index=False, float_format=lambda v: (
    f"{v:>10,.0f}" if v > 1000 else (f"{v:>10.0f}" if v > 5 else (
        f"{v:>10.4f}" if v < 1 else f"{v:>10.2f}"
    ))
)))
print()

# ================== PERFIL: TIPO_B MÉDIO vs RESTO DA REDE vs C01 (cluster maior concentração) ==================
print("===== 4. Perfil médio: tipo_b vs RESTO_DA_REDE vs CLUSTER C01 (o que absorveu mais tipo_b) =====")
linhas = []
for mascara, nome in [(tab["tipo_loja"] == "b", "TIPO_B (17 lojas)"),
                       (tab["tipo_loja"] != "b", "Resto da rede (1.098)"),
                       (tab["cluster_final_v2"] == 1, "C01 - Alto Fluxo (193 lojas)"),
                       (tab["cluster_final_v2"] == 0, "C00 - Ticket Premium (287)"),
                       ]:
    sub = tab[mascara]
    linhas.append({
        "Grupo": nome,
        "VM_dia": sub["venda_media_dia"].mean(),
        "Cli_dia": sub["clientes_medio_dia"].mean(),
        "Ticket": sub["ticket_medio"].mean(),
        "Uplift_promo": (sub["razao_promo"].mean() - 1) * 100,
        "CV_vol": sub["cv_volatilidade_diaria"].mean(),
        "CV_saz": sub["cv_sazonalidade_mensal"].mean(),
        "Dist_concor_med": sub["distancia_concorrente"].median(),
        "Abre_domingo_pct": (sub["tx_domingos_abertos"] >= 0.5).mean() * 100,
        "PromoCont_pct": sub["promo_continua"].mean() * 100,
    })
perf = pd.DataFrame(linhas).set_index("Grupo")
print(perf.round(1).to_string())
