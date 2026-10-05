# -*- coding: utf-8 -*-
"""
predict.py — Classifica novas lojas no modelo k5_v2_prof6 SEM RETREINO.
======================================================================
Modelo NUNCA é reajustado com dados novos. Só usamos os parâmetros
exportados (scaler mean/std + 5 centróides KMeans + mapeamento C01..C03).

Duas rotas suportadas (ambas dão resultado IDÊNTICO):
  Rota A (RECOMENDADA para quem TEM sklearn/scipy instalados):
      carrega scaler_final_v2_prof6.pkl + kmeans_final_k5_v2_prof6.pkl
      roda X = features novas → scaler.transform → kmeans.predict

  Rota B (NUMPy PURO, sem sklearn — ideal para Google Colab, ambientes
      minimais, ou para portar para JS / Excel / qualquer linguagem):
      carrega params_modelo.json → faz log1p → (x-mean)/scale →
      distância Euclidiana a cada centróide → argmin.

ENTRADA (uma linha por loja nova, CSV ou DataFrame):
  colunas OBRIGATÓRIAS raw (ANTES das transformações):
      id_loja              (opcional, só pra te ajudar a identificar)
      nome_loja            (opcional)
      clientes_medio_dia      ≥ 0  float (média de clientes por dia operando)
      ticket_medio            > 0  float (R$)
      razao_promo             ≥ 0  float (venda_promo / venda_sem_promo)
      cv_volatilidade_diaria  ≥ 0  float (std(vendas_dia)/média(vendas_dia))
      cv_sazonalidade_mensal  ≥ 0  float (CV da venda média por mês)
      distancia_concorrente   ≥ 0  float ou VAZIO (metros, nulo → mediana 2.325 m)

USO RÁPIDO (linha de comando):
    # Crie um CSV com as 6 features raw (uma linha por loja)
    # Ex.: deploy/exemplos/teste_3_lojas_novas.csv
    python deploy/predict.py deploy/exemplos/teste_3_lojas_novas.csv
    # → salva saida em deploy/predicoes_YYYYMMDD_HHMM.csv

USO RÁPIDO (Python import):
    from deploy.predict import classificar_novas_lojas
    resultado = classificar_novas_lojas(df_ou_csv)

Validação ARI 1.000000 vs 1.115 lojas de referência.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================
# Encontra ROOT do projeto (mesma regra do preâmbulo dos notebooks)
# ================================================================
_ESTE_DIR = Path(__file__).resolve().parent
_ROOT_CANDIDATOS = [_ESTE_DIR, _ESTE_DIR.parent, Path.cwd().resolve()]
ROOT = next((p for p in _ROOT_CANDIDATOS if (p / "pyproject.toml").exists()), _ESTE_DIR)

DEPLOY = _ESTE_DIR  # = ROOT/deploy
MODELS = ROOT / "models"
ARQ_PARAMS = DEPLOY / "params_modelo.json"
ARQ_SCALER_PKL = MODELS / "scaler_final_v2_prof6.pkl"
ARQ_KMEANS_PKL = MODELS / "kmeans_final_k5_v2_prof6.pkl"

# ================================================================
# 1. Carregar parâmetros do modelo
# ================================================================
def _carregar_params():
    if not ARQ_PARAMS.exists():
        raise FileNotFoundError(
            f"Arquivo {ARQ_PARAMS} não encontrado. Rode `python _exporta_parametros_modelo.py`"
            f"na raiz do projeto primeiro."
        )
    with open(ARQ_PARAMS, "r", encoding="utf-8") as f:
        p = json.load(f)
    FS = p["standard_scaler"]["ordem_features"]
    mean_ = np.array(p["standard_scaler"]["mean_"], dtype=float)
    scale_ = np.array(p["standard_scaler"]["scale_"], dtype=float)
    centers = np.array(p["kmeans"]["cluster_centers_scaled_ordem_kmeans0a4"], dtype=float)
    clusters = p["clusters"]
    mediana_dist = float(p["features"]["mediana_imputacao_distancia_concorrente_metros"])
    return FS, mean_, scale_, centers, clusters, mediana_dist


# ================================================================
# 2. Pipeline numpy puro (Rota B — SEM sklearn)
# ================================================================
COLS_RAW_OBRIGATORIAS = [
    "clientes_medio_dia",
    "ticket_medio",
    "razao_promo",
    "cv_volatilidade_diaria",
    "cv_sazonalidade_mensal",
    "distancia_concorrente",
]


def _transformar_features_raw(df_raw: pd.DataFrame, mediana_dist: float) -> np.ndarray:
    """Recebe DataFrame com colunas raw (não transformadas) e retorna matriz
    X (n_lojas x 6) já com log1p e imputação, na ordem do scaler."""
    df = df_raw.copy()
    faltam = [c for c in COLS_RAW_OBRIGATORIAS if c not in df.columns]
    if faltam:
        raise ValueError(
            f"Faltam colunas obrigatórias no CSV/DataFrame: {faltam}.\n"
            f"Colunas esperadas: {COLS_RAW_OBRIGATORIAS}"
        )
    for c in ["clientes_medio_dia", "ticket_medio", "razao_promo",
              "cv_volatilidade_diaria", "cv_sazonalidade_mensal"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["distancia_concorrente"] = pd.to_numeric(df["distancia_concorrente"], errors="coerce")
    # Trata string vazia / whitespace como NaN (caso comum em CSVs — se o separador ; deixar um campo vazio).
    # Isso garante que a imputação mediana seja aplicada tanto para NaN quanto para campos vazios.
    for c in ["clientes_medio_dia", "ticket_medio", "razao_promo",
              "cv_volatilidade_diaria", "cv_sazonalidade_mensal", "distancia_concorrente"]:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Imputação distância
    nulos_dist = int(df["distancia_concorrente"].isna().sum())
    if nulos_dist > 0:
        df["distancia_concorrente"] = df["distancia_concorrente"].fillna(mediana_dist)
    # Clip por segurança (evita log1p(negativo))
    for c in ["clientes_medio_dia", "cv_sazonalidade_mensal", "distancia_concorrente"]:
        df[c] = df[c].clip(lower=0.0)
    df["razao_promo"] = df["razao_promo"].clip(lower=0.0)
    df["ticket_medio"] = df["ticket_medio"].clip(lower=1e-6)

    X = np.empty((len(df), 6), dtype=float)
    X[:, 0] = np.log1p(df["clientes_medio_dia"].values)
    X[:, 1] = df["ticket_medio"].values
    X[:, 2] = df["razao_promo"].values
    X[:, 3] = df["cv_volatilidade_diaria"].values
    X[:, 4] = np.log1p(df["cv_sazonalidade_mensal"].values)
    X[:, 5] = np.log1p(df["distancia_concorrente"].values)
    return X


def _padronizar(X: np.ndarray, mean_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    return (X - mean_) / scale_


def _atribuir_cluster(X_scaled: np.ndarray, centers: np.ndarray, clusters_meta: dict):
    """Retorna (kmeans_label, ordem_negocio, nome, dist_min, dists_todas).
    Regra de segurança OUTLIER com 2 critérios (qualquer um dos dois → flag):
      A) dist_min > 3.0 * média_dists_essa_loja   (muito mais longe do que o resto)
      B) dist_min > 4.0 (universo padronizado; 99% das 1.115 lojas originais estão < 4.0)
    """
    dists = np.sqrt(np.sum((X_scaled[:, None, :] - centers[None, :, :]) ** 2, axis=2))  # (n, 5)
    labels = np.argmin(dists, axis=1)
    dists_min = dists[np.arange(len(labels)), labels]
    media_dists_por_loja = dists.mean(axis=1)
    limiar_A = 3.0 * media_dists_por_loja
    LIMIAR_B_ABSOLUTO = 4.0
    flag_outlier = (dists_min > limiar_A) | (dists_min > LIMIAR_B_ABSOLUTO)

    ordens = []
    nomes = []
    for lab in labels:
        meta = clusters_meta[str(int(lab))]
        ordens.append(meta["ordem_negocio"])
        nomes.append(meta["nome"])
    return labels, np.array(ordens), np.array(nomes), dists, dists_min, media_dists_por_loja, flag_outlier


# ================================================================
# 3. Rota A (opcional sklearn — carrega .pkl). Resultado deve = Rota B.
# ================================================================
def _rota_A_sklearn(X_raw_transformed: np.ndarray):
    """Retorna labels via scaler + KMeans .pkl originais. Se sklearn não existir
    ou os arquivos .pkl não existirem, retorna None."""
    try:
        import pickle
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except Exception:
        return None
    if not ARQ_SCALER_PKL.exists() or not ARQ_KMEANS_PKL.exists():
        return None
    try:
        with open(ARQ_SCALER_PKL, "rb") as f:
            scaler = pickle.load(f)
        with open(ARQ_KMEANS_PKL, "rb") as f:
            km = pickle.load(f)
        Xs = scaler.transform(X_raw_transformed)
        return km.predict(Xs)
    except Exception:
        return None


# ================================================================
# 4. Função principal de API
# ================================================================
def classificar_novas_lojas(
    entrada: str | Path | pd.DataFrame,
    arquivo_saida: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Classifica L novas lojas em 1 dos 5 clusters (SEM retreino).

    Parâmetros
    ----------
    entrada : str/Path/DataFrame
        Caminho CSV com as 6 colunas raw obrigatórias, ou então um
        pandas.DataFrame já carregado com as mesmas colunas.
    arquivo_saida : str/Path, opcional
        Onde salvar o CSV de saída. Se None, salva em
        deploy/predicoes_<timestamp>.csv.
    verbose : bool
        Imprime resumo.

    Retorna
    -------
    pd.DataFrame com colunas:
        [id_loja*, nome_loja*, ... colunas raw ...,
         kmeans_cluster_id, ordem_negocio (C01..C03), nome_perfil,
         distancia_ao_centroide, flag_outlier_espaco,
         dist_C01, dist_C04, dist_C00, dist_C02, dist_C03]
    """
    FS, mean_, scale_, centers, clusters_meta, mediana_dist = _carregar_params()

    # 4a) Carregar entrada
    if isinstance(entrada, (str, Path)):
        entrada = Path(entrada)
        if not entrada.exists():
            raise FileNotFoundError(f"Arquivo {entrada} não encontrado.")
        try:
            df = pd.read_csv(entrada, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(entrada, sep=None, engine="python", encoding="latin-1")
    elif isinstance(entrada, pd.DataFrame):
        df = entrada.copy()
    else:
        raise TypeError(f"entrada deve ser caminho CSV ou DataFrame, recebi {type(entrada)}")

    if verbose:
        print(f"[predict.py] {len(df)} loja(s) nova(s) para classificar.")

    # 4b) Extrai colunas de identificação se existirem
    cols_id = [c for c in df.columns if c.lower() in {"id_loja", "loja", "cod_loja", "id"}]
    cols_nome = [c for c in df.columns if c.lower() in {"nome_loja", "nome", "descricao"}]
    cols_id = cols_id[:1]
    cols_nome = cols_nome[:1]

    # 4c) Pipeline numpy puro
    X_raw = _transformar_features_raw(df, mediana_dist)

    # 4c.1) Salva colunas de auditoria: distância bruta (sempre numérica — imputada com mediana se faltava) + flag
    #       O _transformar_features_raw faz essa conta internamente; repetimos aqui para expor no CSV de saída.
    dist_bruta = pd.to_numeric(df["distancia_concorrente"], errors="coerce")
    if df["distancia_concorrente"].dtype == "object":
        dist_bruta2 = (
            df["distancia_concorrente"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "NULL": np.nan, "null": np.nan})
        )
        dist_bruta = pd.to_numeric(dist_bruta2, errors="coerce")
    flag_imputada_dist = dist_bruta.isna()
    dist_bruta_ok = dist_bruta.fillna(mediana_dist)

    X_scaled = _padronizar(X_raw, mean_, scale_)
    (
        labels_np,
        ordens,
        nomes,
        dists,
        dists_min,
        media_dists,
        flag_outlier,
    ) = _atribuir_cluster(X_scaled, centers, clusters_meta)

    # 4d) Compara com sklearn se disponível (sanity check).
    labels_sk = _rota_A_sklearn(X_raw)
    if labels_sk is not None:
        coincidem = np.mean(labels_np == labels_sk)
        if verbose:
            print(f"[predict.py] Consistência Rota A (sklearn .pkl) × Rota B (numpy puro):"
                  f" {100*coincidem:.1f}% igual (ARI esperado 1.0).")
        if coincidem < 0.999:
            warnings.warn(
                "Diferença > 0.1% entre rota sklearn e rota numpy puro. "
                "Verifique se os arquivos .pkl correspondem ao params_modelo.json."
            )

    # 4e) Monta DataFrame de saída
    saida = df.copy()
    saida["distancia_concorrente"] = dist_bruta_ok.values
    saida["distancia_foi_imputada"] = flag_imputada_dist.astype(bool).values
    saida["kmeans_cluster_id"] = labels_np.astype(int)
    saida["ordem_negocio"] = ordens
    saida["nome_perfil"] = nomes
    saida["distancia_ao_centroide_escalado"] = np.round(dists_min, 4)
    saida["media_distancia_a_todos_5_centroides"] = np.round(media_dists, 4)
    saida["flag_outlier_fora_do_espaco"] = flag_outlier.astype(bool)
    # Dists por cluster ordenado por ordem_negocio
    ordemsorted = sorted(clusters_meta.keys(), key=lambda k: clusters_meta[k]["ordem_negocio"])
    for chave_str in ordemsorted:
        k_id = int(chave_str)
        ordem = clusters_meta[chave_str]["ordem_negocio"]
        saida[f"dist_{ordem}"] = np.round(dists[:, k_id], 4)
    # Reordena
    cols_finais = (
        cols_id + cols_nome +
        [c for c in COLS_RAW_OBRIGATORIAS] +
        ["distancia_foi_imputada"] +
        ["kmeans_cluster_id", "ordem_negocio", "nome_perfil",
         "distancia_ao_centroide_escalado",
         "media_distancia_a_todos_5_centroides",
         "flag_outlier_fora_do_espaco"] +
        [f"dist_{clusters_meta[str(k)]['ordem_negocio']}" for k in range(5)]
    )
    cols_finais = [c for c in cols_finais if c in saida.columns]
    saida = saida[cols_finais]

    # 4f) Salva CSV
    if arquivo_saida is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arquivo_saida = DEPLOY / f"predicoes_{ts}.csv"
    arquivo_saida = Path(arquivo_saida)
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(arquivo_saida, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"[predict.py] {len(df)} predições salvas em:\n  -> {arquivo_saida}")
        print("\nResumo da classificação:")
        resumo = (
            saida.groupby(["ordem_negocio", "nome_perfil"])
            .size().reset_index(name="n_lojas_novas")
            .sort_values("ordem_negocio")
        )
        print(resumo.to_string(index=False))
        if saida["flag_outlier_fora_do_espaco"].any():
            print(f"\n⚠️  {int(saida['flag_outlier_fora_do_espaco'].sum())} loja(s) foram marcadas "
                  f"como FORA DO ESPAÇO DE REFERÊNCIA (distância ao centroide > 3× a média de distâncias "
                  f"dela aos 5 centróides). Recomenda-se validação manual.")
    return saida


# ================================================================
# 5. CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="Classifica lojas novas no modelo k5_v2_prof6.")
    ap.add_argument("csv_entrada", help="Caminho CSV com as 6 colunas raw obrigatórias.")
    ap.add_argument("-o", "--saida", help="Caminho CSV de saída (opcional).")
    ap.add_argument("-q", "--quiet", action="store_true", help="Não imprimir resumo.")
    args = ap.parse_args()

    try:
        classificar_novas_lojas(
            entrada=args.csv_entrada,
            arquivo_saida=args.saida,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"[ERRO] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
