# -*- coding: utf-8 -*-
"""Testes do pipeline de inferência de lojas novas — deploy/predict.py.

Garantem que a classificação é DETERMINÍSTICA e reprodutível a partir dos
parâmetros congelados (params_modelo.json), sem retreino do modelo, usando
os 3 CSVs de exemplo versionados em deploy/exemplos/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
EXEMPLOS = DEPLOY / "exemplos"


def _carregar_predict():
    """Importa deploy/predict.py por caminho (deploy/ não é um pacote instalável)."""
    if "predict_deploy" in sys.modules:
        return sys.modules["predict_deploy"]
    spec = importlib.util.spec_from_file_location("predict_deploy", DEPLOY / "predict.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["predict_deploy"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


predict = _carregar_predict()


@pytest.fixture
def classificar(tmp_path):
    """Classifica um CSV de exemplo salvando a saída em tmp_path (não polui deploy/)."""

    def _run(nome_csv: str):
        return predict.classificar_novas_lojas(
            entrada=EXEMPLOS / f"{nome_csv}.csv",
            arquivo_saida=tmp_path / f"{nome_csv}_saida.csv",
            verbose=False,
        )

    return _run


# ------------------------------------------------------------------
# Parâmetros do modelo (fonte única da verdade)
# ------------------------------------------------------------------
def test_params_modelo_estrutura():
    with open(DEPLOY / "params_modelo.json", "r", encoding="utf-8") as f:
        p = json.load(f)
    assert p["versao_modelo"] == "k5_v2_prof6"
    assert p["seed"] == 42
    assert p["k"] == 5
    assert len(p["standard_scaler"]["mean_"]) == 6
    assert len(p["standard_scaler"]["scale_"]) == 6
    assert len(p["kmeans"]["cluster_centers_scaled_ordem_kmeans0a4"]) == 5
    assert p["features"]["mediana_imputacao_distancia_concorrente_metros"] == 2325


# ------------------------------------------------------------------
# Caso 1 — 3 perfis típicos (nenhum outlier, nenhuma imputação)
# ------------------------------------------------------------------
def test_teste1_perfis_tipicos(classificar):
    saida = classificar("teste1_3_perfis_tipicos")
    assert list(saida["ordem_negocio"]) == ["C01", "C00", "C02"]
    assert not saida["distancia_foi_imputada"].any()
    assert not saida["flag_outlier_fora_do_espaco"].any()


# ------------------------------------------------------------------
# Caso 2 — distância do concorrente vazia → imputada na mediana (2.325 m)
# A loja é "Nova Premium" e o CSV de exemplo espera o cluster C00; como a
# imputação usa a MEDIANA (valor central), ela NÃO pode virar outlier.
# ------------------------------------------------------------------
def test_teste2_imputacao_distancia(classificar):
    saida = classificar("teste2_imputacao_distancia_vazia")
    linha = saida.iloc[0]
    assert bool(linha["distancia_foi_imputada"]) is True
    assert float(linha["distancia_concorrente"]) == pytest.approx(2325.0)
    assert linha["ordem_negocio"] == "C00"
    assert bool(linha["flag_outlier_fora_do_espaco"]) is False


# ------------------------------------------------------------------
# Caso 3 — lote de 8 lojas, com 1 outlier declarado (id 2099)
# ------------------------------------------------------------------
def test_teste3_lote_e_outlier(classificar):
    saida = classificar("teste3_lote8_e_outlier")
    assert list(saida["ordem_negocio"]) == [
        "C03", "C04", "C04", "C02", "C02", "C00", "C01", "C03",
    ]
    # Apenas a última loja (id 2099, "OUTLIER") é sinalizada.
    assert list(saida["flag_outlier_fora_do_espaco"]) == [
        False, False, False, False, False, False, False, True,
    ]
    assert int(saida.iloc[-1]["id_loja"]) == 2099


# ------------------------------------------------------------------
# Determinismo — mesma entrada ⇒ mesma saída
# ------------------------------------------------------------------
def test_determinismo(classificar):
    a = classificar("teste3_lote8_e_outlier")
    b = classificar("teste3_lote8_e_outlier")
    colunas = ["kmeans_cluster_id", "ordem_negocio", "nome_perfil",
               "distancia_ao_centroide_escalado", "flag_outlier_fora_do_espaco"]
    assert a[colunas].equals(b[colunas])


# ------------------------------------------------------------------
# Colunas obrigatórias ausentes ⇒ ValueError com mensagem clara
# ------------------------------------------------------------------
def test_colunas_obrigatorias_ausentes(tmp_path):
    csv_incompleto = tmp_path / "incompleto.csv"
    csv_incompleto.write_text("id_loja;clientes_medio_dia\n1;500\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltam colunas obrigatórias"):
        predict.classificar_novas_lojas(entrada=csv_incompleto, verbose=False)


# ------------------------------------------------------------------
# Saída é persistida em CSV com as colunas-chave
# ------------------------------------------------------------------
def test_csv_saida_gerado(classificar, tmp_path):
    classificar("teste1_3_perfis_tipicos")
    arq = tmp_path / "teste1_3_perfis_tipicos_saida.csv"
    assert arq.exists()
    conteudo = arq.read_text(encoding="utf-8-sig").splitlines()[0]
    for coluna in ["ordem_negocio", "nome_perfil", "distancia_ao_centroide_escalado",
                   "flag_outlier_fora_do_espaco"]:
        assert coluna in conteudo
