"""Clusterização de Lojas de Varejo — modelo k5_v2_prof6 (KMeans k=5, seed 42).

O pipeline de inferência de lojas novas vive em ``deploy/predict.py`` (NumPy puro,
sem retreino). Este pacote existe apenas para empacotar/instalar o projeto via
``uv_build`` e expor o console script ``retail-store-clustering``.
"""
from __future__ import annotations

__version__ = "1.0.1"


def main() -> None:
    """Console script: aponta para o classificador de lojas novas."""
    print(
        "retail-store-clustering v"
        f"{__version__}\n"
        "Classificador de lojas novas -> deploy/predict.py\n"
        "  uso:  python deploy/predict.py <csv_entrada> [-o saida.csv]\n"
        "  app :  deploy/index.html (GitHub Pages)"
    )
