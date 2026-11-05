"""
Vendored copy of the Kronos model package (github.com/shiyu-coder/Kronos, MIT licence, see
LICENSE). Only `model/kronos.py` and `model/module.py` are included, unchanged apart from
the relative import, so the forecaster works without a separate clone. Weights are
downloaded from Hugging Face (NeoQuasar/Kronos-*) on first use. Requires torch, einops,
huggingface_hub, safetensors: see requirements-kronos.txt.
"""
from .kronos import Kronos, KronosPredictor, KronosTokenizer

__all__ = ["Kronos", "KronosPredictor", "KronosTokenizer"]
