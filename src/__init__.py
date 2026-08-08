"""Sarcasm-aware sentiment analysis for Serbian — glavni paket.

Mapa podpaketa:
  common/         — konfiguracija, šema, I/O, jezik, training flagovi
  collection/     — kolektori po izvoru (YouTube, Reddit, …) + twitter_fetch
  preprocessing/  — čišćenje (dataset/BERTić) i agresivnije za baseline
  dataset/        — build annotation CSV, append po platformi, statistike
  baselines/      — TF-IDF + Naive Bayes / LR / Linear SVM
  modeling/       — BERTić fine-tune (sentiment / sarcasm / multitask)
  pipeline.py     — end-to-end: kolekcija → preprocess → dataset → stats

CLI ulazne tačke su u ``scripts/`` (tanki wrapperi oko ovog paketa).
"""

__version__ = "0.1.0"
