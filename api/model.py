"""Raccord entre l'API et le modèle entraîné au TP2.

L'API n'entraîne rien et ne connaît pas scikit-learn : elle appelle la fonction
predict() de src/predict.py, qui charge models/sentiment.joblib une seule fois.
"""
import sys
from pathlib import Path
from typing import TypedDict

from api.schemas import Label

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from predict import charger_modele  # noqa: E402  (fonctions du TP2)
from predict import predict as _predict  # noqa: E402

# Seuil d'abstention, choisi avec src/courbe_seuil.py sur le jeu de démonstration (58 avis de test) :
# à 0,60, l'API répond à 42 avis (72 %), avec 2 erreurs au lieu de 7 ; les 16 autres vont à un humain.
SEUIL_ABSTENTION = 0.60
# Étiquettes du modèle -> étiquettes du contrat : une correspondance explicite, vérifiée au démarrage
CORRESPONDANCE: dict[str, Label] = {"positive": "POSITIVE", "negative": "NEGATIVE"}


class Prediction(TypedDict):
    label: Label
    score: float
    text: str


class SentimentClassifier:
    """Enveloppe du modèle TP2 avec l'interface attendue par l'API."""

    model_name = "tfidf-logreg (TP2, models/sentiment.joblib)"

    def __init__(self) -> None:
        modele = charger_modele()  # FileNotFoundError si le joblib manque : on ne démarre pas
        inconnues = {str(c) for c in modele.classes_} - set(CORRESPONDANCE)
        if inconnues:  # un modèle réentraîné sur d'autres étiquettes casserait le contrat
            raise ValueError(f"classes inconnues du contrat : {sorted(inconnues)}")

    def predict(self, text: str) -> Prediction:
        """Renvoie l'étiquette du contrat, ou INDETERMINE quand le modèle hésite."""
        r = _predict(text)
        label: Label = "INDETERMINE" if r["score"] < SEUIL_ABSTENTION else CORRESPONDANCE[r["label"]]
        return {"label": label, "score": r["score"], "text": text[:100]}


def load_model() -> SentimentClassifier:
    """Construit le classifier (le cache de chargement est dans src/predict.py)."""
    return SentimentClassifier()
