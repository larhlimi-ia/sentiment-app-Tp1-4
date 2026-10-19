"""Raccord entre l'API et le modèle entraîné au TP2.

L'API n'entraîne rien et ne connaît pas scikit-learn : elle appelle la fonction
predict() de src/predict.py, qui charge models/sentiment.joblib une seule fois.
TP4, étape C : sous le seuil d'abstention, l'API répond INDETERMINE au lieu de deviner.
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from predict import MODELE, charger_modele, predict as _predict  # noqa: E402  (fonctions du TP2)

# Seuil d'abstention, choisi avec src/courbe_seuil.py sur le jeu de démonstration (58 avis de test) :
# à 0,60, l'API répond à 42 avis (72 %), avec 2 erreurs au lieu de 7 ; les 16 autres vont à un humain.
SEUIL_ABSTENTION = 0.60


class SentimentClassifier:
    """Enveloppe du modèle TP2 avec l'interface attendue par l'API."""

    model_name = "tfidf-logreg (TP2, models/sentiment.joblib)"

    def __init__(self) -> None:
        charger_modele()  # échoue immédiatement si le joblib manque : on ne démarre pas sans modèle

    def predict(self, text: str) -> dict:
        """Renvoie {'label': 'POSITIVE'|'NEGATIVE'|'INDETERMINE', 'score': float, 'text': aperçu}."""
        r = _predict(text)  # lève ValueError si le texte est vide
        # Sous le seuil, l'API s'abstient : le score reste renvoyé, l'humain tranche
        label = r["label"].upper() if r["score"] >= SEUIL_ABSTENTION else "INDETERMINE"
        return {"label": label, "score": r["score"], "text": text[:100]}


def load_model() -> SentimentClassifier:
    """Construit le classifier (le cache de chargement est dans src/predict.py)."""
    return SentimentClassifier()
