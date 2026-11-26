"""Partie 4 : évaluation du modèle sur l'ensemble de TEST.

Prérequis : avoir lancé entrainement.py (fichiers modele.pt et classes.json).
Usage     : python evaluation.py
Sorties   : accuracy, F1, rapport par classe, matrice_confusion.png
"""
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# Les images de test, avec les mêmes transformations que la validation (partie 2)
from pretraitement import donnees_test

APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Recharger le modèle entraîné ----------
classes = json.load(open("classes.json"))

# Même architecture qu'à l'entraînement, mais sans télécharger les poids ImageNet
modele = models.mobilenet_v3_small(weights=None)
modele.classifier[3] = nn.Linear(modele.classifier[3].in_features, len(classes))
# On charge les réglages appris (fichier créé par entrainement.py)
modele.load_state_dict(torch.load("modele.pt", map_location=APPAREIL))
modele.to(APPAREIL)
modele.eval()   # mode évaluation

# ---------- 2. Faire les prédictions sur le test ----------
chargeur_test = DataLoader(donnees_test, batch_size=8)

vraies, predites = [], []
with torch.no_grad():                       # pas d'apprentissage ici
    for images, etiquettes in chargeur_test:
        sorties = modele(images.to(APPAREIL))
        predites += sorties.argmax(1).cpu().tolist()   # catégorie prévue
        vraies += etiquettes.tolist()                  # vraie catégorie

# ---------- 3. Les métriques ----------
print(f"Images de test : {len(vraies)}")
print(f"Accuracy : {accuracy_score(vraies, predites):.3f}")
print(f"F1-score (moyenne des classes) : {f1_score(vraies, predites, average='macro'):.3f}")
print()
print(classification_report(vraies, predites, target_names=classes, digits=3))

# ---------- 4. La matrice de confusion ----------
matrice = confusion_matrix(vraies, predites)
print("Matrice de confusion (lignes = vraie catégorie, colonnes = catégorie prévue) :")
print(matrice)

ConfusionMatrixDisplay(matrice, display_labels=classes).plot(cmap="Blues", xticks_rotation=30)
plt.title("Matrice de confusion (ensemble de test)")
plt.tight_layout()
plt.savefig("matrice_confusion.png", dpi=150)
print("Matrice enregistrée : matrice_confusion.png")
plt.show()   # ouvre une fenêtre (fermez-la pour terminer le programme)
