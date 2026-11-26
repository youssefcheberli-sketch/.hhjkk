import ssl

# Placé tout en haut avant les imports réseau/PyTorch pour éviter SSLCertVerificationError[cite: 9, 11]
ssl._create_default_https_context = ssl._create_unverified_context

"""Partie 3 : transfer learning (MobileNetV3-small) + entraînement + courbes[cite: 11].

Prérequis : python "prepare_data_simple (1).py" puis pretraitement.py dans le même dossier[cite: 6, 11].
Usage     : python "entrainement (3).py"[cite: 4, 11]
Sorties   : modele.pt (meilleur modèle), classes.json, courbes_loss_accuracy.png[cite: 11]
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# On réutilise les données et les transformations de la partie 2[cite: 11]
from pretraitement import donnees_entrainement, donnees_validation

BASE_DIR = Path(__file__).resolve().parent

# ---------- 1. Réglages ----------
NB_EPOCHS = 10          # nombre de passages sur toutes les images d'entraînement[cite: 11]
TAILLE_LOT = 8          # nombre d'images traitées ensemble[cite: 11]
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"   # carte graphique si dispo[cite: 11]
classes = donnees_entrainement.classes
print("Classes :", classes, "| appareil :", APPAREIL)

chargeur_entrainement = DataLoader(donnees_entrainement, batch_size=TAILLE_LOT, shuffle=True)
chargeur_validation = DataLoader(donnees_validation, batch_size=TAILLE_LOT)

# ---------- 2. Le modèle (transfer learning) ----------
# Réseau léger déjà entraîné sur ImageNet[cite: 11]
modele = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# On GÈLE tout le réseau : ses connaissances ne changent pas[cite: 11]
for parametre in modele.parameters():
    parametre.requires_grad = False

# On remplace la dernière couche par une nouvelle (une sortie par catégorie).
# Cette nouvelle couche n'est pas gelée : c'est la seule qui va apprendre[cite: 11].
modele.classifier[3] = nn.Linear(modele.classifier[3].in_features, len(classes))
modele.to(APPAREIL)

fonction_perte = nn.CrossEntropyLoss()                 # mesure l'erreur du modèle[cite: 11]
optimiseur = torch.optim.Adam(modele.classifier[3].parameters(), lr=1e-3)


# ---------- 3. Une epoch = un passage sur toutes les images ----------
def une_epoch(chargeur, entrainer):
    modele.train(entrainer)          # mode entraînement ou mode évaluation[cite: 11]
    if entrainer:
        modele.features.eval()       # la partie gelée reste en mode évaluation[cite: 11]
    somme_perte, bonnes, total = 0.0, 0, 0
    with torch.set_grad_enabled(entrainer):
        for images, etiquettes in chargeur:
            images, etiquettes = images.to(APPAREIL), etiquettes.to(APPAREIL)
            sorties = modele(images)                         # les scores du modèle[cite: 11]
            perte = fonction_perte(sorties, etiquettes)      # l'erreur[cite: 11]
            if entrainer:
                optimiseur.zero_grad()    # on efface les anciens calculs[cite: 11]
                perte.backward()          # on calcule comment corriger[cite: 11]
                optimiseur.step()         # on corrige la dernière couche[cite: 11]
            somme_perte += perte.item() * len(etiquettes)
            bonnes += (sorties.argmax(1) == etiquettes).sum().item()
            total += len(etiquettes)
    return somme_perte / total, bonnes / total   # perte moyenne, accuracy[cite: 11]


# ---------- 4. La boucle d'entraînement ----------
historique = {"perte_ent": [], "perte_val": [], "acc_ent": [], "acc_val": []}
meilleure_acc = -1.0

for epoch in range(1, NB_EPOCHS + 1):
    perte_ent, acc_ent = une_epoch(chargeur_entrainement, entrainer=True)
    perte_val, acc_val = une_epoch(chargeur_validation, entrainer=False)

    historique["perte_ent"].append(perte_ent)
    historique["perte_val"].append(perte_val)
    historique["acc_ent"].append(acc_ent)
    historique["acc_val"].append(acc_val)
    print(
        f"epoch {epoch:2d} | perte {perte_ent:.3f} / {perte_val:.3f} "
        f"| accuracy {acc_ent:.2f} / {acc_val:.2f}   (entraînement / validation)"
    )

    # On garde le meilleur modèle selon la validation[cite: 11]
    if acc_val >= meilleure_acc:
        meilleure_acc = acc_val
        torch.save(modele.state_dict(), BASE_DIR / "modele.pt")

with open(BASE_DIR / "classes.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print("Meilleure accuracy de validation :", round(meilleure_acc, 3))

# ---------- 5. Les courbes de loss et d'accuracy ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(historique["perte_ent"], label="entraînement")
axes[0].plot(historique["perte_val"], label="validation")
axes[0].set_title("Loss (perte)")
axes[1].plot(historique["acc_ent"], label="entraînement")
axes[1].plot(historique["acc_val"], label="validation")
axes[1].set_title("Accuracy")
for axe in axes:
    axe.set_xlabel("epoch")
    axe.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "courbes_loss_accuracy.png", dpi=150)
print("Courbes enregistrées : courbes_loss_accuracy.png")
plt.show()   # ouvre une fenêtre avec les courbes (fermez-la pour terminer le programme)[cite: 11]
