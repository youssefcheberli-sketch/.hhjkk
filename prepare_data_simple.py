import os
import random
import shutil
from pathlib import Path

random.seed(42)  # même mélange à chaque lancement

# Base du projet : dossier contenant ce script (évite les problèmes de chemin courant)
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"

if not RAW_DIR.exists() or not RAW_DIR.is_dir():
    raise FileNotFoundError(
        f"Le dossier source '{RAW_DIR}' est introuvable. "
        "Créez 'raw/<classe>/' et placez-y vos images."
    )

# On repart de zéro si le dossier data existe déjà
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)

# Pour chaque classe (facture, cv, ...)
for classe in sorted(os.listdir(RAW_DIR)):
    dossier_classe = RAW_DIR / classe
    if not dossier_classe.is_dir():
        continue  # on ignore les fichiers qui ne sont pas des dossiers

    # 1. Lister les images de la classe
    images = [
        f
        for f in os.listdir(dossier_classe)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # 2. Les mélanger
    random.shuffle(images)

    # 3. Calculer les tailles : 70 % / 15 % / reste
    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    # 4. Couper la liste en 3 (train -> data/train)
    groupes = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    # 5. Copier les images dans data/<groupe>/<classe>/
    for nom_groupe, liste in groupes.items():
        destination = DATA_DIR / nom_groupe / classe
        destination.mkdir(parents=True, exist_ok=True)
        for image in liste:
            shutil.copy(dossier_classe / image, destination / image)

    print(
        f"{classe} : train = {len(groupes['train'])}"
        f" | validation = {len(groupes['val'])}"
        f" | test = {len(groupes['test'])}"
    )

print(f"\nTerminé ! Données générées dans : {DATA_DIR / 'train'}")