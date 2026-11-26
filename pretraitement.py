"""Partie 2 : prétraitement (redimensionner, normaliser) et data augmentation.

Prérequis : avoir lancé prepare_data_simple.py (dossiers data/train, data/val, data/test).
Usage     : python pretraitement.py   (affiche un petit contrôle des données)
"""
from torchvision import transforms, datasets

# Moyenne et écart-type d'ImageNet (une valeur par couleur : rouge, vert, bleu)
MOYENNE = [0.485, 0.456, 0.406]
ECART = [0.229, 0.224, 0.225]

# --- Entraînement : redimensionner + augmentation + normaliser ---
transform_entrainement = transforms.Compose([
    transforms.Resize((224, 224)),                     # toutes les images en 224 x 224
    transforms.RandomRotation(5, fill=255),            # rotation au hasard entre -5 et +5 degrés
    transforms.RandomAffine(0, translate=(0.03, 0.03),  # décalage de 3 % au maximum
                            scale=(0.95, 1.05),         # zoom entre 95 % et 105 %
                            fill=255),                  # coins vides remplis en blanc
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # luminosité et contraste +/- 30 %
    transforms.ToTensor(),                             # image -> nombres entre 0 et 1
    transforms.Normalize(MOYENNE, ECART),              # (x - moyenne) / écart-type
])

# --- Validation et test : redimensionner + normaliser, SANS augmentation ---
transform_evaluation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART),
])

# --- Lecture des dossiers créés à la partie 1 ---
# Chaque sous-dossier (cv, facture, ...) devient une classe.
donnees_entrainement = datasets.ImageFolder(
    "data/train", transform=transform_entrainement)
donnees_validation = datasets.ImageFolder(
    "data/val", transform=transform_evaluation)
donnees_test = datasets.ImageFolder(
    "data/test", transform=transform_evaluation)


if __name__ == "__main__":
    # Petit contrôle : classes, nombre d'images, forme d'une image transformée
    print("Classes :", donnees_entrainement.classes)
    print("Entraînement :", len(donnees_entrainement), "images")
    print("Validation   :", len(donnees_validation), "images")
    print("Test         :", len(donnees_test), "images")
    image, etiquette = donnees_entrainement[0]
    print("Forme d'une image :", tuple(image.shape), "(3 couleurs, 224, 224)")
    print("Étiquette de cette image :", donnees_entrainement.classes[etiquette])
