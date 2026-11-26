import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms

# 1. Paramètres et chemins
DATA_TEST_DIR = Path("data/test")
MODEL_PATH = Path("modele.pt")
CLASSES_PATH = Path("classes.json")
OUTPUT_IMG = Path("erreurs_analyse.png")
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Chargement des classes
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

# 3. Prétraitement identique au test (sans augmentation)
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset sans transformation pour garder l'image originale à afficher
dataset_raw = datasets.ImageFolder(DATA_TEST_DIR)
dataset_eval = datasets.ImageFolder(DATA_TEST_DIR, transform=transform_test)

# 4. Chargement du modèle MobileNetV3-small
model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(in_features, len(classes))
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# 5. Collecte des erreurs
erreurs = []

with torch.no_grad():
    for idx in range(len(dataset_eval)):
        img_tensor, true_idx = dataset_eval[idx]
        img_pil, _ = dataset_raw[idx]
        img_path, _ = dataset_eval.samples[idx]

        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        confiance, pred_idx = torch.max(probs, dim=0)
        true_label = classes[true_idx]
        pred_label = classes[pred_idx.item()]

        if pred_idx.item() != true_idx:
            erreurs.append({
                "confiance": float(confiance.item()),
                "vrai": true_label,
                "predit": pred_label,
                "chemin": img_path,
                "image": img_pil,
            })

# Tri décroissant par score de confiance (les plus "sûres d'elles" en premier)
erreurs.sort(key=lambda x: x["confiance"], reverse=True)

print(f"Nombre total d'images de test : {len(dataset_eval)}")
print(f"Nombre d'erreurs : {len(erreurs)}\n")

for i, err in enumerate(erreurs, start=1):
    print(
        f"{i}. Fichier={Path(err['chemin']).name} | "
        f"Réel={err['vrai']} | Prédit={err['predit']} | "
        f"Confiance={err['confiance']:.2%}"
    )

# 6. Sauvegarde d'une planche visuelle des erreurs
if erreurs:
    top_erreurs = erreurs[:6]
    n = len(top_erreurs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, err in zip(axes, top_erreurs):
        ax.imshow(err["image"])
        ax.set_title(
            f"Réel : {err['vrai']}\nPrédit : {err['predit']} ({err['confiance']:.1%})",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"\nImage enregistrée : {OUTPUT_IMG}")
else:
    print("\nAucune erreur sur l'ensemble de test.")