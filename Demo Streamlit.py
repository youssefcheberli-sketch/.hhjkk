import json
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = Path("modele.pt")
CLASSES_PATH = Path("classes.json")
IMG_SIZE = 224
SEUIL_CONFIANCE = 0.60

st.set_page_config(page_title="Classification de documents", layout="centered")
st.title("Classification de documents par image")
st.write("Déposez une image de document pour prédire sa catégorie et son score de confiance.")

@st.cache_resource
def charger_modele_et_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(classes))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes, device

transform_inference = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Fichiers 'modele.pt' ou 'classes.json' introuvables. Lancez d'abord l'entraînement.")
    st.stop()

model, classes, device = charger_modele_et_classes()

fichier = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if fichier is not None:
    image = Image.open(fichier).convert("RGB")
    st.image(image, caption="Document déposé", use_container_width=True)

    tensor_img = transform_inference(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    score_max, idx_pred = torch.max(probs, dim=0)
    categorie_predite = classes[idx_pred.item()]
    score_val = float(score_max.item())

    st.subheader(f"Catégorie prévue : {categorie_predite}")
    st.metric("Score de confiance", f"{score_val:.1%}")

    if score_val < SEUIL_CONFIANCE:
        st.warning(
            f"Score faible (moins de {SEUIL_CONFIANCE:.0%}) : la prédiction est incertaine. "
            "Vérifiez manuellement le document ou testez une image plus nette."
        )

    st.write("Probabilités par catégorie :")
    scores_dict = {classes[i]: round(float(probs[i].item()), 4) for i in range(len(classes))}
    st.bar_chart(scores_dict)