# Classification automatique de documents par intelligence artificielle

> **Mini-projet IA : Deep Learning, vision par ordinateur et interface web**

Une application qui **reconnaît la catégorie d'un document à partir de son image** (article scientifique, CV, facture ou lettre administrative) et affiche un score de confiance.

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Technologies et bibliothèques](#2-technologies-et-bibliothèques)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Exécution complète](#5-exécution-complète)
6. [Description des scripts](#6-description-des-scripts)
7. [Résultats et fichiers générés](#7-résultats-et-fichiers-générés)
8. [Conseils d'utilisation](#8-conseils-dutilisation)
9. [Problèmes fréquents](#9-problèmes-fréquents)
10. [Limites et améliorations possibles](#10-limites-et-améliorations-possibles)
11. [Publier le projet sur GitHub](#11-publier-le-projet-sur-github)
12. [Conclusion](#12-conclusion)

---

## 1. Présentation

L'utilisateur dépose une image (JPG, JPEG ou PNG) ; le modèle prédit la catégorie du document et affiche la probabilité de chaque catégorie.

Le système repose sur un réseau de neurones convolutif pré-entraîné, **MobileNetV3-Small**, adapté aux catégories du projet par **transfer learning** (apprentissage par transfert) : on réutilise ce que le réseau a appris sur ImageNet, et on n'entraîne que sa dernière couche.

### Catégories du jeu de données

Le dossier `raw/` contient quatre catégories, avec **30 images chacune, soit 120 images** :

| Dossier | Catégorie |
|---|---|
| `article_scientifique/` | Articles scientifiques |
| `cv/` | Curriculum vitæ |
| `facture/` | Factures |
| `lettre_administrative/` | Lettres administratives |

Si des images sont ajoutées ou retirées, les nombres d'images de chaque ensemble changent en conséquence.

### Fonctionnalités

- Répartition automatique des images en ensembles d'entraînement, de validation et de test.
- Redimensionnement, normalisation et augmentation des images d'entraînement.
- Entraînement de MobileNetV3-Small par transfer learning, avec sauvegarde du meilleur modèle.
- Évaluation : accuracy, F1-score macro, rapport par classe et matrice de confusion.
- Analyse des erreurs avec une planche des images mal classées.
- Interface web Streamlit : déposer une image, voir la prédiction et les probabilités par catégorie.

---

## 2. Technologies et bibliothèques

| Technologie | Utilisation |
|---|---|
| Python | Langage du projet |
| PyTorch | Construction, entraînement et utilisation du réseau de neurones |
| Torchvision | Modèle MobileNetV3-Small, chargement des images (`ImageFolder`) et transformations |
| Pillow (PIL) | Lecture des images dans l'interface |
| Matplotlib | Courbes d'entraînement, matrice de confusion et planche d'erreurs |
| Scikit-learn | Métriques (accuracy, F1-score, rapport de classification) et matrice de confusion |
| Streamlit | Interface web |
| uv | Création de l'environnement virtuel et installation des bibliothèques |

---

## 3. Structure du projet

```text
classification-documents-ia/
│
├── raw/                              # Images originales, un sous-dossier par catégorie
│   ├── article_scientifique/
│   ├── cv/
│   ├── facture/
│   └── lettre_administrative/
│
├── data/                             # Créé par prepare_data_simple.py
│   ├── train/                        # Images d'entraînement (70 %)
│   ├── val/                          # Images de validation (15 %)
│   └── test/                         # Images de test (le reste)
│
├── prepare_data_simple.py            # Répartition des images
├── pretraitement.py                  # Prétraitement et augmentation des images
├── entrainement.py                   # Entraînement du modèle
├── evaluation.py                     # Évaluation sur les images de test
├── erreur.py                         # Analyse des erreurs de classification
├── Demo Streamlit.py                 # Interface web de démonstration
│
├── modele.pt                         # Poids du meilleur modèle (généré)
├── classes.json                      # Noms des catégories (généré)
├── courbes_loss_accuracy.png         # Courbes d'entraînement (générées)
├── matrice_confusion.png             # Matrice de confusion (générée)
├── erreurs_analyse.png               # Exemples d'erreurs (généré s'il y a des erreurs)
├── .gitignore
└── README.md
```

Le dossier `data/` et les fichiers marqués « généré » n'existent pas au départ : ils sont créés par les scripts (voir la partie 7).

---

## 4. Installation

### 4.1. Prérequis

- Windows 10 ou 11 ;
- Python, dans une version prise en charge par PyTorch (par exemple **Python 3.12**) ;
- [uv](https://docs.astral.sh/uv/) pour l'environnement virtuel et les bibliothèques ;
- Git, pour récupérer ou publier le projet ;
- Visual Studio Code (recommandé) ;
- une connexion Internet, pour installer les bibliothèques et télécharger les poids pré-entraînés de MobileNetV3-Small au premier entraînement.

### 4.2. Récupérer le projet

Avec Git :

```powershell
git clone https://github.com/TON_NOM_UTILISATEUR/classification-documents-ia.git
cd classification-documents-ia
```

Ou en téléchargeant le ZIP depuis GitHub (**Code → Download ZIP**), puis en l'extrayant.

Ouvrez ensuite le dossier dans VS Code (**File → Open Folder**) et ouvrez un terminal (**Terminal → New Terminal**). Le terminal doit se trouver à la racine du projet, dans le dossier qui contient `entrainement.py`.

### 4.3. Créer et activer l'environnement virtuel

```powershell
uv venv --python 3.12
```

Cette commande crée un environnement virtuel `.venv`. Si vous utilisez une autre version de Python, remplacez `3.12` par la vôtre (ou par le chemin de votre `python.exe`).

Autorisez l'exécution des scripts pour la session en cours, puis activez l'environnement :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python --version
```

Une fois l'environnement activé, `(.venv)` apparaît au début de la ligne de commande.

### 4.4. Installer les bibliothèques

```powershell
uv pip install torch torchvision matplotlib scikit-learn streamlit pillow
```

Vérifiez l'installation :

```powershell
python -c "import torch, torchvision, matplotlib, sklearn, streamlit, PIL; print('Installation réussie !')"
```

Pour utiliser une carte graphique (GPU), choisissez la commande d'installation adaptée sur [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). La commande ci-dessus n'installe pas forcément la version CUDA.

> Chaque environnement `.venv` est indépendant : si vous le supprimez et le recréez, réinstallez les bibliothèques.

---

## 5. Exécution complète

Toutes les commandes se lancent **depuis la racine du projet**, avec l'environnement `.venv` activé.

| Étape | Commande | Résultat |
|---|---|---|
| 1. Préparer les données | `python prepare_data_simple.py` | Dossiers `data/train`, `data/val`, `data/test` |
| 2. Vérifier le prétraitement | `python pretraitement.py` | Catégories, nombre d'images par ensemble, forme d'une image transformée |
| 3. Entraîner le modèle | `python entrainement.py` | `modele.pt`, `classes.json`, `courbes_loss_accuracy.png` |
| 4. Évaluer le modèle | `python evaluation.py` | Métriques dans le terminal, `matrice_confusion.png` |
| 5. Analyser les erreurs | `python erreur.py` | Liste des erreurs, `erreurs_analyse.png` |
| 6. Lancer l'interface | `streamlit run "Demo Streamlit.py"` | Application sur http://localhost:8501 |

L'entraînement peut prendre plusieurs minutes selon l'ordinateur. Pour arrêter Streamlit, revenez dans le terminal et appuyez sur `Ctrl + C`.

---

## 6. Description des scripts

### 6.1. `prepare_data_simple.py` : préparation du jeu de données

1. Repère le dossier `raw/` situé à côté du script.
2. Parcourt ses sous-dossiers : chaque sous-dossier est une catégorie.
3. Garde les images `.jpg`, `.jpeg` et `.png`.
4. Mélange les images de chaque catégorie avec une graine fixe (`random.seed(42)`), pour obtenir toujours la même répartition.
5. Répartit les images : **70 %** pour l'entraînement, **15 %** pour la validation, le reste pour le test.
6. Copie les images dans `data/train`, `data/val` et `data/test`, en gardant un sous-dossier par catégorie.

| Ensemble | Proportion | Pour 30 images | Pour les 120 images |
|---|---:|---:|---:|
| Entraînement | 70 % | 21 | 84 |
| Validation | 15 % | 4 | 16 |
| Test | le reste | 5 | 20 |

> ⚠️ Ce script **supprime entièrement le dossier `data/`** s'il existe, puis le recrée. Toute modification faite à la main dans `data/` est perdue. Les images originales de `raw/` ne sont jamais supprimées.

### 6.2. `pretraitement.py` : prétraitement des images

- **Redimensionnement** : toutes les images passent en **224 × 224 pixels**, la taille attendue par le modèle.
- **Augmentation** (images d'entraînement uniquement) : rotation jusqu'à 5°, déplacement jusqu'à 3 %, zoom entre 95 % et 105 %, luminosité et contraste jusqu'à ±30 %. Le modèle voit ainsi des variantes des images et généralise mieux.
- **Conversion et normalisation** : `ToTensor()` transforme l'image en tenseur PyTorch, puis `Normalize()` normalise les trois canaux avec les statistiques d'ImageNet :

```python
MOYENNE = [0.485, 0.456, 0.406]
ECART = [0.229, 0.224, 0.225]
```

La validation et le test n'ont **aucune augmentation** : seulement le redimensionnement, la conversion et la normalisation.

Le script crée trois objets `ImageFolder` (`donnees_entrainement`, `donnees_validation`, `donnees_test`) : chaque sous-dossier devient automatiquement une catégorie.

### 6.3. `entrainement.py` : entraînement du modèle

**Transfer learning avec MobileNetV3-Small**, un réseau convolutif léger :

1. Charger MobileNetV3-Small avec ses poids pré-entraînés sur ImageNet.
2. **Geler** ses paramètres pour garder les caractéristiques visuelles déjà apprises.
3. Remplacer la dernière couche par une couche linéaire à 4 sorties (une par catégorie).
4. N'entraîner que cette nouvelle couche.

| Paramètre | Valeur | Signification |
|---|---:|---|
| Époques | 10 | Nombre de passages sur les images d'entraînement |
| Taille du batch | 8 | Nombre d'images traitées ensemble |
| Taille des images | 224 × 224 | Dimensions d'entrée du modèle |
| Fonction de perte | `CrossEntropyLoss` | Mesure l'erreur de classification |
| Optimiseur | `Adam`, learning rate 0.001 | Met à jour les poids |
| Appareil | CUDA ou CPU | GPU s'il est disponible, sinon processeur |

À chaque époque, le script calcule la perte (*loss*) et l'accuracy sur l'entraînement et la validation. **Le modèle qui a la meilleure accuracy de validation** est enregistré dans `modele.pt`.

Fichiers produits : `modele.pt` (poids), `classes.json` (noms des catégories dans l'ordre du modèle) et `courbes_loss_accuracy.png`.

### 6.4. `evaluation.py` : évaluation du modèle

Le script charge `modele.pt` et `classes.json`, puis prédit la catégorie des images de **test**, qui n'ont jamais servi à l'entraînement.

- **Accuracy** : part des prédictions correctes.
- **F1-score macro** : moyenne des F1-scores de chaque catégorie, chacune ayant le même poids.
- **Rapport de classification** : précision, rappel et F1-score par catégorie.
- **Matrice de confusion** : catégories réelles contre catégories prédites, enregistrée dans `matrice_confusion.png`.

### 6.5. `erreur.py` : analyse des erreurs

Pour chaque image de test mal classée, le script affiche le nom du fichier, la catégorie réelle, la catégorie prédite et la confiance. Les erreurs sont triées de la plus confiante à la moins confiante. Une planche de six erreurs au plus est enregistrée dans `erreurs_analyse.png`.

Cette étape montre quelles catégories le modèle confond et quelles images sont difficiles.

### 6.6. `Demo Streamlit.py` : interface web

1. Charge `modele.pt` et `classes.json`.
2. Permet de déposer une image JPG, JPEG ou PNG, et l'affiche.
3. Prédit la catégorie et affiche le score de confiance.
4. Affiche un graphique des probabilités de chaque catégorie.

Si la confiance maximale est **inférieure à 60 %**, l'interface avertit que la prédiction est incertaine. Ce score est la probabilité calculée par le modèle (Softmax) : il ne garantit pas que la prédiction est juste.

---

## 7. Résultats et fichiers générés

| Fichier | Description | Produit par |
|---|---|---|
| `data/` | Images réparties en train, validation et test | `prepare_data_simple.py` |
| `modele.pt` | Poids du meilleur modèle | `entrainement.py` |
| `classes.json` | Ordre des catégories du modèle | `entrainement.py` |
| `courbes_loss_accuracy.png` | Courbes de perte et d'accuracy | `entrainement.py` |
| `matrice_confusion.png` | Matrice de confusion sur le test | `evaluation.py` |
| `erreurs_analyse.png` | Exemples d'images mal classées | `erreur.py` |

L'accuracy et le F1-score dépendent de l'entraînement effectué : lancez `evaluation.py` pour les obtenir.

---

## 8. Conseils d'utilisation

- Utiliser des images nettes et lisibles ; éviter les photos sombres, floues ou très inclinées.
- Garder des images variées dans chaque catégorie.
- Ne jamais utiliser les images de test pour régler le modèle : elles servent seulement à mesurer ses performances finales.
- Pour ajouter une catégorie : créer un sous-dossier dans `raw/`, y mettre ses images, puis relancer la préparation et l'entraînement.
- Après chaque entraînement, relancer l'évaluation avant d'utiliser le modèle dans l'interface.

---

## 9. Problèmes fréquents

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | L'environnement `.venv` n'est pas activé, ou PyTorch n'y est pas installé. Activez-le, puis `uv pip install torch torchvision`. |
| `ModuleNotFoundError` pour `streamlit`, `PIL`, `sklearn` ou `matplotlib` | `uv pip install streamlit pillow scikit-learn matplotlib` dans l'environnement activé. |
| `modele.pt` ou `classes.json` introuvable | Lancez d'abord `python entrainement.py`, depuis la racine du projet. |
| Le dossier `data/` est introuvable | Lancez `python prepare_data_simple.py`, et vérifiez que `raw/` contient un sous-dossier par catégorie. |
| Erreur au téléchargement des poids pré-entraînés | Vérifiez la connexion Internet et les certificats SSL (voir la remarque ci-dessous). |
| Streamlit ne trouve pas le modèle | Lancez Streamlit depuis la racine du projet, là où se trouvent `modele.pt` et `classes.json`. |

> **Remarque de sécurité :** `entrainement.py` désactive pour l'instant la vérification des certificats SSL, pour contourner une erreur de téléchargement des poids. Cela réduit la sécurité de la connexion. Il vaut mieux corriger la configuration des certificats et garder la vérification activée.

---

## 10. Limites et améliorations possibles

Le jeu de données est petit (120 images, dont 20 pour le test) : les résultats sont à interpréter avec prudence, car une seule erreur fait varier l'accuracy de 5 points.

Améliorations possibles :

- ajouter des images plus nombreuses et plus variées dans chaque catégorie ;
- dégeler et ré-entraîner les dernières couches du réseau (*fine-tuning*) ;
- tester d'autres architectures et ajuster les hyperparamètres (époques, batch, learning rate) ;
- utiliser une validation croisée si le volume de données le permet ;
- ajouter à l'interface un historique des prédictions et l'export des résultats.

---

## 11. Publier le projet sur GitHub

1. Sur [GitHub](https://github.com/), cliquez sur **New repository**, nommez-le (par exemple `classification-documents-ia`) et **ne cochez pas** les options README, `.gitignore` ou licence, pour éviter un conflit au premier envoi.
2. Vérifiez que Git est installé : `git --version` (sinon, installez-le depuis [git-scm.com](https://git-scm.com/) et redémarrez VS Code).
3. Créez à la racine un fichier `.gitignore` :

```gitignore
# Environnement virtuel
.venv/
venv/

# Fichiers temporaires Python
__pycache__/
*.py[cod]

# Données générées (recréées par prepare_data_simple.py)
data/

# Paramètres locaux et fichiers système
.vscode/
.DS_Store
Thumbs.db
```

4. Envoyez le projet :

```powershell
git init
git branch -M main
git remote add origin https://github.com/TON_NOM_UTILISATEUR/classification-documents-ia.git
git add .
git commit -m "Premier envoi : classification de documents par IA"
git push -u origin main
```

Si Git demande un nom ou un e-mail : `git config --global user.name "Votre nom"` et `git config --global user.email "votre-adresse@example.com"`.

Pour les modifications suivantes :

```powershell
git add .
git commit -m "Courte description des modifications"
git push
```

---

## 12. Conclusion

Ce mini-projet met en œuvre les étapes principales d'une application de vision par ordinateur : préparation des données, prétraitement, apprentissage par transfert, évaluation et démonstration interactive.

Avec MobileNetV3-Small et Streamlit, il propose une première solution de classification automatique de documents, qui prédit la catégorie d'une image et montre les probabilités associées. Plus de données et d'expériences permettraient d'améliorer sa fiabilité.#   p r o j e t - c l a s s i f i c a t i o n  
 