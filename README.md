# Pipeline complet de prédiction – Modèle général & modèle hard

Ce projet implémente un système de prédiction en deux étages :

1. un **modèle général**, entraîné sur tout le dataset ;
2. un **modèle hard**, spécialisé dans les cas incertains ou ambigus ;
3. un **système hybride**, qui combine intelligemment les deux modèles pour la prédiction finale.

L’ensemble repose sur trois notebooks à exécuter **dans l’ordre suivant** :

1. `all_mighty_pred.ipynb`
2. `specialized_hard_pred.ipynb`
3. `testDataset.ipynb`

Les sections ci-dessous expliquent précisément ce que fait chacun d’eux.

---

## 📘 1. `all_mighty_pred.ipynb` — Modèle général (BERT + XGBoost)

Ce notebook est le cœur du pipeline.  
Il construit un modèle robuste entraîné sur **l’intégralité du dataset**.

### Il effectue les opérations suivantes :

- **Chargement du jeu de données d’entraînement**.
- **Prétraitement complet** :
  - gestion des features numériques et catégorielles,
  - extraction d’embeddings avec **BERTweetFR** pour le texte,
  - normalisation / encodage.
- **Entraînement du modèle général** (souvent XGBoost ou ensemble optimisé mais empiriquement le XGBoost tout seul fournit de meilleurs résultats).
- **Production des probabilités du modèle général** :
  - sur le train → pour repérer les cas difficiles,
  - sur le test → utilisées lors de la prédiction finale.
- **Export des probabilités sous forme de CSV** :
  - `proba_general_train.csv`
  - `proba_general_test.csv`

Ces fichiers sont nécessaires pour le notebook suivant.

---

## 📘 2. `specialized_hard_pred.ipynb` — Modèle hard spécialisé (bios + features)

Ce notebook crée un modèle secondaire destiné à **corriger les erreurs du modèle général**.  
Il se concentre uniquement sur les cas où le modèle général est incertain.

### Il réalise les étapes suivantes :

- **Import des probabilités du modèle général** (issues de `all_mighty_pred.ipynb`).
- **Import du “hard set”** (issues de `misclassified_{i}.ipynb`).
- **Extraction des probas du modèle générale issues du “hard set”** :
  - sélection des lignes concernées via `challenge_id`.
- **Enrichissement des features** :
  - La proba du modèle générale : `proba_general`,
  - Son incertitude : `uncertainty_general = |proba_general - 0.5|`,
  - et les features que l'on a sélectionnées du dataset original.
- **Prétraitement dédié au hard set** :
  - TF-IDF + SVD sur les textes,
  - traitement numérique / catégoriel.
- **Entraînement du modèle hard**  
  (ex : XGBoost, ou voting/stacking model XGB + LR + MLP, mais encore une fois, empiriquement, le XGBoost tout seul fournit de meilleurs résultats).
- **Détermination des cas difficiles du test**.
  - Export de ces sous-dataset dans les misclassified_{i}.csv (le nom provient de l'époque où l'on selectionnait uniquement les misclassified, mais on a trouvé de meilleures méthodes après)
- **Export des prédictions du modèle hard** :
  - `specialized_predictions.csv`

Ces fichiers sont nécessaires pour le notebook `testDataset.ipynb`.

---

## 📘 3. `testDataset.ipynb` — Assemblage final & génération de `submission.csv`

Ce notebook combine les prédictions du modèle général et du modèle hard pour créer la **prédiction finale**.

### Il effectue :

- **Chargement** :
  - des prédictions du général (`proba_general_test.csv`),
  - des prédictions du hard (`specialized_predictions.csv`),
  - du dataset test pour récupérer les `challenge_id`.
- **Système hybride** :
Le système hybride combine le modèle général et le modèle hard selon la confiance
du modèle général :

- **Si la probabilité du modèle général est en dehors de l’intervalle `[T_low, T_high]` :**  
  → on utilise la prédiction du **modèle général**

- **Si la probabilité du modèle général est à l’intérieur de l’intervalle `[T_low, T_high]` :**  
  → on utilise la prédiction du **modèle hard**, spécialisé dans les cas ambigus

Autrement dit :
Si proba_general ∉ [T_low, T_high] → prédiction = modèle général
Sinon                               → prédiction = modèle hard


- **Génération des prédictions finales** (`label`) pour chaque challenge_id.
- **Création du fichier de soumission** :
  - `final_predictions.csv`, format :
    ```
    ID,Prediction
    ...
    12345,0
    12346,1
    ...
    ```

---

# 🧩 Structure des données attendues

Le pipeline suppose l'existence des fichiers suivants :

- un **Kaggle2025/train.jsonl** contenant :
  - `challenge_id`,
  - features numériques / catégorielles,
  - textes (`full_text`, `user_desc`…),
  - le `label` cible.

- un **Kaggle2025/kaggle_test.jsonl** contenant les mêmes features **sans label**.

Les notebooks génèrent ensuite leurs propres fichiers intermédiaires (probabilités, cas difficiles, etc.).

---

# 🔧 Prérequis

- Python ≥ 3.10 (éviter 3.13 qui cause des problèmes avec ipykernel)
- Jupyter Notebook ou JupyterLab
- Dépendances Python principales :

```bash
pip install pandas numpy scikit-learn xgboost tqdm matplotlib transformers torch

