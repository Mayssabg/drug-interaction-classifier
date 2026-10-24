# Prédiction d'interactions médicamenteuses (DDI) par apprentissage automatique

Projet de classification binaire : à partir de deux médicaments (nom DrugBank ou SMILES), prédire s'il existe une **interaction médicamenteuse** ou non. Le projet compare trois modèles (XGBoost, Random Forest, MLP) et fournit une **interface graphique Tkinter** pour tester des combinaisons de médicaments.

> **Avertissement** : projet académique à visée pédagogique. Il ne remplace en aucun cas l'avis d'un médecin ou d'un pharmacien.

---

## Aperçu du pipeline

```
DrugBank + interactions connues
        │
        ▼
Génération des négatifs (paires absentes des interactions connues, ratio 1:1)
        │
        ▼
Représentation moléculaire de chaque médicament (via SMILES, RDKit)
   ├── Descripteurs moléculaires (36 par médicament)
   ├── Fingerprints Morgan / ECFP4 (rayon 2, 2048 bits par médicament)
   └── Similarité de Tanimoto entre les deux molécules
        │
        ▼
Dataset final : 4169 features numériques par paire
        │
        ▼
Entraînement et comparaison : XGBoost · Random Forest · MLP
        │
        ▼
Interface Tkinter (prédiction + description de l'interaction connue)
```

## Features

Pour chaque paire (médicament 1, médicament 2), le vecteur d'entrée contient :

| Bloc | Nombre | Détail |
|---|---|---|
| Descripteurs moléculaires | 72 | 36 descripteurs RDKit par médicament (suffixes `_1` et `_2`), choisis pour leur pertinence pharmacogénomique (ex. MolLogP, TPSA, qed, BertzCT, HallKierAlpha) |
| Fingerprints Morgan (ECFP4) | 4096 | 2048 bits par médicament (`fp1_*`, `fp2_*`) |
| Similarité de Tanimoto | 1 | Similarité structurale entre les deux molécules |
| **Total** | **4169** | |

## Résultats

Évaluation sur un split stratifié 80/20 (`random_state=42`), dataset de 18 000 paires.

| Modèle | Accuracy | F1-score | AUC-ROC |
|---|---|---|---|
| **XGBoost** | **0.920** | **0.922** | **0.972** |
| Random Forest (300 arbres) | 0.914 | 0.916 | 0.972 |
| MLP (256-128-64) | 0.909 | 0.910 | n/a |

Le Random Forest s'appuie surtout sur les descripteurs moléculaires (qed, MolLogP, TPSA...), alors que XGBoost accorde plus de poids aux fingerprints.

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `train-XGboost.py` | Entraînement XGBoost avec barre de progression (tqdm), évaluation complète et sauvegarde du modèle |
| `train-rf.py` | Variante d'entraînement XGBoost sans barre de progression, même split pour la comparaison |
| `mlp.py` | Entraînement d'un MLP scikit-learn (3 couches cachées, early stopping, standardisation des features) |
| `APP-final.py` | Interface graphique Tkinter de prédiction |
| `descripteurs.py` | Calcul des descripteurs moléculaires RDKit (requis par l'application) |
| `fingerprint.py` | Calcul des fingerprints Morgan et de la similarité de Tanimoto (requis par l'application) |

Les modèles entraînés sont sauvegardés avec `joblib`, **avec la liste ordonnée des colonnes de features**, indispensable pour garantir la même structure d'entrée à l'inférence.

## Installation

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>

pip install pandas numpy scikit-learn xgboost rdkit joblib tqdm
```

Tkinter est inclus avec Python (sous Linux : `sudo apt install python3-tk`).

## Utilisation

### 1. Entraîner un modèle

Placer le dataset final (colonne cible `label`, 0 = pas d'interaction, 1 = interaction) à la racine du dépôt, puis :

```bash
python train-XGboost.py     # XGBoost -> xgb_ddi_model.joblib
python mlp.py               # MLP     -> mlp_ddi_sklearn.joblib + scaler_ddi.joblib
```

### 2. Lancer l'interface

Fichiers nécessaires à la racine : `xgb_ddi_model.joblib`, `xgb_ddi_feature_columns.joblib`, `drugbank.csv`, `db_drug_interactions1.csv`.

```bash
python APP-final.py
```

Dans l'interface :
- saisir les médicaments déjà pris, séparés par des virgules ;
- saisir le nouveau médicament ;
- chaque entrée peut être un **nom** ou un **SMILES** (détecté automatiquement) ;
- en cas de faute de frappe ou de casse différente, l'application **propose les noms ou SMILES les plus proches** de la base ;
- l'application affiche, pour chaque médicament, la prédiction, la probabilité, et la description de l'interaction lorsqu'elle est connue dans la base.

Exemple : `Aspirine, Paracetamol, Ibuprofene` avec le nouveau médicament `Warfarine`.

## Données

- **DrugBank** : noms des médicaments et structures (SMILES).
- **Interactions connues** : fichier de paires (colonnes `Drug 1`, `Drug 2`, `Interaction Description`).
- **Négatifs** : paires tirées aléatoirement parmi celles absentes des interactions connues (ratio 1:1).

Les fichiers de données volumineux ne sont pas versionnés dans ce dépôt. Vérifier les conditions de licence de DrugBank avant toute redistribution.

## Limites et pistes d'amélioration

- Les négatifs sont des paires **non répertoriées**, pas des paires prouvées sans interaction : certaines peuvent être de vrais positifs non documentés.
- Le modèle prédit la présence d'une interaction, pas sa nature ni sa gravité.
- Pistes : split par médicament (pour tester la généralisation à de nouvelles molécules), optimisation des hyperparamètres, modèles de graphes moléculaires (GNN), prédiction du type d'interaction à partir de la description textuelle.

## Auteure

**Mayssa Ben Guedria** — étudiante en Génie Biologique Industriel, INSAT
[LinkedIn](https://www.linkedin.com/in/mayssa-ben-guedria-69a144308/)
