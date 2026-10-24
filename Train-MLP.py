"""
Entraînement d'un MLP pour la classification binaire interaction/non-interaction (DDI).
Réutilise le dataset déjà construit (descripteurs + tanimoto, +/- fingerprints).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
# ------------------------------------------------------------------
# CONFIG — à adapter à tes noms de colonnes réels
# ------------------------------------------------------------------
CSV_PATH = "dataset-18000-final.csv"          # ou ton échantillon de 8000 lignes
TARGET_COL = "label"     # colonne 0/1 à prédire — adapte si le nom diffère
COLS_TO_DROP = ["Drug 1", "Drug 2", "description", "smiles_rdkit"]  # colonnes non numériques à exclure
USE_FINGERPRINTS = True        # inclut fp1_*/fp2_* (4096 colonnes, plus lourd en mémoire/calcul)
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 10                  # early stopping

# ------------------------------------------------------------------
# 1. Chargement et préparation
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
if not USE_FINGERPRINTS:
    fp_cols = [c for c in df.columns if c.startswith("fp1_") or c.startswith("fp2_")]
    cols_to_drop += fp_cols

X = df.drop(columns=cols_to_drop + [TARGET_COL])
X = X.select_dtypes(include=[np.number])
y = df[TARGET_COL].values

print(f"Features utilisées : {X.shape[1]} colonnes, {len(X)} lignes")
FEATURE_COLUMNS = X.columns.tolist()
X_train, X_val, y_train, y_val = train_test_split(
    X.values, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ------------------------------------------------------------------
# 2. Modèle — MLP à 3 couches cachées (256, 128, 64), équivalent à la version PyTorch
# ------------------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,              # régularisation L2 (rôle proche du dropout)
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,     # arrête automatiquement si val_loss stagne
    validation_fraction=0.1, # pris sur le train pour l'early stopping interne
    n_iter_no_change=10,     # patience équivalente à PATIENCE=10
    random_state=42,
    verbose=True,
)

print("Entraînement en cours...")
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 3. Évaluation
# ------------------------------------------------------------------
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\nAccuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")
print("\n=== Rapport détaillé ===")
print(classification_report(y_val, y_pred, target_names=["non-interaction", "interaction"]))

# ------------------------------------------------------------------
# 4. Sauvegarde du modèle + scaler pour l'inférence future
# ------------------------------------------------------------------
joblib.dump(model, "mlp_ddi_sklearn.joblib")
joblib.dump(scaler, "scaler_ddi.joblib")
joblib.dump(FEATURE_COLUMNS, "mlp_ddi_feature_columns.joblib")
print(f"💾 Colonnes de features sauvegardées : mlp_ddi_feature_columns.joblib ({len(FEATURE_COLUMNS)} colonnes)")
print("\nModèle sauvegardé : mlp_ddi_sklearn.joblib")