import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class BarreAvancement(xgb.callback.TrainingCallback):
    """Affiche une barre de progression tqdm pendant l'entraînement,
    une étape = un arbre ajouté (n_estimators étapes au total)."""

    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Entraînement XGBoost", unit="arbre")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # False = ne pas arrêter l'entraînement

    def after_training(self, model):
        self.pbar.close()
        return model

FICHIER_DATASET = "data-final.csv"

# 1. Charger le dataset préparé
df = pd.read_csv(FICHIER_DATASET)
print(f"📁 Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 2. Features (X) = colonnes numériques sauf label / y = label
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
y = df['label']
print(f"🔢 Features utilisées : {X.shape[1]}")

# Même logique que pour Random Forest : on garde la liste des colonnes
# de features dans l'ordre exact, indispensable pour l'inférence plus tard
FEATURE_COLUMNS = X.columns.tolist()

# 3. Split train/test stratifié (identique au script Random Forest,
# même random_state=42 pour pouvoir comparer les deux modèles sur
# exactement le même split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✂️ Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# 4. Entraîner XGBoost (avec barre d'avancement tqdm)
N_ESTIMATORS = 300
model = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    callbacks=[BarreAvancement(N_ESTIMATORS)]
)
model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

# 5. Évaluer
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n✅ Résultats :")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision : {precision_score(y_test, y_pred):.3f}")
print(f"Recall    : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.3f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Positif (1)']))

# 6. Importance des features (top 15) — utile pour voir si les fingerprints
# ou les descripteurs pèsent le plus dans la décision du modèle
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
print("\n🔝 Top 15 features les plus importantes :")
print(importances.sort_values(ascending=False).head(15))

# 7. Sauvegarder le modèle ET la liste des colonnes de features
joblib.dump(model, "xgb_ddi_model.joblib")
joblib.dump(FEATURE_COLUMNS, "xgb_ddi_feature_columns.joblib")
print("\n💾 Modèle sauvegardé : xgb_ddi_model.joblib")
print(f"💾 Colonnes de features sauvegardées : xgb_ddi_feature_columns.joblib ({len(FEATURE_COLUMNS)} colonnes)")
