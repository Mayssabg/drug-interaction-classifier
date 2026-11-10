import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

FICHIER_DATASET = "dataset-18000-final.csv"
df = pd.read_csv(FICHIER_DATASET)
print(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
if 'label' not in df.columns:
    raise ValueError("La colonne 'label' est introuvable dans le dataset")
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
y = df['label']
if X.shape[1] == 0:
    raise ValueError("Aucune feature numérique trouvée dans le dataset")

print(f"Features utilisées : {X.shape[1]}")
FEATURE_COLUMNS = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
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
    verbosity=0 
)

print("Entraînement en cours...")
model.fit(X_train, y_train, verbose=False)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(" Résultats :")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Positif (1)']))
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
print("Top 15 features les plus importantes :")
print(importances.sort_values(ascending=False).head(15).round(4))
joblib.dump(model, "xgb_ddi_model.joblib")
joblib.dump(FEATURE_COLUMNS, "xgb_ddi_feature_columns.joblib")
print("Modèle sauvegardé : xgb_ddi_model.joblib")
print(f"💾 Colonnes de features sauvegardées : xgb_ddi_feature_columns.joblib ({len(FEATURE_COLUMNS)} colonnes)")

# Optionnel : afficher quelques statistiques supplémentaires
print("\n📊 Statistiques du modèle :")
print(f"Nombre d'arbres : {len(model.get_booster().get_dump())}")
print(f"Classes : {model.classes_}")
