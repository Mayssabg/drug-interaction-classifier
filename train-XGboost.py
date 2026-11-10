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
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Entraînement XGBoost", unit="arbre")
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  
    def after_training(self, model):
        self.pbar.close()
        return model
FICHIER_DATASET = "data-final.csv"
df = pd.read_csv(FICHIER_DATASET)
print(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
X = df.drop(columns=['label']).select_dtypes(include=[np.number])
y = df['label']
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
    callbacks=[BarreAvancement(N_ESTIMATORS)]
)
model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(" Résultats :")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision : {precision_score(y_test, y_pred):.3f}")
print(f"Recall    : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.3f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Positif (1)']))
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
print("Top 15 features les plus importantes :")
print(importances.sort_values(ascending=False).head(15))
joblib.dump(model, "xgb_ddi_model.joblib")
joblib.dump(FEATURE_COLUMNS, "xgb_ddi_feature_columns.joblib")
print(" Modèle sauvegardé : xgb_ddi_model.joblib")
print(f"Colonnes de features sauvegardées : xgb_ddi_feature_columns.joblib ({len(FEATURE_COLUMNS)} colonnes)")
