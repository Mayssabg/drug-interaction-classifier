import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

CSV_PATH = "dataset-18000-final.csv"        
TARGET_COL = "label"   
COLS_TO_DROP = ["Drug 1", "Drug 2", "description", "smiles_rdkit"]  
USE_FINGERPRINTS = True       
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 10                  

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

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,              
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,    
    validation_fraction=0.1, 
    n_iter_no_change=10,     
    random_state=42,
    verbose=True,
)

print("Entraînement en cours...")
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\nAccuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")
print("\n=== Rapport détaillé ===")
print(classification_report(y_val, y_pred, target_names=["non-interaction", "interaction"]))

joblib.dump(model, "mlp_ddi_sklearn.joblib")
joblib.dump(scaler, "scaler_ddi.joblib")
joblib.dump(FEATURE_COLUMNS, "mlp_ddi_feature_columns.joblib")
print(f"Colonnes de features sauvegardées : mlp_ddi_feature_columns.joblib ({len(FEATURE_COLUMNS)} colonnes)")
print("\nModèle sauvegardé : mlp_ddi_sklearn.joblib")
