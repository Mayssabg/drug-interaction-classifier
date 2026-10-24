# app_tkinter.py - Interface avec Tkinter (version sans emojis)

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from descripteurs import (
    get_relevant_descriptors_for_pharmacogenomics,
    calculate_descriptors_fast,
)
from fingerprint import calculer_fingerprint, calculer_similarite_tanimoto, N_BITS
from difflib import get_close_matches

class InterfaceDDI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction d'interactions médicamenteuses")
        self.root.geometry("750x650")
        
        # Chargement des ressources
        self.charger_ressources()
        
        # Création de l'interface
        self.creer_interface()
    
    def charger_ressources(self):
        try:
            self.model = joblib.load("xgb_ddi_model.joblib")
            self.FEATURE_COLUMNS = joblib.load("xgb_ddi_feature_columns.joblib")
            self.drugbank = pd.read_csv("drugbank.csv")
            self.interactions = pd.read_csv("db_drug_interactions1.csv")
            
            # Créer un dictionnaire pour la recherche rapide
            self.dict_noms = {row["name"].lower(): row["name"] for _, row in self.drugbank.iterrows()}
            self.dict_smiles = {row["smiles_rdkit"]: row["name"] for _, row in self.drugbank.iterrows() if pd.notna(row["smiles_rdkit"])}
            self.liste_noms = list(self.dict_noms.keys())
            self.liste_smiles = list(self.dict_smiles.keys())
            
            self.DESCRIPTOR_NAMES = get_relevant_descriptors_for_pharmacogenomics()
            self.CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(self.DESCRIPTOR_NAMES)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les ressources:\n{str(e)}")
            self.root.quit()
    
    def get_drug_info(self, input_text):
        """Retourne les informations du médicament à partir d'un nom ou SMILES"""
        input_clean = input_text.strip()
        drug_name = None
        smiles = None
        
        # Vérifier si c'est un SMILES
        is_smiles = any(c in input_clean for c in ['[', ']', '=', '#', '(', ')', '@', '\\', '/']) and len(input_clean) > 10
        
        if is_smiles:
            if input_clean in self.dict_smiles:
                drug_name = self.dict_smiles[input_clean]
                smiles = input_clean
            else:
                matches = get_close_matches(input_clean, self.liste_smiles, n=5, cutoff=0.6)
                if matches:
                    raise ValueError(f"SMILES non trouvé. Voulez-vous dire:\n" + "\n".join(f"  - {m}" for m in matches[:5]))
                else:
                    raise ValueError(f"SMILES non reconnu. Vérifiez le format.")
        else:
            input_lower = input_clean.lower()
            if input_lower in self.dict_noms:
                drug_name = self.dict_noms[input_lower]
                smiles = self.get_smiles(drug_name)
            else:
                matches = get_close_matches(input_lower, self.liste_noms, n=5, cutoff=0.6)
                if matches:
                    suggestions = [self.dict_noms[m] for m in matches[:5]]
                    raise ValueError(f"Médicament non trouvé. Voulez-vous dire:\n" + "\n".join(f"  - {s}" for s in suggestions))
                else:
                    raise ValueError(f"Médicament '{input_clean}' introuvable dans la base de données")
        
        return drug_name, smiles
    
    def get_smiles(self, nom):
        row = self.drugbank[self.drugbank["name"].str.lower() == nom.lower()]
        if row.empty:
            raise ValueError(f"Médicament '{nom}' introuvable")
        smiles = row.iloc[0]["smiles_rdkit"]
        if pd.isna(smiles) or smiles == "":
            raise ValueError(f"Pas de SMILES disponible pour '{nom}'")
        return smiles
    
    def construire_features(self, smiles_1, smiles_2):
        desc_1 = calculate_descriptors_fast(smiles_1, self.DESCRIPTOR_NAMES, self.CALCULATOR).to_dict()
        desc_2 = calculate_descriptors_fast(smiles_2, self.DESCRIPTOR_NAMES, self.CALCULATOR).to_dict()
        
        features = {}
        for k, v in desc_1.items():
            features[f"{k}_1"] = v
        for k, v in desc_2.items():
            features[f"{k}_2"] = v
        
        fp_1 = calculer_fingerprint(smiles_1)
        fp_2 = calculer_fingerprint(smiles_2)
        for i in range(N_BITS):
            features[f"fp1_{i}"] = fp_1[i]
            features[f"fp2_{i}"] = fp_2[i]
        
        features["tanimoto_similarity"] = calculer_similarite_tanimoto(smiles_1, smiles_2)
        return pd.DataFrame([features])[self.FEATURE_COLUMNS]
    
    def chercher_description(self, nom_1, nom_2):
        d1 = self.interactions["Drug 1"].str.lower()
        d2 = self.interactions["Drug 2"].str.lower()
        n1, n2 = nom_1.lower(), nom_2.lower()
        match = self.interactions[((d1 == n1) & (d2 == n2)) | ((d1 == n2) & (d2 == n1))]
        if not match.empty:
            return match.iloc[0]["Interaction Description"]
        return None
    
    def predire(self):
        """Fonction appelée quand on clique sur le bouton"""
        # Récupérer les entrées
        medicaments_str = self.entry_medicaments.get("1.0", tk.END).strip()
        nouveau_med = self.entry_nouveau.get().strip()
        
        if not medicaments_str or not nouveau_med:
            messagebox.showwarning("Attention", "Veuillez remplir tous les champs")
            return
        
        medicaments_pris = [m.strip() for m in medicaments_str.split(",") if m.strip()]
        
        # Effacer les résultats précédents
        self.text_resultats.delete("1.0", tk.END)
        
        try:
            # Récupérer les infos du nouveau médicament
            nom_nouveau, smiles_nouveau = self.get_drug_info(nouveau_med)
            
            # En-tête des résultats
            self.text_resultats.insert("1.0", "=" * 70 + "\n")
            self.text_resultats.insert("end", "RESULTATS DES PREDICTIONS\n")
            self.text_resultats.insert("end", "=" * 70 + "\n\n")
            
            # Nouveau médicament (affichage épuré)
            self.text_resultats.insert("end", f"NOUVEAU MEDICAMENT: {nom_nouveau}\n")
            self.text_resultats.insert("end", f"SMILES: {smiles_nouveau[:80]}...\n\n")
            self.text_resultats.insert("end", "-" * 70 + "\n\n")
            
            resultats = []
            
            for med in medicaments_pris:
                try:
                    nom_med, smiles_med = self.get_drug_info(med)
                    
                    X = self.construire_features(smiles_med, smiles_nouveau)
                    proba = self.model.predict_proba(X)[0][1]
                    
                    if proba >= 0.5:
                        prediction = "INTERACTION DETECTEE"
                    else:
                        prediction = "PAS D'INTERACTION"
                    
                    desc = self.chercher_description(nom_med, nom_nouveau)
                    
                    # Formater le résultat (version sans emojis)
                    resultat = f"MEDICAMENT: {nom_med}\n"
                    resultat += f"SMILES: {smiles_med[:80]}...\n"
                    resultat += f"PREDICTION: {prediction} (probabilite: {proba:.2%})\n"
                    
                    if desc and proba >= 0.5:
                        resultat += f"INTERACTION: {desc}\n"
                    
                    resultat += "\n"
                    resultats.append(resultat)
                    
                except Exception as e:
                    resultats.append(f"ERREUR avec '{med}': {str(e)}\n\n")
            
            # Afficher les résultats
            for r in resultats:
                self.text_resultats.insert("end", r)
                self.text_resultats.insert("end", "-" * 70 + "\n\n")
            
            # Résumé
            nb_interactions = sum(1 for r in resultats if "INTERACTION DETECTEE" in r)
            self.text_resultats.insert("end", "=" * 70 + "\n")
            if nb_interactions > 0:
                self.text_resultats.insert("end", f"ATTENTION: {nb_interactions} interaction(s) detectee(s)\n")
                self.text_resultats.insert("end", "Consultez un professionnel de sante avant de prendre ce medicament.")
            else:
                self.text_resultats.insert("end", "Aucune interaction detectee avec les medicaments pris.")
            self.text_resultats.insert("end", "\n" + "=" * 70 + "\n")
            
        except Exception as e:
            self.text_resultats.insert("1.0", f"ERREUR: {str(e)}")
    
    def creer_interface(self):
        # Titre
        titre = tk.Label(self.root, text="Prediction d'interactions medicamenteuses", 
                         font=("Arial", 16, "bold"))
        titre.pack(pady=10)
        
        # Cadre pour les entrées
        cadre_input = tk.Frame(self.root)
        cadre_input.pack(pady=10, padx=20, fill="x")
        
        # Type d'entrée
        cadre_type = tk.Frame(cadre_input)
        cadre_type.pack(fill="x", pady=5)
        
        label_type = tk.Label(cadre_type, text="Format d'entree:", font=("Arial", 10))
        label_type.pack(side="left", padx=(0, 10))
        
        self.input_type_var = tk.StringVar(value="Auto (detection automatique)")
        type_options = ["Auto (detection automatique)", "Nom", "SMILES"]
        type_menu = ttk.Combobox(cadre_type, textvariable=self.input_type_var, 
                                 values=type_options, width=30)
        type_menu.pack(side="left")
        
        # Info
        label_info = tk.Label(cadre_type, text="(SMILES detecte automatiquement)", 
                              font=("Arial", 8), fg="gray")
        label_info.pack(side="left", padx=(10, 0))
        
        # Médicaments déjà pris
        label_medicaments = tk.Label(cadre_input, text="Medicaments deja pris (separes par des virgules):")
        label_medicaments.pack(anchor="w", pady=(10,0))
        
        self.entry_medicaments = scrolledtext.ScrolledText(cadre_input, height=3, font=("Arial", 10))
        self.entry_medicaments.pack(fill="x", pady=5)
        
        # Nouveau médicament
        label_nouveau = tk.Label(cadre_input, text="Nouveau medicament (nom ou SMILES):")
        label_nouveau.pack(anchor="w", pady=(10,0))
        
        self.entry_nouveau = tk.Entry(cadre_input, font=("Arial", 10))
        self.entry_nouveau.pack(fill="x", pady=5)
        
        # Bouton prédire
        bouton_predire = tk.Button(self.root, text="Predire les interactions", 
                                   command=self.predire, bg="#22439e", fg="white",
                                   font=("Arial", 12, "bold"), height=2)
        bouton_predire.pack(pady=10, padx=20, fill="x")
        
        # Cadre pour les résultats
        cadre_resultats = tk.Frame(self.root)
        cadre_resultats.pack(pady=10, padx=20, fill="both", expand=True)
        
        label_resultats = tk.Label(cadre_resultats, text="Resultats:", font=("Arial", 12, "bold"))
        label_resultats.pack(anchor="w")
        
        self.text_resultats = scrolledtext.ScrolledText(cadre_resultats, height=15, 
                                                        font=("Courier", 10), bg="#f0f0f0")
        self.text_resultats.pack(fill="both", expand=True)
        
        # Exemple en bas
        cadre_exemple = tk.Frame(self.root)
        cadre_exemple.pack(pady=5)
        label_exemple = tk.Label(cadre_exemple, 
                                 text="Exemple (noms): Aspirine, Paracetamol, Ibuprofene | Warfarine\n"
                                      "Exemple (SMILES): CC(=O)OC1=CC=CC=C1C(=O)O, CC(=O)NC1=CC=C(O)C=C1 | C12CC3=C(CCN1CC=C3)C=C4C=CC=CC4=C2", 
                                 font=("Arial", 8), fg="gray", justify="left")
        label_exemple.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceDDI(root)
    root.mainloop()