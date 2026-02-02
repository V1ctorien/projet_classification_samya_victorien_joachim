import sqlite3 as sql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os
# -----------------------------------
#        class data
# -----------------------------------

model = joblib.load('model_test/rf_model.pkl')


class traitement_data:
    def sans_guillemets(cvs):
            
            base, ext = os.path.splitext(cvs) # séparation du nom de fichier et de son extension
            nouveau_fichier = f"{base}_sans_guillemets{ext}" # création du nom du nouveau fichier en ajoutant _clean avant l'extension

            with open(cvs, "r", encoding="utf-8") as f:
                 lines = f.readlines()  #lecture de toutes les lignes du fichier

            cleaned_lines = [line.strip().strip('"') for line in lines] # suppression des guillemets en début et fin de ligne

            with open(nouveau_fichier, "w", encoding="utf-8") as f: # réécriture du fichier sans guillemets
                 f.write("\n".join(cleaned_lines)) # "\n".join() permet de recréer le contenu du fichier en joignant les lignes nettoyées avec des sauts de ligne

    def traitement_splt_csv(file_path):
            data = pd.read_csv(file_path, sep=";", decimal=',')
            dft=data
            cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            dft[cols] = dft[cols].astype(float)
            x,y= dft.drop("isFraud", axis=1), dft["isFraud"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            x_test["isFraud"] = y_test.values
            x_train["isFraud"] = y_train.values
            x_test.to_csv("data\\data_test_set.csv", sep=";", index=False)
            x_train.to_csv("data\\data_train_set.csv", sep=";", index=False)
            return dft


    def recup_stat_gen(file_path):
            df = pd.read_csv(file_path, sep=";", decimal=',')
            nb_lignes = df.shape[0]
            nb_colonnes = df.shape[1]
            colonnes_stat = []
            for col in df.columns:
                colonnes_stat.append({"col_name": col, "type_donnee": str(df[col].dtype), "nb_valeurs_nulles": df[col].isnull().sum()})
                
            type_stat=df.groupby("type")["is_fraud"].agg(nb_lignes="count",proportion_fraud="mean").reset_index() # groupe par type puis selectione la colonne is_fraud pour faire des agrégations
            # .agg permet de creer plusieurs colonnes d'agrégations avec titre="fonction d'agrégation".puis reset_index pour remettre le type comme colonne classique
            type_stat["proportion_fraud"] = type_stat["proportion_fraud"].round(4) # arrondir à 4 décimales
            type_stat=type_stat.to_dict(orient="records") # transformer en liste de dictionnaires car plus facile à manipuler pour affichage
            return nb_lignes, nb_colonnes, colonnes_stat, type_stat

def create_bdd():
        import sqlite3 as sql

        connection = sql.connect("database/DB_fraud.db")                   # creation de la base si elle n'existe pas
        curseur = connection.cursor()                           # Initialisation du cursor pour les interaction avec la base


        # -- ----------------------------
        # -- Table: Utilisateur
        # -- ----------------------------
        connection = sql.connect("database/DB_fraud.db")
        curseur = connection.cursor()
        curseur.execute("""CREATE TABLE IF NOT EXISTS Utilisateur (
        ID INTEGER PRIMARY KEY NOT NULL,
        Nom VARCHAR(20) NOT NULL,
        Prenom VARCHAR(20) NOT NULL,
        Telephone VARCHAR(50) NOT NULL
        );""")

        connection.commit()                                    
        connection.close()  

        # -- ----------------------------
        # -- Table: Compte
        # -- ----------------------------
        connection = sql.connect("database/DB_fraud.db")
        curseur = connection.cursor()
        curseur.execute("""CREATE TABLE IF NOT EXISTS Compte (
        ID VARCHAR(50) PRIMARY KEY NOT NULL,
        ID_Utilisateur INT,
        CONSTRAINT Compte_ID_Utilisateur_FK FOREIGN KEY (ID_Utilisateur) REFERENCES Utilisateur (ID)
        );""")

        connection.commit()                                    
        connection.close()  

        # -- ----------------------------
        # -- Table: Transaction
        # -- ----------------------------
        connection = sql.connect("database/DB_fraud.db")
        curseur = connection.cursor()
        curseur.execute("""CREATE TABLE IF NOT EXISTS Transactions (
        ID INTEGER PRIMARY KEY NOT NULL,
        type VARCHAR(20) NOT NULL,
        amount FLOAT NOT NULL,
        oldbalanceOrg FLOAT NOT NULL,
        newbalanceOrig FLOAT NOT NULL,
        oldbalanceDest FLOAT NOT NULL,
        newbalanceDest FLOAT NOT NULL,
        isFraud INTEGER NOT NULL,
        isfraud_pred INTEGER NOT NULL,
        ID_Compte VARCHAR(50) NOT NULL,
        ID_Compte_recoie VARCHAR(50) NOT NULL,
        CONSTRAINT Transaction_ID_Compte_FK FOREIGN KEY (ID_Compte) REFERENCES Compte (ID),
        CONSTRAINT Transaction_ID_Compte_recoie_FK FOREIGN KEY (ID_Compte_recoie) REFERENCES Compte (ID)
        );""")

        connection.commit()                                    
        connection.close()
        connection = sql.connect("database/DB_fraud.db")
        curseur = connection.cursor()
        curseur.execute("""CREATE TABLE IF NOT EXISTS interface_user (
        ID INTEGER PRIMARY KEY NOT NULL,
        Nom_utilisateur VARCHAR(20) NOT NULL UNIQUE,
        mot_de_passe VARCHAR(20) NOT NULL
        );""")

        connection.commit()                                    
        connection.close()




class FraudDetectorFunctions:
    
    
    def __init__(self,model,file_path):
        self.model = model
        self.dataset = pd.read_csv(file_path, sep=";", decimal=',')
        

    
    
    def setindex(self):
        self.dataset.set_index("transactionId",inplace=True)
        self.dataset.reset_index(drop=True,inplace=True)
        self.dataset.insert(0, "transactionId", range(len(self.dataset)))
        
    
    def saisie_transaction(self, ID):
        
        ligne=self.dataset[self.dataset['transactionId'] == ID]
        cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        ligne[cols] = ligne[cols].astype(float)
        ligne['isFraud'] = ligne['isFraud'].astype(int) 
        
        
        predligne = 0
        if ligne.empty:
            raise ValueError(f"Transaction avec ID={ID} introuvable")
        
        
        
        if ligne["type"].iloc[0]=="CASH_OUT":
            ligne["type_CASH_OUT"]=1
            ligne["type_TRANSFER"]=0
            ligned=ligne.drop(columns=['type','nameOrig','nameDest'])
            ligned=ligned.drop(columns=['isFraud'])
            
            # ligned = ligned[self.model.feature_names_in_]
            
            predligne_array=self.model.predict(ligned)
            predligne = int(predligne_array[0])


        elif ligne["type"].iloc[0]=="TRANSFER":
            ligne["type_CASH_OUT"]=0
            ligne["type_TRANSFER"]=1
            ligned=ligne.drop(columns=['type','nameOrig','nameDest'])
            ligned=ligned.drop(columns=['isFraud'])
            
            # ligned = ligned[self.model.feature_names_in_]
            
            predligne_array=self.model.predict(ligned)
            predligne = int(predligne_array[0])
            
        
        connection = sql.connect("database/DB_fraud.db")
        curseur = connection.cursor()
        transaction_id = int(ligne['transactionId'].iloc[0])
        curseur.execute("SELECT 1 FROM Transactions WHERE ID = ?", (transaction_id,))
        exists = curseur.fetchone()
        connection.commit()
        connection.close()

        if exists :
             pass
        else :
            connection = sql.connect("database/DB_fraud.db")
            curseur = connection.cursor()
            
            curseur.execute("""INSERT INTO Transactions (ID, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isfraud_pred, ID_Compte, ID_Compte_recoie) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (int(ligne['transactionId'].iloc[0]),
            str(ligne['type'].iloc[0]),
            float(ligne['amount'].iloc[0]),
            float(ligne['oldbalanceOrg'].iloc[0]),
            float(ligne['newbalanceOrig'].iloc[0]),
            float(ligne['oldbalanceDest'].iloc[0]),
            float(ligne['newbalanceDest'].iloc[0]),
            int(ligne['isFraud'].iloc[0]),
            int(predligne),
            str(ligne['nameOrig'].iloc[0]),
            str(ligne['nameDest'].iloc[0])
            ))

            connection.commit()
            connection.close()

            connection = sql.connect("database/DB_fraud.db")
            curseur = connection.cursor()

            curseur.execute("""INSERT INTO Compte (ID) values (?) ON CONFLICT(ID) DO NOTHING""",
            (str(ligne['nameOrig'].iloc[0]),))
            curseur.execute("""INSERT INTO Compte (ID) values (?) ON CONFLICT(ID) DO NOTHING""",
            (str(ligne['nameDest'].iloc[0]),))

            connection.commit()
            connection.close()

        return predligne
        
    

    def random_echantillon(self):
        dataset_sampled_1 = self.dataset[self.dataset['isFraud'] == 1].sample(n=5, random_state=None)
        dataset_sampled_0 = self.dataset[self.dataset['isFraud'] == 0].sample(n=10, random_state=None)
        echantillon = pd.concat([dataset_sampled_1, dataset_sampled_0])
        return echantillon

    def charger_transactions(self):
        ech_transactions = self.random_echantillon()
        df = pd.DataFrame(ech_transactions)
        count=0
        f=0
        nf=0
        for i in df["transactionId"]:
            pred=self.saisie_transaction(i)
            count+=1
            if pred==0 :
                 nf+=1
            else:
                f+=1
        return count,f,nf   
        
    
    
    def afficher_stat_gen(self,file_path):
        
        """
        Statistiques globales du dataset
        """
        
        df=pd.read_csv(file_path, sep=";", decimal=',')
        
        stats = {}

        stats["nb_transactions"] = len(df)
        stats["nb_fraudes"] = int(df["isFraud"].sum())
        stats["taux_fraude"] = round(
            100 * df["isFraud"].mean(), 2
        )

        stats["montant_moyen"] = round(df["amount"].mean(), 2)
        stats["montant_max"] = round(df["amount"].max(), 2)

        # Fraudes par type
        fraude_par_type = (df.groupby("type")["isFraud"].value_counts().unstack(fill_value=0))
        df_fraud=pd.DataFrame(fraude_par_type)
        df_fraud.index.name="type" # decide de prendre pour index la colonne type
        df_fraud.columns=["nonfraude","fraude"]
        fraude_par_type_liste = df_fraud.reset_index().to_dict(orient="records")
        stats["fraude_par_type"] = fraude_par_type_liste

        # Transactions par type
        stats["transactions_par_type"] = (
            df["type"]
            .value_counts()
            .to_dict()
        )
        # stats["metrics"]= self.model.
        return stats

    


    def prediction_dataframe(self, df):
        """
        Prédiction pour un DataFrame complet
        Retourne : DataFrame avec isFraud_pred et proba
        """

        df_clean = df.copy()

        # Conversion types numériques
        cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        df_clean[cols] = df_clean[cols].astype(float)

        # One-hot encoding
        df_clean = pd.get_dummies(df_clean, columns=["type"], drop_first=False)

        # Alignement EXACT avec les features du modèle
        df_clean = df_clean.reindex(columns=self.model.feature_names_in_, fill_value=0)

        # Prédictions
        predictions = self.model.predict(df_clean)
        probabilities = self.model.predict_proba(df_clean)[:, 1]

        # Résultat final
        df_result = df.copy()
        df_result["isFraud_pred"] = predictions
        df_result["fraud_probability"] = probabilities

        return df_result


