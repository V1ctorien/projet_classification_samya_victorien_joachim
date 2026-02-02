import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import io
import base64
import os

model = joblib.load('model_test/rf_model.pkl')

def olap():
# Pour Jupyter interactif
    # %matplotlib qt
    # Charger le CSV
    df = pd.read_csv("data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv",sep=";",decimal=",")

    # Création d’un mapping type → code
    type_mapping = {t: i for i, t in enumerate(df["type"].unique())}

    df["type_code"] = df["type"].map(type_mapping)



    # Séparer les fraudes / non-fraudes
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0]

    # Créer figure 3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Nuage points non-fraude
    ax.scatter(
        non_fraud["type_code"],
        non_fraud["step"],
        non_fraud["amount"],
        c='blue',
        label="Non fraude",
        alpha=0.5,
        s=20
    )

    # Nuage points fraude
    ax.scatter(
        fraud["type_code"],
        fraud["step"],
        fraud["amount"],
        c='red',
        label="Fraude",
        alpha=0.8,
        s=40
    )

    # Labels et titre
    ax.set_xlabel("type")
    ax.set_ylabel("step")
    ax.set_zlabel("montant")
    ax.set_title(" Step vs type vs montant")

    # Légende
    ax.legend()
    buf = io.BytesIO() # Créer un buffer mémoire
    plt.savefig(buf, format='png', bbox_inches='tight') # Sauvegarder la figure dans le buffer
    buf.seek(0) # Revenir au début du buffer
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') # Encoder en base64
    plt.close()
    return img_base64

def fraudpartype():
    df = pd.read_csv("data\data_train_set.csv",sep=";",decimal=",")
    table=pd.crosstab(df["type"], df["isFraud"])
    grph=table.div(table.sum(1).astype(float),axis=0 )
    ax=grph.plot(kind="bar", stacked=True)
    ax.set_ylim(0.9,1)
    ax.legend(
    title="Fraude",
    bbox_to_anchor=(1.05, 1),
    loc="upper left")
    
    
    
    plt.title("Proportion de fraude par type de transaction")
    plt.xlabel("Type de transaction")
    plt.ylabel("Proportion de Fraude")
    plt.tight_layout()

    buf = io.BytesIO() # Créer un buffer mémoire
    plt.savefig(buf, format='png', bbox_inches='tight') # Sauvegarder la figure dans le buffer
    buf.seek(0) # Revenir au début du buffer
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') # Encoder en base64
    plt.close()
    return img_base64

def matrice_correlation():
    
    df = pd.read_csv(r"data\data_train_set.csv",sep=";",decimal=",")
    dft = df.drop(["nameOrig", "nameDest"], axis=1).copy()

    cols = [
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]
    dft[cols] = dft[cols].astype(float)
    dft = pd.get_dummies(dft, columns=["type"], drop_first=True)

    # Bool → int
    dft = dft.replace({True: 1, False: 0})
    correlation_matrix = dft.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Matrice de corrélation des caractéristiques")

    buf = io.BytesIO() # Créer un buffer mémoire
    plt.savefig(buf, format='png', bbox_inches='tight') # Sauvegarder la figure dans le buffer
    buf.seek(0) # Revenir au début du buffer
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') # Encoder en base64
    plt.close()
    return img_base64


def matrice_confusion(file_path):
    
    df = pd.read_csv(file_path,sep=";",decimal=",")
    dft=df.drop(["nameOrig","nameDest"], axis=1).copy()
    cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    dft[cols] = dft[cols].astype(float)
    dft=pd.get_dummies(dft, columns=["type"], drop_first=False,dtype=int)
    dft=dft.drop(["type_DEBIT","type_PAYMENT","type_CASH_IN"], axis=1)
    


    
    X = dft.drop("isFraud", axis=1)
    y = dft["isFraud"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    
    buf = io.BytesIO() # Créer un buffer mémoire
    plt.savefig(buf, format='png', bbox_inches='tight') # Sauvegarder la figure dans le buffer
    buf.seek(0) # Revenir au début du buffer
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8') # Encoder en base64
    plt.close()
    return img_base64



