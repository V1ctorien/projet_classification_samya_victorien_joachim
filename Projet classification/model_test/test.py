import pandas as pd
from sklearn.model_selection import train_test_split
from fraud import traitement_data as td
from fraud import FraudDetectorFunctions as ff
from fraud import create_bdd
from sklearn.ensemble import RandomForestClassifier
from fraud import FraudDetectorFunctions as ff
import joblib
model = joblib.load('model_test/rf_model.pkl')
from EDA import matrice_confusion
# sauvegarde du csv sans guillemets

#input_file = "Projet classification\data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv"
# output_file = "C:\\Users\\samya.taqi\\Documents\\Projet classification\\data\\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv"

# with open(input_file, "r", encoding="utf-8") as f:
#      lines = f.readlines()

# cleaned_lines = [line.strip().strip('"') for line in lines]

# with open(output_file, "w", encoding="utf-8") as f:
#      f.write("\n".join(cleaned_lines))
#df=pd.read_csv(output_file, sep=";")
# df.head()
# df.info()
# df.describe()
# df=pd.read_csv("Projet classification\\data\\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv", sep=";")
#df.drop("transactionId", axis=1, inplace=True)
# df.drop("step", axis=1, inplace=True)
# x,y= df.drop("isFraud", axis=1), df["isFraud"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# x_test["isFraud"] = y_test.values
# x_train["isFraud"] = y_train.values
# x_test.to_csv("Projet classification\\data\\data_test_set.csv", sep=";", index=False)
# x_train.to_csv("Projet classification\\data\\data_train_set.csv", sep=";", index=False)
# print(x_train.info())
# print(x_test.info())
# dataa=pd.read_csv("Projet classification\data\data_train_set.csv", sep=";",decimal=',')
# dataa.info()
# df=pd.read_csv("data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv", sep=";",decimal=',')
# print(df.head())
# create_bdd()
# ff.insert_utilisateur("database/DB_fraud.db", "Taqi", "Samya", "0612345678")
# df=nettoyage(df)
# x,y= df.drop("isFraud", axis=1), df["isFraud"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# model=RandomForestClassifier(n_estimators=100, random_state=0, min_samples_leaf=1, min_samples_split=2)
# model=model.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# print(y_pred.shape())

# td.traitement_splt_csv("data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv")


# print(ff(model=model,file_path=r"data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv" ).afficher_stat_gen(file_path=r"data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv"))

# data_test=pd.read_csv("uploads\data_test_set.csv",sep=";",decimal=",")
# data_test.set_index("transactionId",inplace=True)
# data_test.reset_index(drop=True,inplace=True)


# data_test.insert(0, "transactionId", range(len(data_test)))
# print(data_test.head())

# data_test=ff(model=model,file_path="uploads\data_test_set.csv")
# data_test.setindex()
# dataf=data_test.dataset[data_test.dataset["isFraud"] == 1]
# print(dataf.head())

        
# y_pred=None
# transID=12
                               
# if transID not in data_test.dataset["transactionId"]:
#     print("ID non valide")

# else:
#             y_pred=data_test.saisie_transaction(transID)


# print(y_pred)


# [26.32.199.357.365] sont fraude
matrice_confusion("data\data_train_set.csv")