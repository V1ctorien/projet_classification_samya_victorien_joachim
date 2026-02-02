import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as xg
import joblib
from imblearn.under_sampling import RandomUnderSampler

def nettoyage(table):
            dft=table.drop(["nameOrig","nameDest"], axis=1).copy()
            cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            dft[cols] = dft[cols].astype(float)
            dft=pd.get_dummies(dft, columns=["type"], drop_first=False,dtype=int)
            dft=dft.drop(["type_DEBIT","type_PAYMENT","type_CASH_IN"], axis=1)
            
            return dft
df=pd.read_csv("data\data_train_set.csv", sep=";", decimal=',')
data=nettoyage(df)
print(data.info())



x,y= data.drop("isFraud", axis=1), data["isFraud"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
rus = RandomUnderSampler(
    sampling_strategy=0.5,
    random_state=42
)
X_train_under, y_train_under = rus.fit_resample(x_train, y_train)
print(X_train_under.head())

# clf=xg(subsample= 0.6, max_depth= 9, learning_rate= 0.2, gamma= 0.1, colsample_bytree= 0.8, eval_metric='logloss')
# model=clf.fit(X_train_under, y_train_under)
# joblib.dump(model, "model_test/rf_model.pkl")
