
import numpy as np
import pandas as pd 

def nettoyage(table):
            dft=table.drop(["nameOrig","nameDest"], axis=1).copy()
            cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            dft[cols] = dft[cols].astype(float)
            #dft.loc[(dft.oldbalanceDest==0) & (dft.newbalanceDest==0) & ((dft.amount) != 0), ["oldbalanceDest", "newbalanceDest"]] = -1
            #dft.loc[(dft.oldbalanceOrg==0) & (dft.newbalanceOrig==0) & ((dft.amount) != 0),  ["oldbalanceOrg", "newbalanceOrig"]] = np.nan
            return dft
    