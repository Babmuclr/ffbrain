import sys
import os
import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

def get_morgan_fingerprint(smile, radius=2, bits=1024):
    mol_ = Chem.MolFromSmiles(smile)
    bit_morgan_ = {}
    if mol_ == None:
        return [0] * bits
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_, radius=radius, nBits=bits, bitInfo=bit_morgan_)
    return np.array(fp)

input_data = []
for line in sys.stdin:
    input_data.append(line.strip().split(","))

input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
input_df["morgan_fingerprint"] = input_df["SMILES"].map(get_morgan_fingerprint)
input_df = input_df.fillna(0)

X = np.array(input_df["morgan_fingerprint"].values.tolist())
X_rest = input_df.drop(["SMILES","morgan_fingerprint"], axis=1).values
X = np.hstack([X, X_rest])

model = pickle.load(open(os.path.dirname(__file__) + "/model.pkl", "rb"))
y_pred = model.predict(X)

for val in y_pred:
    print(val)

