from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor

from rdkit import Chem
from rdkit.Chem import AllChem

import os
import pickle
import pandas as pd
import numpy as np
import time

def get_morgan_fingerprint(smile, radius=2, bits=1024):
    mol_ = Chem.MolFromSmiles(smile)
    bit_morgan_ = {}
    if mol_ == None:
        return [0] * bits
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_, radius=radius, nBits=bits, bitInfo=bit_morgan_)
    return np.array(fp)

start_time = time.perf_counter()
dataset_df = pd.read_csv("datasets/dataset.csv")
for col in dataset_df.columns:
    if col == "SMILES":
        continue
    dataset_df[col] = dataset_df[col].fillna(0)

# モルガンフィンガープリントの作成
print("経過時間:{}, Start Make Morgan Fingerprint".format(time.perf_counter()- start_time))
dataset_df["morgan_fingerprint"] = dataset_df["SMILES"].map(get_morgan_fingerprint)
print("経過時間:{}, Finish Make Morgan Fingerprint".format(time.perf_counter()- start_time))

# データセットの定義と分割
X = np.array(dataset_df["morgan_fingerprint"].values.tolist())
X_rest = dataset_df.drop(['SMILES','λmax',"morgan_fingerprint"], axis=1).values
# X = dataset_df.drop(['SMILES','λmax'], axis=1).values
X = np.hstack([X, X_rest])
y = dataset_df["λmax"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 回帰モデル
print("経過時間:{}, Start Training".format(time.perf_counter()- start_time))
model = MLPRegressor(hidden_layer_sizes=(128,128,))
# model = LinearRegression()
model.fit(X_train, y_train)
print("経過時間:{}, Finish Training".format(time.perf_counter()- start_time))

# 予測
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print(f"Train Mean Squared Error: {mse(y_train, pred_train)}")
print(f"Test Mean Squared Error: {mse(y_test, pred_test)}")

pickle.dump(model, open(os.path.dirname(__file__) +  "/model.pkl", "wb"))