from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import os
import pickle
import pandas as pd

dataset_df = pd.read_csv("datasets/dataset.csv")
for col in dataset_df.columns:
    if col == "SMILES":
        continue
    dataset_df[col] = dataset_df[col].fillna(0)

X = dataset_df[["MaxEStateIndex", "MinEStateIndex"]]
y = dataset_df["λmax"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Ridge()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print(f"Train Mean Squared Error: {mse(y_train, pred_train)}")
print(f"Test Mean Squared Error: {mse(y_test, pred_test)}")

pickle.dump(model, open(os.path.dirname(__file__) +  "/model.pkl", "wb"))