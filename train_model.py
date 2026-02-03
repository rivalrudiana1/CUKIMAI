import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
print(os.getcwd())


data = pd.read_csv("dataset.csv")

X = data.drop("label_masalah", axis=1)
y = data["label_masalah"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y_encoded)

with open("model_knn.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Training selesai. Model dan encoder tersimpan.")
