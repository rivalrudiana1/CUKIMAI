import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("dataset_text.csv")

X_text = data["teks_masalah"]
y_label = data["label_masalah"]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vector = vectorizer.fit_transform(X_text)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_label)

model = LogisticRegression(max_iter=1000)
model.fit(X_vector, y_encoded)

with open("model_knn.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Training Logistic Regression selesai")
