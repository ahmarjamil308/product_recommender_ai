import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("product_data.csv")
df["text"] = df["name"] + " " + df["description"]

# Step 2: Split the data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.3, random_state=42)

# Step 3: Build classification pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"Model Accuracy: {accuracy}%")

# Step 6: Save model and accuracy
joblib.dump(model, "model.pkl")
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

# Step 7: Save TF-IDF vectorizer and matrix for similarity-based recommendations
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["text"])

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
