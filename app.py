from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved model, vectorizer, TF-IDF matrix, and dataset
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
df = pd.read_csv("product_data.csv")

# Load accuracy
try:
    with open("accuracy.txt") as f:
        model_accuracy = f.read()
except:
    model_accuracy = "Not available"

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    category = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form["query"]

        # Predict category
        category = model.predict([input_text])[0]

        # Recommend similar products
        input_vector = vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1]

        seen = set()
        for idx in top_indices:
            item = df.iloc[idx]
            key = (item["name"], item["description"])
            if item["category"] == category and key not in seen:
                results.append({"name": item["name"], "description": item["description"]})
                seen.add(key)
            if len(results) == 5:
                break

    return render_template("index.html", category=category, results=results, query=input_text, accuracy=model_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
