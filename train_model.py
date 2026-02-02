import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def clean_title(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    # Load data
    df = pd.read_csv("data/products.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean titles
    df["clean_title"] = df["Product Title"].apply(clean_title)

    # Drop missing values
    df = df.dropna(subset=["clean_title", "Category Label"])

    X = df["clean_title"]
    y = df["Category Label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model trained successfully. Accuracy: {acc:.4f}")

    # Save model
    joblib.dump(model, "product_category_model.pkl")
    print("Model saved as product_category_model.pkl")


if __name__ == "__main__":
    main()
