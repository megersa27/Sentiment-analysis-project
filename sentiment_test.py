import numpy as np
import pandas as pd

# Scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Daataseetii Yaada Filmii (Sample Movie Reviews Dataset)
data = {
    'review': [
        'This movie was fantastic and absolutely wonderful I loved every minute of it',
        'Terrible waste of time the acting was horrible and the plot made no sense',
        'An absolute masterpiece of cinema with brilliant performances and stunning visuals',
        'Boring and dull from start to finish I fell asleep halfway through',
        'Brilliant direction and an engaging storyline that keeps you hooked until the end',
        'Awful film with a dreadful script and terrible characters'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

print("--- Daataa Yaada Filmii Jalqabaa ---")
print(df.head())

# 2. Label gara lakkofsaatti jijjiiruu (positive: 1, negative: 0)
df['sentiment_num'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['review']
y = df['sentiment_num']

# Training fi Testing qooduu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Text Vectorization (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 4. Modelii Leenjisuu (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
print("\nModeliin Sentiment Analysis milkaa'inaadhaan leenjifiameera!")

# 5. Madaallii (Evaluation)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Bu'aa Madaallii Modelichaa ---")
print(f"Accuracy (Sirrummaa): {accuracy * 100:.2f}%")

# ==========================================
# 6. YAADA FILMII HAARAA TILMAAMA BAASUU
# ==========================================
new_reviews = [
    "An incredible and breathtaking movie that I will watch again",
    "Worst movie ever made complete garbage and waste of money"
]

# Text-to-Vector jijjiiruu
new_reviews_vectorized = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vectorized)

print("\n--- Tilmaama Yaada Haaraa ---")
for rev, pred in zip(new_reviews, predictions):
    result = "POSITIVE 😍 (Miira Gaarii)" if pred == 1 else "NEGATIVE 😡 (Miira Gadhee)"
    print(f"Yaada: '{rev}'\nBu'aa: {result}\n")