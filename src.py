import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# Load dataset
# -------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample rows:\n", train_df.head())

# -------------------------
# Stratified Subsample (حفظ نسبت کلاس‌ها)
# -------------------------
train_df = train_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(5000, len(x)), random_state=42)
)
test_df = test_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(1000, len(x)), random_state=42)
)

print("Subsampled Train:", train_df.shape)
print("Subsampled Test:", test_df.shape)

# -------------------------
# Prepare features and labels
# -------------------------
X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# -------------------------
# Vectorization (CountVectorizer)
# -------------------------
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Train Multinomial Naive Bayes
# -------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Model training finished.")

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)
print("Prediction done.")

# -------------------------
# Evaluation metrics
# -------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nEvaluation Metrics:")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix (Normalized)
# -------------------------
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels_text = ["World", "Sports", "Business", "Sci/Tech"]

plt.figure(figsize=(7,6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels_text, yticklabels=labels_text)
plt.title("Normalized Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# -------------------------
# Metrics Bar Chart
# -------------------------
metrics_values = [acc, prec, rec, f1]
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]

plt.figure(figsize=(8,5))
y_min = min(metrics_values) - 0.02  
y_max = max(metrics_values) + 0.05 
plt.ylim(y_min, y_max)

colors = ['skyblue', 'orange', 'green', 'red']
bars = plt.bar(metrics_names, metrics_values, color=colors)

for bar, value in zip(bars, metrics_values):
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        value + 0.002, 
        f"{value:.3f}", 
        ha='center', 
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Model Performance Metrics (CountVectorizer + Stratified Sampling)", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# Test custom text
# -------------------------
new_text = ["The stock market reacted positively to the new economic policy"]
new_vec = vectorizer.transform(new_text)
prediction = model.predict(new_vec)
proba = model.predict_proba(new_vec)

print("Custom news prediction:", prediction[0])
print("Prediction probabilities:", proba)
