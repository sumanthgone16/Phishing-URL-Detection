import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib

# Step 1: Load dataset
df = pd.read_csv('urlset.csv', encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

# Show columns for debugging
print("Available Columns in CSV:", list(df.columns))

# If dataset uses 'domain' and 'label'
if "url" in df.columns and "status" in df.columns:
    df.rename(columns={"url": "domain", "status": "label"}, inplace=True)

# Step 2: Drop missing values
df.dropna(subset=["domain", "label"], inplace=True)

# Step 3: Balance the dataset (upsample minority class)
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# For faster training, sample only 50,000 rows

# df_balanced = df_balanced.sample(50000, random_state=42)

print("Data balanced. Class distribution:\n", df_balanced['label'].value_counts())

# Step 4: Feature and label
X = df_balanced["domain"]
y = df_balanced["label"]

# Step 5: Vectorize using TF-IDF with character n-grams (reduced range)
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
X_vectorized = vectorizer.fit_transform(X)

# Step 6: Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, stratify=y, random_state=42
)

print("Data vectorized and split.")

# Step 7: Train a Random Forest model (optimized for speed)
model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(" Model trained successfully!")
print(f" Accuracy      : {acc:.4f}")
print(f"F1-score      : {f1:.4f}")
print(f"ROC-AUC Score : {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Predict on custom URLs
print("\n Testing on custom URLs:\n")

test_urls = [
    "paypal.com/login",                          # Legit
    "update-account.verify-paypal.com/login",    # Likely phishing
    "google.com",                                # Legit
    "secure.appleid.com-sign-in.security.com"    # Likely phishing
]

X_new = vectorizer.transform(test_urls)
predictions = model.predict(X_new)

for url, label in zip(test_urls, predictions):
    result = "Phishing" if label == 1 else "Legit"
    print(f"{url} --> {result}")

# Step 10: Save the model and vectorizer
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n Model and vectorizer saved successfully!")
