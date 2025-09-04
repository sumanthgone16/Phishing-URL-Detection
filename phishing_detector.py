import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib

df = pd.read_csv('urlset.csv', encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)
print("Available Columns in CSV:", list(df.columns))
if "url" in df.columns and "status" in df.columns:
    df.rename(columns={"url": "domain", "status": "label"}, inplace=True)

df.dropna(subset=["domain", "label"], inplace=True)

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

X = df_balanced["domain"]
y = df_balanced["label"]

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, stratify=y, random_state=42
)
print("Data vectorized and split.")

model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

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

print("\n Testing on custom URLs:\n")

test_urls = [
    "paypal.com/login",
    "update-account.verify-paypal.com/login",
    "google.com",
    "secure.appleid.com-sign-in.security.com"
]

X_new = vectorizer.transform(test_urls)
predictions = model.predict(X_new)
for url, label in zip(test_urls, predictions):
    result = "Phishing" if label == 1 else "Legit"
    print(f"{url} --> {result}")

joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n Model and vectorizer saved successfully!")
