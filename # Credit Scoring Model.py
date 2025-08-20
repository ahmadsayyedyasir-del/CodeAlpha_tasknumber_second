# Credit Scoring Model
# Logistic Regression, Decision Tree, Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ======================
# Step 1: Create Sample Dataset
# ======================
data = {
    'income': [50000, 60000, 35000, 80000, 20000, 30000, 100000, 40000, 70000, 45000],
    'debts': [20000, 15000, 25000, 10000, 15000, 20000, 5000, 18000, 12000, 17000],
    'payment_history': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1],  # 1 = good, 0 = bad
    'default': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]           # 1 = bad credit, 0 = good credit
}

df = pd.DataFrame(data)

# Features and Target
X = df[['income', 'debts', 'payment_history']]
y = df['default']

# ======================
# Step 2: Split Data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# Step 3: Models
# ======================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# ======================
# Step 4: Train + Evaluate
# ======================
for name, model in models.items():
    if name == "Logistic Regression":
        # scale only for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
