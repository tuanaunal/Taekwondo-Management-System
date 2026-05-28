import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import os

csv_path = os.path.join(os.path.dirname(__file__), "output", "reports", "batch_summary_detailed.csv")
df = pd.read_csv(csv_path)

# Prepare data
# Replace inf with 999
df['min_distance'] = df['min_distance'].replace(float('inf'), 999)

X = df[['net_disp', 'max_acc', 'min_distance', 'max_overlap']]
y = (df['true_label'] == 'Real_Hit').astype(int)

# Train decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Predictions
y_pred = clf.predict(X)

print(f"Decision Tree Accuracy: {accuracy_score(y, y_pred) * 100:.1f}%")

gh_correct = ((y == 0) & (y_pred == 0)).sum()
rh_correct = ((y == 1) & (y_pred == 1)).sum()

print(f"Ghost Hit Accuracy: {gh_correct}/60 ({(gh_correct/60)*100:.1f}%)")
print(f"Real Hit Accuracy: {rh_correct}/60 ({(rh_correct/60)*100:.1f}%)")

print("\n--- DECISION RULES ---")
rules = export_text(clf, feature_names=list(X.columns))
print(rules)
