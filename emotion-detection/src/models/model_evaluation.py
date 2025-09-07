import numpy as np
import pandas as pd
import pickle
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load model
clf = pickle.load(open('./models/model.pkl', 'rb'))

# Load test data
test_data = pd.read_csv('./data/external/test_bow.csv')
X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
metrics_dict = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'auc': roc_auc_score(y_test, y_pred_proba)
}

# Define metrics file path (DVC expects this)
metrics_file = 'reports/metrics.json'

# Ensure the reports directory exists
# os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

# Write metrics to JSON
with open(metrics_file, 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print(f"Metrics saved to {metrics_file}")
