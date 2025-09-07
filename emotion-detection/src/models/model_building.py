import numpy as np
import pandas as pd
import os 
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import yaml

# Load parameters
params = yaml.safe_load(open('params1.yaml'))['model_building']

# Load training features
train_data = pd.read_csv('./data/external/train_bow.csv')  # Ensure this exists

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# Train Gradient Boosting model
gb_model = GradientBoostingClassifier(
    learning_rate=params['learning_rate'],
    n_estimators=params['n_estimators'],
    random_state=42
)
gb_model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
pickle.dump(gb_model, open('models/model.pkl', 'wb'))

