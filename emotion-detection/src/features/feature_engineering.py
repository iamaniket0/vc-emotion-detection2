import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# Load processed data
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# Fill missing values
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Extract features and labels
X_train, y_train = train_data['content'].values, train_data['sentiment'].values
X_test, y_test = test_data['content'].values, test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer) with max 50 features
vectorizer = CountVectorizer(max_features=50)

# Fit and transform training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform test data
X_test_bow = vectorizer.transform(X_test)

# Convert to DataFrame and add labels
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Save features
data_path = os.path.join("data", "external")
os.makedirs(data_path, exist_ok=True)
train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

