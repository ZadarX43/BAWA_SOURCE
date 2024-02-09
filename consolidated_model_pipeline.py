
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pickle

# Load datasets
def load_data():
    cleaned_matches_data_df = pd.read_csv('path/to/cleaned_matches_data_df.csv')
    return cleaned_matches_data_df

# Preprocess data
def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    # Feature selection (Example: Selecting relevant features)
    features = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names
    return df[features]

# Split data
def split_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    model = LogisticRegression()  # Example model
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')

# Save model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Main execution
if __name__ == "__main__":
    df = load_data()
    df_preprocessed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, 'model.pkl')  # Save the trained model
