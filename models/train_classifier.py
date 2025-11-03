import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# pip install nltk
# pip install scikit-learn
#>>> import nltk
#>>> nltk.download('punkt_tab')
#    nltk.download('wordnet')

# We have already used:
# python .\data\process_data.py .\data\disaster_messages.csv .\data\disaster_categories.csv .\data\DisasterResponse.db

# The suggestion is:
# python train_classifier.py ../data/DisasterResponse.db classifier.pkl

# Considering the existing folders, we should:
# python .\models\train_classifier.py .\data\DisasterResponse.db classifier.pkl

def load_data(database_filepath):
    """
    Load data from SQLite database and prepare feature and target variables.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        X (pd.Series): Messages (features)
        Y (pd.DataFrame): Categories (targets)
        category_names (list): List of category column names
    """
    from sqlalchemy import create_engine
    import pandas as pd

    # Connect to the SQLite database
    engine = create_engine(f"sqlite:///{database_filepath}")

    # Read the DisasterResponse table
    df = pd.read_sql_table("DisasterResponse", engine)

    # Define features and target variables
    X = df["message"]
    Y = df.iloc[:, 4:]  # assuming first 4 columns are id, message, original, genre
    category_names = Y.columns.tolist()

    return X, Y, category_names

def tokenize(text):
    """
    Normalize, tokenize, and lemmatize text string.
    
    Args:
        text (str): Input message text.
    
    Returns:
        tokens (list): List of cleaned tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and perform grid search.

    Returns:
        model (GridSearchCV): Grid search model pipeline.
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    
    parameters = {
        "clf__estimator__n_estimators": [50, 100],
        "clf__estimator__min_samples_split": [2, 4]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance using classification report.
    
    Args:
        model: Trained model
        X_test: Test feature data
        Y_test: True labels
        category_names: List of category names
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save model as a pickle file.
    
    Args:
        model: Trained model object
        model_filepath (str): Path to save model pickle file
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('v.0.0.1')
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        model.fit(X_train, Y_train)
        
        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)
        
        print("Trained model saved!")
    else:
        print("Please provide the filepath of the disaster messages database as the first argument and "
              "the filepath of the pickle file to save the model to as the second argument. "
              "\n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()

'''
              precision    recall  f1-score   support

           0       0.96      1.00      0.98      5018
           1       0.00      0.00      0.00       225

    accuracy                           0.96      5243
   macro avg       0.48      0.50      0.49      5243
weighted avg       0.92      0.96      0.94      5243

Category: weather_related
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      3771
           1       0.86      0.66      0.75      1472

    accuracy                           0.88      5243
   macro avg       0.87      0.81      0.83      5243
weighted avg       0.88      0.88      0.87      5243

Category: floods
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      4812
           1       0.93      0.43      0.59       431

    accuracy                           0.95      5243
   macro avg       0.94      0.72      0.78      5243
weighted avg       0.95      0.95      0.94      5243

Category: storm
              precision    recall  f1-score   support

           0       0.95      0.99      0.97      4764
           1       0.80      0.44      0.57       479

    accuracy                           0.94      5243
   macro avg       0.87      0.72      0.77      5243
weighted avg       0.93      0.94      0.93      5243

Category: fire
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      5190
           1       1.00      0.02      0.04        53

    accuracy                           0.99      5243
   macro avg       1.00      0.51      0.52      5243
weighted avg       0.99      0.99      0.99      5243

Category: earthquake
              precision    recall  f1-score   support

           0       0.98      0.99      0.98      4728
           1       0.89      0.79      0.83       515

    accuracy                           0.97      5243
   macro avg       0.93      0.89      0.91      5243
weighted avg       0.97      0.97      0.97      5243

Category: cold
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      5139
           1       0.88      0.07      0.12       104

    accuracy                           0.98      5243
   macro avg       0.93      0.53      0.56      5243
weighted avg       0.98      0.98      0.97      5243

Category: other_weather
              precision    recall  f1-score   support

           0       0.95      1.00      0.97      4976
           1       0.56      0.02      0.04       267

    accuracy                           0.95      5243
   macro avg       0.75      0.51      0.51      5243
weighted avg       0.93      0.95      0.93      5243

Category: direct_report
              precision    recall  f1-score   support

           0       0.86      0.98      0.92      4233
           1       0.84      0.34      0.48      1010

    accuracy                           0.86      5243
   macro avg       0.85      0.66      0.70      5243
weighted avg       0.86      0.86      0.84      5243

Saving model...
    MODEL: classifier.pkl
Trained model saved!
'''