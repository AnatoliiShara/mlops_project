import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the synthetic data
@pytest.fixture
def load_data():
    df = pd.read_csv('data/raw/synthetic_data.csv')
    return df

# Test that the synthetic data is loaded correctly
def test_load_data(load_data):
    df = load_data
    assert len(df) == 10000, 'Data should have 10000 records'
    assert len(df.columns) == 3, 'Data should have 3 columns'
    
# test that TFI-DF vectorizer works correctly
def test_vectorizer(load_data):
    df = load_data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    assert X_train_vec.shape[1] == 1000, 'Should have 1000 features'
    
# test that Naive Bayes model is trained correctly
def test_naive_bayes_train(load_data):
    df = load_data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    assert nb_model, "NaiveBayes should be trained succesdfully"
    
# test that Random Forest model is trained correctly
def test_random_forest_train(load_data):
    df = load_data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_vec, y_train)
    assert rf_model, "RandomForest should be trained successfully"
    
# test that LightGBM model is trained correctly
def test_lightgbm_train(load_data):
    df = load_data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(X_train_vec, y_train)
    assert lgbm_model, "LightGBM should be trained successfully"
    
# test that the model is saved correctly
def test_model_save_and_load():
    #create a dummy model
    dummy_model = MultinomialNB()
    joblib.dump(dummy_model, 'models/dummy_model.pkl')
    # load the dummy model
    loaded_model = joblib.load('models/dummy_model.pkl')
    assert isinstance(loaded_model, MultinomialNB), 'Model should be an instance of MultinomialNB'
    # remove the dummy model
    os.remove('models/dummy_model.pkl')
