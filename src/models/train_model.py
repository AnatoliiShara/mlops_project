import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv('data/raw/synthetic_data.csv')

X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)
nb_accuracy = accuracy_score(y_test, nb_preds) 
print('Naive Bayes Model Accuracy: {:.2f}%'.format(nb_accuracy * 100)) 
print(classification_report(y_test, nb_preds))

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vec, y_train)
rf_preds = rf_model.predict(X_test_vec)
rf_accuracy = accuracy_score(y_test, rf_preds)
print('Random Forest Model Accuracy: {:.2f}%'.format(rf_accuracy * 100))
print(classification_report(y_test, rf_preds))

# Train a LightGBM model
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_vec, y_train)
lgbm_preds = lgbm_model.predict(X_test_vec)
lgbm_accuracy = accuracy_score(y_test, lgbm_preds)
print('LightGBM Model Accuracy: {:.2f}%'.format(lgbm_accuracy * 100))
print(classification_report(y_test, lgbm_preds))

# Save the best model
if max(nb_accuracy, rf_accuracy, lgbm_accuracy) == nb_accuracy:
    best_model = nb_model
    model_name = 'Naive_Bayes'
elif max(nb_accuracy, rf_accuracy, lgbm_accuracy) == rf_accuracy:
    best_model = rf_model
    model_name = 'Random_Forest'
else:
    best_model = lgbm_model
    model_name = 'LightGBM'
    
joblib.dump(best_model, f'models/{model_name}_model.pkl')
print(f'Best model ({model_name}) saved to models/{model_name}.pkl')