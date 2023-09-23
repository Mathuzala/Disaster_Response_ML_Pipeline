import sys
import pandas as pd
import sqlite3
import nltk
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump

for dataset in ['punkt', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{dataset}')
    except LookupError:
        nltk.download(dataset)

# Check if punkt tokenizer is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK data tokenizers not found. Downloading now...")
    nltk.download('punkt')
    print("Download complete!")

database_filepath = './DisasterResponse.db'

def load_data(database_filepath):
    """Load data from SQLite database.
    
    Parameters:
    - database_filepath: String, path to the SQLite database file.
    
    Returns:
    - X: DataFrame, feature data (messages).
    - Y: DataFrame, target data (categories).
    - category_names: List, names of the categories.
    """
    #connect to the SQLite database
    conn = sqlite3.connect(database_filepath)
    
    #read data into a DataFrame
    df = pd.read_sql_query("SELECT * FROM DisasterData", conn)

    #close the connection
    conn.close()
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    """Tokenize and normalize a text string.
    
    Parameters:
    - text: String, input text to be tokenized.
    
    Returns:
    - tokens: List, tokenized version of the input text.
    """
    tokens = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()
    return tokens

def build_model():
    """Build a machine learning pipeline using RandomForestClassifier.
    
    Returns:
    - model: Pipeline, multi-output classification model.
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model's performance on test data.
    
    Parameters:
    - model: Pipeline, trained machine learning model.
    - X_test: DataFrame, feature test data.
    - Y_test: DataFrame, target test data.
    - category_names: List, names of the categories.
    
    Returns:
    - None
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i], "\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))

def save_model(model, model_filepath):
    """Save the trained model to a file.
    
    Parameters:
    - model: Pipeline, trained machine learning model.
    - model_filepath: String, path to save the model.
    
    Returns:
    - None
    """
    dump(model, model_filepath)

def main():
    """Main function to orchestrate model training, evaluation, and saving.
    
    - Loads data from SQLite database
    - Splits data into training and test sets
    - Builds and trains a model
    - Evaluates the model on the test set
    - Saves the trained model to a file
    
    Returns:
    - None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        parameters = {
        'tfidf__use_idf': [True],
        'vect__max_df': [0.75, 1.0],
        'vect__max_features': [None, 5000],
        'clf__estimator__n_estimators': [100],
        'clf__estimator__min_samples_split': [2]
        }


        cv = GridSearchCV(model, param_grid=parameters, verbose=1, n_jobs=-1)
        
        print('Training model...')
        cv.fit(X_train, Y_train)

        print('Best Parameters:', cv.best_params_)
        
        print('Evaluating model...')
        evaluate_model(cv.best_estimator_, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              
              'as the first argument and the filepath of the pickle file to '              
              'save the model to as the second argument. \n\nExample: python '              
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
