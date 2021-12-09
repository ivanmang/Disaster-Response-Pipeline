import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
import re
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', 'sqlite:///'+database_filepath)  
    X = df.message
    Y = df.iloc[:, 4:]
    return X, Y

def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
