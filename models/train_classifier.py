import sys
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
import pandas as pd
import numpy as np
import string
from nltk import pos_tag, ne_chunk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine) 
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    #remove punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    #lemmatize, convert to lowercase, remove leading/trailing white space
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text).lower().strip()
    #tokenize
    words = word_tokenize(text)
    #stop words removal
    words = [w for w in words if w not in stopwords.words("english")]
    #Stemming
    words = [PorterStemmer().stem(w) for w in words]
    clean_tokens = []
    for tok in words:
        clean_tokens.append(tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier())),
                    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    
    #predict on test data
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
