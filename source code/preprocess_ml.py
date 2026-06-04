import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

def preprocess_ml_data(max_features=5000):
    print(" Loading ML data...")
    
    fake_df = pd.read_csv("Fake.csv", low_memory=False)
    true_df = pd.read_csv("True.csv", low_memory=False)
    
    fake_df['content'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
    true_df['content'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df[['content', 'label']], true_df[['content', 'label']]], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)  # Fixed regex
        text = re.sub(r'\s+', ' ', text)  # Fixed regex
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    
    df['content'] = df['content'].apply(clean_text)
    df = df[df['content'] != ''].reset_index(drop=True)  # Filter empty content
    
    X = df['content']
    y = df['label']
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)
    
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f" ML Data ready: {X_train.shape}")
    return X_train, X_test, y_train, y_test, vectorizer