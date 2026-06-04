import pandas as pd
import re
import string
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_dl_data():
    print("  Loading DL data...")
    fake_df = pd.read_csv("Fake.csv", low_memory=False)
    true_df = pd.read_csv("True.csv", low_memory=False)
    
    fake_df['content'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
    true_df['content'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
    
    fake_df['label'] = 0
    true_df['label'] = 1
    
    df = pd.concat([fake_df[['content', 'label']], true_df[['content', 'label']]], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def clean_text(text):
        if pd.isna(text):
            return "empty"
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)  # Fixed regex
        text = re.sub(r'\s+', ' ', text)  # Fixed regex
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    
    df['content'] = df['content'].apply(clean_text)
    df = df[df['content'] != 'empty'].reset_index(drop=True)
    
    X = df['content'].tolist()
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)
    
    pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
    
    print(f" DL Data ready: {X_train_pad.shape}")
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer