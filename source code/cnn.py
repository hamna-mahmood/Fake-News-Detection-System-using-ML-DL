from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from preprocess_dl import get_dl_data

def run_cnn():
    print(" Training CNN...")
    X_train, X_test, y_train, y_test, tokenizer = get_dl_data()
    
    cnn_model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=100),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)
    
    y_pred = (cnn_model.predict(X_test, verbose=0) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    print(f" CNN Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title("CNN Confusion Matrix")
    plt.show()
    
    return cnn_model

if __name__ == "__main__":
    run_cnn()