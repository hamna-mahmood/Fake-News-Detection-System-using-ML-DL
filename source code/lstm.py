from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess_dl import get_dl_data

def run_lstm():
    print(" Training LSTM...")
    X_train, X_test, y_train, y_test, tokenizer = get_dl_data()
    
    lstm_model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=100),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)
    
    y_pred = (lstm_model.predict(X_test, verbose=0) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    print(f" LSTM Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title("LSTM Confusion Matrix")
    plt.show()
    
    return lstm_model

if __name__ == "__main__":
    run_lstm()