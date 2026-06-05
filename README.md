# Fake News Detection System

## Overview

The Fake News Detection System is a Machine Learning and Deep Learning project developed to classify news articles as **Real** or **Fake** based on their textual content.

The project uses a dataset containing over **44,000 news articles** and compares the performance of multiple Machine Learning and Deep Learning models. The objective is to identify the most effective approach for detecting misinformation in news content.

- Input: News article text  
= Output: Real / Fake prediction with confidence score
---

## Dataset

The dataset was obtained from Kaggle and contains:

* **21,000+ Fake News Articles**
* **23,000+ Real News Articles**
* Total: **44,000+ News Records**

The dataset was split into:

* **80% Training Data**
* **20% Testing Data**

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Performance Comparison

---

## Data Preprocessing

### Machine Learning Pipeline

For traditional Machine Learning models:

* Text cleaning
* Lowercasing
* Removal of punctuation and special characters
* Stopword removal
* TF-IDF Vectorization

### Deep Learning Pipeline

For Deep Learning models:

* Text cleaning
* Tokenization
* Sequence Generation
* Padding
* Numerical Representation of Text

---

## Models Implemented

### Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest

### Deep Learning Models

* Convolutional Neural Network (CNN)
* Long Short-Term Memory Network (LSTM)

---

### Confusion Matrix - Logistic Regression
![Logistic Regression Confusion Matrix](Images/confusion_matrix_logistic_regression.PNG)

### Confusion Matrix - Decision Tree
![Logistic Regression Confusion Matrix](Images/confusion_matrix_decision_tree.PNG)

### Confusion Matrix - Random Forest
![Logistic Regression Confusion Matrix](Images/confusion_matrix_random_forest.PNG)

### Confusion Matrix - CNN
![Logistic Regression Confusion Matrix](Images/confusion_matrix_cnn.PNG)

### Confusion Matrix - LSTM
![Logistic Regression Confusion Matrix](Images/confusion_matrix_lstm.PNG)


## Results

The performance of all models was evaluated on the testing dataset.

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | 98.86%     |
| Decision Tree       | 99.59%     |
| Random Forest       | **99.76%** |
| CNN                 | 98.35%     |
| LSTM                | 97.80%     |

Among all evaluated models, **Random Forest achieved the highest test accuracy of 99.76%**, making it the best-performing model for this dataset.

---

## Model Performance (F1-Score)

| Model                     | F1-Score | Type |
|--------------------------|----------|------|
| Logistic Regression      | 0.99     | ML   |
| ⭐ Random Forest         | 1.00     | ML   |
| Decision Tree            | 0.99     | ML   |
| LSTM                     | 0.97     | DL   |
| CNN (1D Convolution)     | 0.99     | DL   |


## Project Structure
```text
Fake-News-Detection-System/
│
├── source_code/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── cnn.py
│   ├── lstm.py
│   ├── main.py
│   └── preprocess_dl.py
│   └── preprocess_ml.py
│
├── images/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_lstm.png
│   ├── confusion_matrix_cnn.png
│   ├── comparison.png
│   └── sample_output.png
│
├── requirements.txt
├── README.md
└── .gitignore

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* Matplotlib
* Seaborn

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Future Improvements

* Integration of Transformer-based models such as BERT
* Real-time news verification
* Web-based user interface
* Deployment using Flask or Streamlit
* Multi-language fake news detection

---
## Dataset Credit:
- C. Bisaillon, “Fake and Real News Dataset,” Kaggle, 2019.
- Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- License: CC BY-NC-SA 4.0

## Author
- Developed as a team.
- Hamna Mahmood
- linkedin.com/in/hamnamahmood
- hamnamahmood004@gmail.com
