#------------------------------- RANDOM FOREST------------------------ #

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest(X_train, X_test, y_train, y_test):

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print(" Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Fake', 'True'],
                yticklabels=['Fake', 'True'])

    plt.title("Random Forest Confusion Matrix")
    plt.show()
    return rf_model