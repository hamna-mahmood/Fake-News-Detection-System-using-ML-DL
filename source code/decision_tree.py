#-------------------------DECISION TREE--------------------------#

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_decision_tree(X_train, X_test, y_train, y_test):

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    print(" Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Fake', 'True'],
                yticklabels=['Fake', 'True'])

    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    return dt_model