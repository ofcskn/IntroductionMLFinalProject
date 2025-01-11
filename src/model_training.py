from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models on the dataset.
    """
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Predict on test data
        y_pred = model.predict(X_test)
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.2f}")
    return results

# For testing this module standalone
if __name__ == "__main__":
    # Placeholder data (replace with real data when testing)
    X_train, X_test, y_train, y_test = None, None, None, None
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Neural Network": MLPClassifier(max_iter=500)
    }
    try:
        if X_train is not None:
            results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    except Exception as e:
        print(f"Error during model training: {e}")
