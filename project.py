import sys
from src.data_preprocessing import load_dataset, preprocess_data, split_train_test
from src.model_training import train_and_evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def main():
    # Configuration
    DATA_PATH = "data/bank-full.csv"
    TARGET_COLUMN = "y"
    
    try:
        # Step 1: Load the dataset
        print("Loading dataset...")
        dataset = load_dataset(DATA_PATH)
        
        # Step 2: Preprocess the dataset
        print("Preprocessing dataset...")
        preprocessed_data = preprocess_data(dataset)
        
        # Step 3: Split into training and test sets
        print("Splitting dataset into training and test sets...")
        X_train, X_test, y_train, y_test = split_train_test(preprocessed_data, TARGET_COLUMN)
        
        # Step 4: Train and evaluate models
        print("Training and evaluating models...")
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Neural Network": MLPClassifier(max_iter=500)
        }
        results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
        
        # Step 5: Display results
        print("\nModel Evaluation Results:")
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.2f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
