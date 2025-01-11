import os
import sys
import pandas as pd
from src.data_preprocessing import load_dataset, preprocess_data, split_train_test
from src.model_training import train_and_evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def save_results_to_file(results, file_path):
    """
    Save model evaluation results to a text file.
    """
    try:
        with open(file_path, 'w') as file:
            file.write("Model Evaluation Results:\n")
            for model_name, accuracy in results.items():
                file.write(f"{model_name}: {accuracy:.2f}\n")
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def save_results_to_csv(results, file_path):
    """
    Save model evaluation results to a CSV file.
    """
    try:
        df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def main():
    # Configuration
    DATA_PATH = "data/bank-full.csv"
    TARGET_COLUMN = "y"
    RESULTS_FOLDER = "results"
    
    # Create results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    TEXT_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "model_evaluation_results.txt")
    CSV_RESULTS_PATH = os.path.join(RESULTS_FOLDER, "model_evaluation_results.csv")
    
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
        
        # Step 6: Save results to file
        save_results_to_file(results, TEXT_RESULTS_PATH)
        save_results_to_csv(results, CSV_RESULTS_PATH)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
