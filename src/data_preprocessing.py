import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_dataset(file_path):
    """
    Load the dataset from the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, sep=';')

def preprocess_data(data):
    """
    Preprocess the dataset: handle missing values, format data, and encode categorical features.
    """
    # Convert categorical columns to lowercase for consistency
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].str.lower()
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    one_hot_cols = []
    for col in categorical_cols:
        if col == "y":  # Target variable, apply label encoding
            data[col] = label_encoder.fit_transform(data[col])
        else:
            # Apply One-Hot Encoding for nominal features
            one_hot_encoded = pd.get_dummies(data[col], prefix=col)
            one_hot_cols.append(one_hot_encoded)
            data.drop(columns=[col], inplace=True)
    
    # Concatenate one-hot encoded columns
    if one_hot_cols:
        data = pd.concat([data] + one_hot_cols, axis=1)

    return data

def split_train_test(data, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# For testing this module standalone
if __name__ == "__main__":
    DATA_PATH = "data/bank-full.csv"
    TARGET_COLUMN = "y"

    try:
        dataset = load_dataset(DATA_PATH)
        preprocessed_data = preprocess_data(dataset)
        X_train, X_test, y_train, y_test = split_train_test(preprocessed_data, TARGET_COLUMN)
        print("Data preprocessing and splitting completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
