# Introduction to Machine Learning Final Project

## Usage (for Linux)
1. Create a virtual environment for packages with venv and activate it.
```bash
python -m venv .venv
source .venv/bin/activate
```
2. Install required packages.
```bash
pip install -r requirements.txt
```
3. Start the project.
```bash
python project.py
```

## Steps
1. Select a dataset from the UCI Machine Learning Repository or Kaggle.
2. Problem Definition: Explain the problem to be solved using the selected dataset.
3. Dataset Source: Indicate where the dataset was collected and which platform it was obtained from.
4. Number of Examples: Provide the total number of examples (rows) in the dataset and the number of classes (if applicable).
5. Data Preprocessing: Perform tasks such as missing data analysis, data cleaning, and formatting the data appropriately.
6. Training and Test Set Split: Divide the dataset into training and test sets (e.g., 80% training, 20% test).
7. Classification: Apply at least three different classification algorithms (e.g., Decision Tree, k-Nearest Neighbors, Artificial Neural Networks).
8. Accuracy: Compare and analyze the accuracy of each classification method.

## Project Overview

### File Structure
```bash
project/
│
├── data/                       # For storing raw and processed datasets
│   └── bank_marketing.csv      # Add the dataset here
│
├── src/                        # For Python scripts
│   └── data_preprocessing.py   # For data preprocessing tasks
│
├── results/                    # For saving outputs like accuracy reports, graphs, etc.
│
├── project.py                  # Entry point for running the project
│
└── README.md                   # For project overview and instructions
```

### Dataset
> Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
