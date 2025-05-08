import pandas as pd

def load_and_preprocess_data(file_path, text_column='message', label_column='sentiment'):
    """
    Loads data from a CSV file and performs basic preprocessing.
    This is a placeholder and might need significant adjustments based on your
    actual data format and preprocessing needs for Naive Bayes.

    Args:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing the text data.
        label_column (str): Name of the column containing the labels.

    Returns:
        tuple: A tuple containing two lists: texts and labels.
               Returns (None, None) if file not found or columns are missing.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None

    if text_column not in df.columns or label_column not in df.columns:
        print(f"Error: Required columns ('{text_column}', '{label_column}') not in {file_path}")
        return None, None

    texts = df[text_column].astype(str).str.lower().tolist()
    labels = df[label_column].tolist()

    print(f"Loaded {len(texts)} texts and {len(labels)} labels from {file_path}")
    # print(f"First 5 texts: {texts[:5]}")
    # print(f"First 5 labels: {labels[:5]}")

    return texts, labels

if __name__ == '__main__':
    data_path = 'data/train_data.csv' 
    
    if pd.io.common.file_exists(data_path):
        texts, labels = load_and_preprocess_data(data_path, text_column='message', label_column='sentiment')
        if texts and labels:
            print(f"Successfully loaded {len(texts)} items.")
    else:
        print(f"Test data file not found at {data_path}.")

