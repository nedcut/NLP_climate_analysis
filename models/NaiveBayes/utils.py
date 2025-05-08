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

    # Basic preprocessing (example: lowercase)
    # You'll likely want to add more:
    # - Removing punctuation
    # - Removing stop words
    # - Stemming or Lemmatization
    # These steps are crucial for good Naive Bayes performance.
    
    texts = df[text_column].astype(str).str.lower().tolist() # Ensure text is string and lowercase
    
    # Convert labels to numerical format if they aren't already
    # Example: {'anti': 0, 'neutral': 1, 'pro': 2}
    # This mapping should be consistent across your project.
    # For now, assuming labels are already numerical or can be directly used.
    # If your labels are categorical (e.g., 'anti', 'neutral', 'pro'),
    # you'll need to map them to integers.
    # Example mapping:
    label_mapping = {'anti': 0, 'neutral': 1, 'pro': 2, -1:0, 0:1, 1:2, 2:2} # Handling potential variations
    
    # Check if labels need mapping
    if df[label_column].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[label_column]):
        print(f"Attempting to map labels from column '{label_column}' using: {label_mapping}")
        labels = df[label_column].map(label_mapping).tolist()
        # Check for NaNs after mapping, which indicates unmapped original labels
        if pd.Series(labels).isnull().any():
            print(f"Warning: Some labels in '{label_column}' could not be mapped and resulted in NaN.")
            # Option: fillna or raise error
            # For now, let's see if the user has a preference or if it runs.
            # Consider how to handle unmapped labels (e.g., skip, default value, error)
    else: # Assuming labels are already numeric and correctly represent classes 0, 1, 2
        labels = df[label_column].tolist()


    print(f"Loaded {len(texts)} texts and {len(labels)} labels from {file_path}")
    # print(f"First 5 texts: {texts[:5]}")
    # print(f"First 5 labels: {labels[:5]}")


    return texts, labels

if __name__ == '__main__':
    # Example usage of the utility function
    # This assumes you have a 'train_data.csv' in the 'data' directory
    # relative to the 'NLP_climate_analysis' root.
    
    # Adjust the path to be relative to this utils.py file for standalone testing
    example_data_path = '../../data/train_data.csv' 
    
    if pd.io.common.file_exists(example_data_path):
        texts, labels = load_and_preprocess_data(example_data_path, text_column='message', label_column='sentiment')
        if texts and labels:
            print(f"Successfully loaded {len(texts)} items.")
            # print("Sample texts:", texts[:2])
            # print("Sample labels:", labels[:2])
    else:
        print(f"Test data file not found at {example_data_path}, skipping example usage.")

