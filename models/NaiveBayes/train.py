import pandas as pd
from sklearn.model_selection import train_test_split
from models.NaiveBayes.model import NaiveBayesModel
import joblib
import os
from models.NaiveBayes.utils import load_and_preprocess_data
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path, model_save_path, vectorizer_save_path):
    """
    Trains the Naive Bayes model and saves it.

    Args:
        data_path (str): Path to the training data CSV file.
        model_save_path (str): Path to save the trained model.
        vectorizer_save_path (str): Path to save the vectorizer.
    """
    # Load and preprocess data
    texts, labels = load_and_preprocess_data(data_path, text_column='message', label_column='sentiment')

    # Initialize and train the model
    nb_model = NaiveBayesModel()
    print("Training Naive Bayes model...")
    nb_model.train(texts, labels)
    print("Training complete.")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_save_path), exist_ok=True)

    # Save the trained model and vectorizer
    joblib.dump(nb_model.model, model_save_path)
    print(f"Model saved to {model_save_path}")
    joblib.dump(nb_model.vectorizer, vectorizer_save_path)
    print(f"Vectorizer saved to {vectorizer_save_path}")

def evaluate_model(data_path, model_load_path, vectorizer_load_path, dataset_name="test"):
    """
    Loads a trained Naive Bayes model and vectorizer, evaluates it on the given dataset.

    Args:
        data_path (str): Path to the evaluation data CSV file.
        model_load_path (str): Path to load the trained model from.
        vectorizer_load_path (str): Path to load the vectorizer from.
        dataset_name (str): Name of the dataset (e.g., "dev", "test") for printing.
    """
    print(f"\nEvaluating model on {dataset_name} data: {data_path}")
    texts, true_labels = load_and_preprocess_data(data_path, text_column='message', label_column='sentiment')

    if texts is None or true_labels is None:
        print(f"Could not load data from {data_path}. Skipping evaluation for {dataset_name} set.")
        return

    # Load the model and vectorizer
    try:
        loaded_model = joblib.load(model_load_path)
        loaded_vectorizer = joblib.load(vectorizer_load_path)
    except FileNotFoundError:
        print(f"Error: Model or vectorizer not found at {model_load_path} or {vectorizer_load_path}")
        print("Please train the model first.")
        return

    # Create a NaiveBayesModel instance and assign the loaded components
    nb_model_eval = NaiveBayesModel()
    nb_model_eval.model = loaded_model
    nb_model_eval.vectorizer = loaded_vectorizer

    # Make predictions
    predictions = nb_model_eval.predict(texts)
    
    # Filter out None values from true_labels and corresponding predictions if any label mapping failed
    valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
    if len(valid_indices) < len(true_labels):
        print(f"Warning: {len(true_labels) - len(valid_indices)} samples were removed due to missing labels after mapping.")
    
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_predictions = predictions[valid_indices]

    if not filtered_true_labels:
        print("No valid labels found for evaluation after filtering. Skipping metrics.")
        return

    # Calculate and print metrics
    accuracy = accuracy_score(filtered_true_labels, filtered_predictions)
    report = classification_report(filtered_true_labels, filtered_predictions, zero_division=0, target_names=['anti', 'neutral', 'pro'])

    print(f"\nResults for {dataset_name} set:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save results to CSV
    output_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
    mapped_true_labels = [output_map.get(label, label) for label in filtered_true_labels]
    mapped_predicted_labels = [output_map.get(label, label) for label in filtered_predictions]

    results_df = pd.DataFrame({
        'text': [texts[i] for i in valid_indices], # Save original texts corresponding to valid labels
        'true_label': mapped_true_labels,
        'predicted_label': mapped_predicted_labels
    })
    results_save_path = os.path.join("results", "NaiveBayes", f"naive_bayes_{dataset_name}_results.csv")
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    results_df.to_csv(results_save_path, index=False)
    print(f"{dataset_name} set results saved to {results_save_path}")

if __name__ == '__main__':
    train_path = "data/train_data.csv"
    dev_path = "data/dev_data.csv"
    test_path = "data/test_data.csv"
    
    saved_model_dir = "saved_models/NaiveBayes"
    model_save_path = os.path.join(saved_model_dir, "naive_bayes_model.joblib")
    vector_save_path = os.path.join(saved_model_dir, "vectorizer.joblib")

    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please ensure 'data/train_data.csv' exists or update TRAIN_DATA_PATH.")
    else:
        train_model(train_path, model_save_path, vector_save_path)
        
        # Evaluate on Dev Set
        if os.path.exists(dev_path):
            evaluate_model(dev_path, model_save_path, vector_save_path, dataset_name="dev")
        else:
            print(f"\nWarning: Dev data not found at {dev_path}. Skipping dev set evaluation.")

        # Evaluate on Test Set
        if os.path.exists(test_path):
            evaluate_model(test_path, model_save_path, vector_save_path, dataset_name="test")
        else:
            print(f"\nWarning: Test data not found at {test_path}. Skipping test set evaluation.")
