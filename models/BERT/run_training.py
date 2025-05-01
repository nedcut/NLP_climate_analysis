import os
import sys

# Bypass transformers version check
os.environ['TRANSFORMERS_SKIP_DEPENDENCY_CHECK'] = '1'

# Import the training script and run it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NLP_climate_analysis.models.BERT.train import main

# Run the main function
if __name__ == "__main__":
    main()