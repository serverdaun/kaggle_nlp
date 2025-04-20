# Kaggle Disaster Tweets NLP

This project is focused on solving the Kaggle competition to classify tweets as disaster or non-disaster tweets using Natural Language Processing (NLP). The solution is built using Python and deep learning librarie PyTorch.

Link for Kaggle competition: https://www.kaggle.com/competitions/nlp-getting-started

## Project Structure

- **`kaggle_nlp/main.py`**: 
  The main script for model training and prediction. It includes:
  - Data loading and preprocessing
  - Vocabulary building
  - Model training and evaluation (BiLSTM)
  - Saving/loading model and vocabulary
  - Command-line interface for both training and prediction

- **`kaggle_nlp/utils/utilities.py`**: 
  Utility functions for data preprocessing, such as:
  - Cleaning and formatting keywords
  - Combining keyword and text fields
  - Removing unnecessary characters and links from tweets

- **`kaggle_nlp/predict_test.py`**: 
  (Legacy) Script for making predictions on the test dataset. The main workflow is now in `main.py`.

- **`kaggle_nlp/data/`**: 
  Contains `train.csv` and `test.csv` datasets.

- **`kaggle_nlp/model/`**: 
  Stores trained model weights `bilstm.pt` and vocabulary `vocab.json`.

- **`requirements.txt`**: 
  Python dependencies for the project.

## How to Use

1. **Install Dependencies**:
   Clone the repository and install dependencies (using pip or poetry):
   ```bash
   git clone https://github.com/serverdaun/kaggle_nlp
   cd kaggle_nlp
   pip install -r requirements.txt
   ```

2. **Download Data**:
   Download the competition data and place `train.csv` and `test.csv` in the `kaggle_nlp/data/` directory.

3. **Model Training**:
   Run the following command to train the model:
   ```bash
   python -m kaggle_nlp.main train --train_csv kaggle_nlp/data/train.csv --model_dir kaggle_nlp/model --epochs 10
   ```
   - Model weights and vocabulary will be saved in `kaggle_nlp/model/`.

4. **Predictions**:
   Run the following command to generate predictions on the test set:
   ```bash
   python -m kaggle_nlp.main predict --test_csv kaggle_nlp/data/test.csv --model_dir kaggle_nlp/model
   ```
   - The predictions will be saved as `predictions.csv` in the project root.


## Acknowledgments

This project is inspired by Kaggle’s Disaster Tweets competition. It leverages PyTorch for model implementation. Special thanks to the open-source community for providing tools that enable seamless model training and evaluation.
