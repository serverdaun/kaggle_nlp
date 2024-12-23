# Kaggle Disaster Tweets NLP

This project is focused on solving the Kaggle competition to classify tweets as disaster or non-disaster tweets using Natural Language Processing (NLP). The solution is built using Python and deep learning libraries such as PyTorch and Hugging Face Transformers.

Link for Kaggle competition: https://www.kaggle.com/competitions/nlp-getting-started

## Project Structure

- **`main.ipynb`**: 
  The Jupyter Notebook containing the main workflow for the project. It includes:
  - Data loading and preprocessing
  - Model training and evaluation
  - Visualizations and insights
  - Implementation of deep learning techniques

- **`predict_test.py`**: 
  A script to make predictions on the test dataset using the trained model. It includes:
  - Loading the trained model and tokenizer
  - Preprocessing the test data
  - Predicting disaster or non-disaster for each tweet

- **`utilities.py`**: 
  A utility script for data preprocessing. It provides helper functions such as:
  - Cleaning and formatting keywords
  - Combining keyword and text fields
  - Removing unnecessary characters and links from tweets


- **`pyproject.toml`** & **`poetry.lock`**:
  Configuration files for Poetry, which manage dependencies and ensure reproducibility.

## How to Use

1. **Install Poetry**:
   Poetry is required for dependency management. Install it using:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -

2. **Install Dependencies**:
    Clone repository and navigate to the project directory. Then install dependencies and activate virtual environment.
    ```bash
    git clone https://github.com/serverdaun/kaggle_nlp
    cd kaggle_nlp
   poetry install
   poetry shell

3. **Download Data**:
    Data can be downloaded using Kaggle API or directly from the competition page.
    ```bash
    mkdir data
   cd data
   kaggle competitions download -c nlp-getting-started

4. **Model Training**
    - Open `main.ipynb` and run the cells sequentially to train the model and save it

5. **Predictions**
   - Place your test data in `data/test.csv`.
   - Run the `predict_test.py` script:
     ```bash
     python predict_test.py
     ```
   - The predictions will be saved as a CSV file.

## Features

- **Deep Learning**:
  Utilizes Hugging Face Transformers for building and fine-tuning a sequence classification model.

- **Data Preprocessing**:
  - Handles missing data in the `keyword` column.
  - Cleans and prepares text data by removing links and special characters.

- **Modularity**:
  - Preprocessing utilities are separated for reusability.
  - The training and prediction pipelines are easy to modify and extend.

## Acknowledgments

This project is inspired by Kaggle’s Disaster Tweets competition. It leverages Hugging Face Transformers for state-of-the-art NLP capabilities, alongside PyTorch for model implementation. Special thanks to the Hugging Face team for providing open-source tools that enable seamless model training and fine-tuning.

## Future Improvements

- Fine-tuning hyperparameters for better performance
- Adding more robust preprocessing techniques
- Exploring alternative model architectures

---