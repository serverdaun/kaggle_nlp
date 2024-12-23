import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from kaggle_nlp.utils.utilities import PreprocessingUtils
from torch.utils.data import DataLoader, Dataset

FILE_PATH = 'data/test.csv'
MODEL_DIR = 'model'

# Load trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create object for preprocessing utils
utils = PreprocessingUtils()

# Define torch dataset class
class DisasterTweetsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


# Create function for preprocessing dataset
def preprocess_test_data(file_path, tokenizer, max_length=128, bath_size=32):
    # Load dataset
    df = pd.read_csv(file_path)

    # Apply data preprocessing methods
    df = utils.clean_keywords(df)
    df['combined_text'] = df.apply(utils.combine_text, axis=1)
    df['combined_text'] = df['combined_text'].apply(utils.clean_text)

    # Create a pytorch dataset and dataloader objects
    test_dataset = DisasterTweetsDataset(df.combined_text.tolist(), tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=bath_size, shuffle=False)
    print('Dataset is loaded and preprocessed')

    return test_loader, df

# Create function for making predictions
def predict():
    test_loader, df = preprocess_test_data(FILE_PATH, tokenizer)

    predictions = []

    # Perform predictions in batches
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_predictions.tolist())

    # Add predictions to initial dataset
    df['target'] = predictions
    final_df = df[['id', 'target']]

    # Save new dataset
    final_df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')

if __name__ == '__main__':
    predict()
