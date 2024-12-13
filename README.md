Here's your updated project description formatted as a **README** file:

---

# Sentiment Analysis of Product Reviews

This project is part of the Artificial Intelligence and Data Mining course. The goal is to classify product reviews as either positive or negative using sentiment analysis techniques.

## Project Overview

The project reads product reviews from JSON files, processes the text (removing common words, converting text to numeric features using TF-IDF), trains a Logistic Regression classifier, and predicts whether reviews in the test set are positive or negative. The results are saved to a CSV file.

Additionally, advanced models were explored, including:

- **BERT** (Bidirectional Encoder Representations from Transformers), a pre-trained language model.
- **LSTM** (Long Short-Term Memory), a recurrent neural network leveraging word embeddings.

It was found that **BERT achieved the highest accuracy** among the models tested. However, it required significantly more computational time and resources compared to LSTM and Logistic Regression due to its complexity and pretraining.

The best-performing model, implemented in `best_bert.py`, outputs results stored in the `results` folder.

## Team Members

This project was a collaborative effort between the following team members:

- **Aaryan Kandiah**
- **Kheng Lin Hao**
- **Benjamin Lee Wen Long**

## Requirements

The project requires the following dependencies:

- Python 3.12+
- pandas
- scikit-learn
- nltk
- transformers
- torch

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Key Steps

### Data Preparation

- Data consists of two JSON files: `train.json` and `test.json`.
- Each review includes:
  - **reviews**: The user review in raw text.
  - **sentiments**: The ground truth sentiment (0 for negative, 1 for positive).
- Preprocessing steps included tokenization, stopword removal, and feature extraction using TF-IDF and embeddings.

### Model Development

- **Logistic Regression** was used as a baseline model with TF-IDF features.
- **LSTM** models were tested with word embeddings for sequential learning.
- **BERT** was fine-tuned for sentiment analysis, leveraging its bidirectional context modeling for superior performance.

### Results

- BERT achieved the best accuracy due to its ability to capture deep semantic relationships and bidirectional dependencies in text.
- It required substantial computational resources, including a GPU for training efficiency.
- Results for all models, including predictions on the test set, are in the `Results` folder.
