# Sentiment Analysis of Product Reviews

This project is part of the IE4483 Artificial Intelligence and Data Mining course. The goal is to classify product reviews as either positive or negative using sentiment analysis techniques.

## Project Overview

The project reads product reviews from JSON files, processes the text (removing common words, converting text to numeric features using TF-IDF), trains a Logistic Regression classifier, and predicts whether reviews in the test set are positive or negative. The results are saved to a CSV file.

Additionally, advanced models were explored, including:

- **BERT** (Bidirectional Encoder Representations from Transformers), a pre-trained language model.
- **LSTM** (Long Short-Term Memory), a recurrent neural network leveraging word embeddings.

It was found that **BERT achieved the highest accuracy** among the models tested. However, it required significantly more computational time and resources compared to LSTM and Logistic Regression due to its complexity and pretraining.

The best-performing model, implemented in `best_bert.py`, outputs results stored in the `results` folder.

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
- Results for all models, including predictions on the test set, are in the `results` folder.

## Project Guidelines Summary

This project follows the guidelines from the IE4483 Artificial Intelligence and Data Mining Mini Project:

1. **Data Loading**: Data loaded from `train.json` and `test.json`, formatted into reviews and sentiments.
2. **Feature Extraction**: Features generated using TF-IDF, pre-trained embeddings (e.g., GloVe), and BERT.
3. **Model Selection**: Explored Logistic Regression, LSTM, and BERT.
4. **Training**: Models trained with various parameters, including learning rates and optimizer configurations.
5. **Evaluation**: Predictions evaluated on test data and saved in `submission.csv`.
6. **Analysis**:
   - Correctly and incorrectly classified samples reviewed to identify model strengths and weaknesses.
   - Feature formats compared for resource consumption and accuracy.
7. **Adaptability**:
   - Hotel review classification discussed for scenarios with or without noisy rating scores.

## References

1. John Blitzer, Mark Dredze, Fernando Pereira. *Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification*. Association of Computational Linguistics (ACL), 2007.
2. Tomas Mikolov et al. *Efficient estimation of word representations in vector space*. arXiv preprint arXiv:1301.3781, 2013.
3. Jeffrey Pennington et al. *GloVe: Global vectors for word representation*. Proceedings of EMNLP, 2014.
4. Jacob Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.
5. [5 Things About Sentiment Analysis](https://www.kdnuggets.com/2018/03/5-things-sentiment-analysis-classification.html)

---

For further details, refer to the project report and accompanying code files.

