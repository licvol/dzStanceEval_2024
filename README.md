# Arabic Stance Detection

This project aims to perform stance detection on Arabic text using machine learning techniques. Stance detection involves identifying the attitude or perspective expressed in a piece of text towards a particular topic.

## Overview

The python script `dzStance_StanceEval2024.py` provides a step-by-step guide to:

1. **Data Preparation:**
   - Reading the dataset from a local CSV file.
   - Normalizing Arabic text by standardizing characters and replacing emojis with their Arabic equivalents.

2. **Model Training:**
   - Using the XLM-RoBERTa model to encode Arabic text into numerical embeddings.
   - Training a logistic regression classifier on the encoded text embeddings.

3. **Model Evaluation:**
   - Evaluating the trained model's performance on a development set using classification metrics such as precision, recall, and F1-score.

4. **Blind Test Prediction:**
   - Making predictions on a blind test dataset using the trained model.
   - Saving the predictions to a CSV file for further analysis.

## File Description

- `dzStance_StanceEval2024.py`: Python script containing the entire code pipeline.
- `gold_labels.txt`: File containing gold labels (IDs, topic, text, uppercase stance) for the development set.
- `predictions.txt`: File containing predicted labels (IDs, topic as target, text, prediction) for the development set.
- `dzStanceBlindTestPred.csv`: CSV file containing predictions for the blind test dataset.

## Usage

To run this Python script, use the following command:

python dzStance_StanceEval2024.py

## Requirements

- Python 3
- Required libraries: `pandas`, `scikit-learn`, `sentence-transformers`

## Dataset

- The dataset used in this project is stored in a local CSV file and contains Arabic text along with corresponding stance labels.
- The blind test dataset is also stored in a local CSV file and is used for making predictions on unseen data.

## Credits

- This project was developed by dzStance Team in Stance Evaluation Shared Task.
- The notebook makes use of the `sentence-transformers` library for generating text embeddings.

