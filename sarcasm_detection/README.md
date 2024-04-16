# Sarcasm Detection

## Description

This project is designed to analyze text for sarcasm. It uses machine learning models to classify the input text and determine whether it contains sarcasm.

## Methodology

#### Data:
The dataset contains over 28K news headlines with labels indicating if the headline contains sarcasm or not. The dataset in json format was obtained from
<a href="https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection">here</a>.
There are no gaps in the data, the data is balanced, not littered.

The following steps are followed to train the model

#### Preprocessing:
- Duplicates removed
- Columns with links removed
- Outliers removed
- Stop words, special characters removed
- Text is tokenized, lemmatized

#### Training:
The ability to change the trained models is specified,
The list is fed to the input in the pipeline.py file (requires importing the corresponding libraries).
4 models were tested:
- Logistic regression (LR);
- Naive Bayes (NB);
- RandomForestClassifier;
- Support Vector Machine (SVM).

#### Result:
The best model is selected based on the calculation of the model's KPI by the parameters:
accuracy, recall, f1-score, precision, time normalized. The best KPI was found in the NB model.

### Usage:
```python
cd src
```

train model:
```python
python pipeline.py
```

run inference:
```python
python run_inference.py
```
