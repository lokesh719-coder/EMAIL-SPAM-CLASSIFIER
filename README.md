# Email Spam Classifier

A Python and Streamlit application that classifies a text message as **SPAM** or **HAM**. It uses NLTK to normalize text, TF-IDF for feature extraction, and compares three supervised scikit-learn classifiers on a stratified held-out test set.

## Features

- Loads the UCI SMS Spam Collection, downloading it on the first training run when it is not present locally.
- Handles missing rows, invalid labels, empty messages, and duplicate messages.
- Uses the same lowercase, punctuation removal, tokenization, and English stopword removal for training, evaluation, and prediction.
- Compares Multinomial Naive Bayes, Logistic Regression, and a linear-kernel Support Vector Machine.
- Reports accuracy, spam precision, spam recall, and spam F1-score, and saves confusion matrices.
- Selects the model using measured spam F1-score (then recall, precision, and accuracy as tie-breakers).
- Saves the classifier, TF-IDF vectorizer, model comparison table, and feature visualizations.
- Provides a Streamlit interface with a confidence score when supported by the classifier.

## Technologies

Python, pandas, NumPy, scikit-learn, NLTK, Matplotlib, joblib, and Streamlit.

## Dataset

The project uses the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), a tab-separated collection of messages labeled `ham` or `spam`. Training downloads the official archive into `data/` if `data/SMSSpamCollection` is not already present. An existing local dataset can be passed to the training functions or placed at that path.

## Project Structure

```text
.
├── app.py
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
├── visualizations/
├── README.md
└── requirements.txt
```

Training creates `models/spam_classifier.pkl` and `models/tfidf_vectorizer.pkl`. It also creates `visualizations/model_comparison.csv`, `visualizations/confusion_matrices.png`, and `visualizations/important_features.png`.

## Installation

Run these commands from the project root in PowerShell or the VS Code terminal. Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current terminal, or invoke `.venv\Scripts\python.exe` directly.

## Train the Model

```powershell
python -m src.train_model
```

The script downloads the dataset if needed, cleans and splits the messages, fits TF-IDF on the training split, measures all three models on the test split, prints the comparison, and saves the selected model and visualizations. The default split is stratified, 80% training and 20% testing, with a fixed random seed for repeatable comparison.

To evaluate the saved artifacts again on the same held-out split:

```powershell
python -m src.evaluate_model
```

## Run the Streamlit Application

Train the model first, then run:

```powershell
python -m streamlit run app.py
```

The app loads the persisted classifier and vectorizer; it does not retrain when started. Enter a message and select **Classify message**. Select **Clear** to reset the input and result.

## Example

Input:

```text
Congratulations! You have won a free prize. Call now to claim your reward!
```

Possible output:

```text
Prediction: SPAM
Model confidence: 98.7%
```

Predictions and confidence depend on the model trained from the dataset; the example score is illustrative, not a guaranteed result.

## Future Enhancements

- Add an email-focused labeled corpus and evaluate performance beyond SMS-length messages.
- Track precision-recall tradeoffs and allow a configurable spam decision threshold.
- Add cross-validation and experiment tracking for more robust model comparisons.
- Support batch classification and export of results.
- Add multilingual preprocessing and user feedback for model improvement.
