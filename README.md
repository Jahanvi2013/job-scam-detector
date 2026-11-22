
# Job Scam Detector

A machine learning application that analyses job advertisements
and identifies patterns associated with potentially fraudulent
job postings.

## 🚀 Live Demo

Try the application here:

**[Job Scam Detector – Live Application](https://jobscam.streamlit.app/)**

Paste a job advertisement to receive a machine learning
classification and explore the features influencing the prediction.

## Project Overview

Online job scams can expose job seekers to financial loss,
identity theft and misleading employment opportunities.

This project explores how Natural Language Processing (NLP)
and machine learning can be used to screen job advertisements
for potentially fraudulent patterns.

The project includes an interactive Streamlit application
that allows users to paste a job advertisement and receive
a classification from a trained machine learning model.

## Objectives

- Explore and preprocess job advertisement data.
- Apply NLP techniques to extract meaningful text features.
- Compare different machine learning classification algorithms.
- Evaluate model performance using multiple metrics.
- Build an interactive application to demonstrate the model.

## Methodology

### Data Preprocessing
- Combined relevant job advertisement text fields.
- Converted text to lowercase.
- Removed URLs, email addresses and non-alphabetic characters.
- Applied TF-IDF vectorisation.

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency)
was used to represent job advertisement text.

Both unigrams and bigrams were included.

### Models Evaluated

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

### Model Selection

Linear SVM was selected based on its overall
performance across the evaluation metrics.

## Model Performance

| Metric | Linear SVM |
|---|---:|
| Accuracy | 98.46% |
| Precision | 85.12% |
| Recall | 82.66% |
| F1 Score | 83.87% |

Results are based on the held-out test set from
the labelled dataset.

## Application

The Streamlit application allows users to:

- Paste a job advertisement.
- Receive a predicted classification.
- View the Linear SVM decision score.
- Explore text features contributing towards
  fraudulent or legitimate classifications.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing
- TF-IDF
- Linear SVM
- Streamlit
- Git and GitHub

## Running the Application Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages.
4. Run the Streamlit application.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Limitations

- Predictions depend on patterns learned from the training data.
- Dataset biases may affect classification results.
- The SVM decision score is not a probability.
- A legitimate job may be incorrectly classified as fraudulent,
  and vice versa.

This application is an educational screening tool and
does not establish whether an employer or job posting
is genuinely fraudulent.

## Future Improvements

- Expand the training dataset.
- Investigate model calibration.
- Improve model explainability.
- Explore additional NLP techniques.
- Deploy the application online.

## GitHub Repository

[View the source code](https://github.com/Jahanvi2013/job-scam-detector)

## Author

Jahanvi Tripathi
