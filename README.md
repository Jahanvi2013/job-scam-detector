
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

## Model Performance and Evaluation

Three machine learning algorithms were evaluated using a
held-out test dataset:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Linear SVM was selected based on its overall performance
across accuracy, precision, recall and F1 score.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Linear SVM | 98.46% | 85.12% | 82.66% | 83.87% |
| Logistic Regression | 97.71% | 70.59% | 90.17% | 79.19% |
| Naive Bayes | 96.03% | 66.67% | 35.84% | 46.62% |

### Understanding the Results

Although Logistic Regression achieved higher recall,
Linear SVM achieved a stronger balance between precision
and recall, resulting in the highest F1 score.

This was important because the dataset was imbalanced,
with significantly more legitimate job postings than
fraudulent ones.

Accuracy alone would therefore not provide a complete
picture of model performance.

### Confusion Matrix Analysis

For the Linear SVM model, the test-set confusion matrix
contained the following results:

| | Predicted Legitimate | Predicted Fraudulent |
|---|---:|---:|
| Actual Legitimate | 3,378 | 25 |
| Actual Fraudulent | 30 | 143 |

This means:

- 3,378 legitimate postings were correctly classified.
- 143 fraudulent postings were correctly identified.
- 25 legitimate postings were incorrectly flagged as fraudulent.
- 30 fraudulent postings were incorrectly classified as legitimate.

### Why False Positives and False Negatives Matter

False positives can cause genuine employment opportunities
to be incorrectly flagged, potentially discouraging job seekers
from applying.

False negatives are particularly concerning because fraudulent
job advertisements may be classified as legitimate, exposing
job seekers to potential scams.

The model therefore requires a balance between identifying
fraudulent postings and avoiding incorrect classifications.

## Model Limitations

Several limitations should be considered when interpreting
the results:

1. **Dataset imbalance:** The labelled dataset contained
   substantially more legitimate than fraudulent postings.

2. **Dataset representativeness:** Patterns learned from the
   available dataset may not represent all job scams or
   legitimate recruitment practices.

3. **Synthetic data:** An additional dataset containing
   10,000 fake job descriptions was explored separately.
   Its repetitive, template-generated content meant it was
   not used to train or evaluate the final model.

4. **False classifications:** The model can incorrectly
   classify both legitimate and fraudulent advertisements.

5. **Decision score interpretation:** Linear SVM produces
   a decision score rather than a calibrated probability.
   The score indicates which side of the classification
   boundary a posting falls on; it does not represent the
   percentage likelihood of fraud.

6. **Explainability:** Individual feature contributions
   reflect learned statistical associations, not definitive
   evidence that a particular word or phrase indicates fraud.

The application should therefore be treated as an initial
screening tool rather than a definitive fraud detection system.

## Exploratory Data Analysis and Visualisations

### Class Distribution

![Class Distribution](images/class%20distribution.png)

### Confusion Matrix

![Confusion Matrix](images/confusion%20matrix.png)

### Fake Job Word Cloud

![Fake Job Word Cloud](images/fake%20job%20wordcloud.png)


## Future Improvements

Potential areas for further development include:

- Expanding the labelled dataset with more recent job advertisements.
- Testing the model on data from different sources.
- Exploring class-weighting and resampling techniques.
- Evaluating alternative classification thresholds.
- Investigating probability calibration.
- Monitoring false positives and false negatives.
- Exploring additional NLP and explainability techniques.


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
