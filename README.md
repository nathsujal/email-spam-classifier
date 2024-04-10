# High-Precision Spam Email Classifier

website: https://email-spam-classifier-r25n.onrender.com

This repository contains the code for a spam email classifier focused on achieving high precision and minimizing false positives.

## Project Overview

This project utilizes Python libraries to explore, clean (if necessary), preprocess, and analyze email data to build a machine-learning model capable of classifying emails as spam or ham (not spam).

### Steps Involved:

#### 1. Data Exploration & Feature Engineering (EDA):
  - Used NumPy and Pandas to explore and understand characteristics of email data obtained from a Kaggle dataset.  
  - Performed feature engineering by creating new features like character count to identify potential spam indicators.  

#### 2. Text Preprocessing:
  - Utilized the NLTK library for text preprocessing tasks:  
    - Lowercase all text for consistency.  
    - Performed tokenization (splitting text into words).  
    - Removed special characters and stop words (common words like "the" or "a").  
    - Applied stemming (reducing words to their base form) to improve model performance.  

#### 3. Data Visualization (Optional):
  - Used Seaborn to create a word cloud visualizing the most frequent words in both spam and ham emails. This helped identify potential keywords or patterns associated with spam.  

#### 4. Machine Learning Model Development:
  - Leveraged scikit-learn for machine learning tasks:
    - Employed TF-IDF vectorization to convert text data into numerical features suitable for machine learning algorithms.  
    - Experimented with various classification algorithms to find the best fit. Ultimately, the ExtraTreesClassifier was chosen due to its high precision in spam detection.  

#### 5. Model Persistence:
  - Used Pickle, a Python object persistence library, to save the trained ExtraTreesClassifier model and the TF-IDF vectorizer for future use (deployment). This allows for model reuse without retraining every time.


### Project Dependencies:
NumPy
Pandas
NLTK
scikit-learn
Pickle
Seaborn
Streamlit (for deployment)
Render (for website hosting)

### Getting Started:
1. Clone this repository:
  ```
  git clone https://github.com/<your-username>/<your-repo-name>.git
  ```
2. Install required dependencies:
  ```
  pip install -r requirements.txt
  ```
3. To run the app locally
  ```
  streamlit run app.py
  ```

#### Feel free to explore the code and experiment further!
