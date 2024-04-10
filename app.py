import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import sklearn

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_message(message):
    
    # converting to lowercase
    message = message.lower()
    
    # tokenization using nltk lib
    message = nltk.word_tokenize(message)
    # message is a list now
    
    # removing special characters
    text = []
    for i in message:
        if i.isalnum(): # keeping only alpha-numeric
            text.append(i)
    
    # removing stop words and punctuation marks
    message = text[:]
    text.clear() # clearing the text list
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            text.append(i)
    
    # stemming
    message = text[:]
    text.clear()
    for i in message:
        text.append(ps.stem(i))
    
    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Classifier")

input_email = st.text_input("Enter the email")

if st.button("Predict"):
  # 1. pre-process
  transformed_email = transform_message(input_email)
  # 2. vectorize
  vector_input = tfidf.transform([transformed_email])
  # 3. predict
  result = model.predict(vector_input)[0]
  # 4. display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Ham")