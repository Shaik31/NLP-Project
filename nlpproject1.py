
import pandas as pd
import streamlit as st
import time
#from sklearn.feature_extraction.text import CountVectorizer
import pickle
#import cleantext
from afinn import Afinn
afinn = Afinn()

# Suppress warnings (if any)
import warnings
warnings.filterwarnings('ignore')

# Load your model and other required files
with open('log_mod_nlp.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

#with open('afinn_score.pkl', 'rb') as afinn_file:
    #afinn = pickle.load(afinn_file)

def get_sentiment_label_color(sentiment_score):
    if sentiment_score > 0:
        return 'Positive', 'green'
    elif sentiment_score == 0:
        return 'Neutral', 'blue'
    else:
        return 'Negative', 'red'

# Streamlit app
st.title('NLP using Sentiment Analysis')

text = st.text_input('Enter your Text here')
submit = st.button('Predict')

if submit:
    start = time.time()
    # Clean the input text using cleantext module
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True,
                                   stemming=True, stopwords=True, lowercase=True, numbers=True, punct=True)
    # Calculate sentiment score
    text_new = cv.transform([cleaned_text])
    prediction = model.predict(text_new)[0]
    sentiment_score = afinn.score(cleaned_text)
    
    end = time.time()

    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    st.write('Sentiment Score: ', sentiment_score)

    sentiment_label, sentiment_color = get_sentiment_label_color(sentiment_score)
    st.write('Sentiment:', f'<span style="color:{sentiment_color}">{sentiment_label}</span>', unsafe_allow_html=True)
'''
pre = st.text_input('Clean Text: ')
if pre:
    cleaned_pre = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                  stemming=True, stopwords=True, lowercase=True, numbers=True, punct=True)
    st.write(cleaned_pre)

'''
