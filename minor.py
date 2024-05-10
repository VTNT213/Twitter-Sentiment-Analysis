import pandas as pd
import streamlit as st
from langdetect import detect
from textblob import TextBlob
import cleantext
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up stopwords for cleaning English text
stop_words_en = set(stopwords.words('english'))

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

st.header('Sentiment Analysis')

# Define sentiment categories and corresponding thresholds
sentiment_categories = {
    'Very Negative': (-1, -0.5),
    'Negative': (-0.5, 0),
    'Neutral': (0, 0.5),
    'Positive': (0.5, 1)
}

# Text input for single text analysis
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    
    if text:
        # Display synonyms for each word in the input text
        st.write("Synonyms:")
        words = text.split()
        for word in words:
            synonyms = get_synonyms(word)
            st.write(f"{word}: {synonyms}")

        # English text analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        for category, (lower, upper) in sentiment_categories.items():
            if lower <= polarity < upper:
                st.write('Sentiment: ', category)
                break
        st.write('Polarity: ', round(polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write(cleaned_text)

# File upload for CSV analysis
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def read_file(upl):
        # Function to read the uploaded file based on its extension
        if upl is not None:
            file_extension = upl.name.split('.')[-1].lower()
            if file_extension == 'csv':
                return pd.read_csv(upl)
            elif file_extension in ['xls', 'xlsx']:
                return pd.read_excel(upl, engine='openpyxl')
        return None

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        for category, (lower, upper) in sentiment_categories.items():
            if lower <= x < upper:
                return category

    if upl:
        df = read_file(upl)
        if df is not None:
            if 'tweets' in df.columns:
                df['score'] = df['tweets'].apply(score)
                df['analysis'] = df['score'].apply(analyze)
                st.write(df.head(10))

                # Visualize sentiment distribution
                st.subheader('Sentiment Distribution')
                fig1, ax1 = plt.subplots()
                sns.countplot(x='analysis', data=df, ax=ax1)
                st.pyplot(fig1)

                # Create and display word cloud
                st.subheader('Word Cloud')
                wordcloud = WordCloud(stopwords=stop_words_en, background_color='white').generate(' '.join(df['tweets']))
                fig2, ax2 = plt.subplots()
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis('off')
                st.pyplot(fig2)

                @st.cache_data
                def convert_df(df):
                    # Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='sentiment.csv',
                    mime='text/csv',
                )
            else:
                st.write("Error: 'tweets' column not found in the uploaded file.")
        else:
            st.write("Error: Unable to read the uploaded file. Please upload a valid CSV or Excel file.")