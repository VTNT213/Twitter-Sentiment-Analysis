import pandas as pd
import streamlit as st
import spacy
from spacy import displacy
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nlp = spacy.load("en_core_web_sm")
stop_words_en = set(stopwords.words('english'))

def get_sentiment(text):
    doc = nlp(text)
    polarity = doc.sentiment.polarity
    if polarity < -0.5:
        return 'Very Negative'
    elif polarity < 0:
        return 'Negative'
    elif polarity < 0.5:
        return 'Neutral'
    else:
        return 'Positive'

st.header('Sentiment Analysis')

# Text input for single text analysis
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    
    if text:
        # Display synonyms for each word in the input text
        doc = nlp(text)
        synonyms = {token.text: [token.text] + [lemma.text for lemma in token.lemma_] for token in doc}
        st.write("Synonyms:")
        for word, syns in synonyms.items():
            st.write(f"{word}: {', '.join(syns)}")

        # Sentiment analysis
        sentiment = get_sentiment(text)
        st.write('Sentiment: ', sentiment)
        st.write('Polarity: ', round(doc.sentiment.polarity, 2))
        st.write('Subjectivity: ', round(doc.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        doc = nlp(pre)
        cleaned_text = ' '.join(token.text for token in doc if not token.is_stop and not token.is_punct)
        st.write(cleaned_text)

# File upload for CSV analysis
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def read_file(upl):
        if upl is not None:
            file_extension = upl.name.split('.')[-1].lower()
            if file_extension == 'csv':
                return pd.read_csv(upl)
            elif file_extension in ['xls', 'xlsx']:
                return pd.read_excel(upl, engine='openpyxl')
        return None

    if upl:
        df = read_file(upl)
        if df is not None:
            if 'tweets' in df.columns:
                df['sentiment'] = df['tweets'].apply(get_sentiment)
                st.write(df.head(10))

                # Visualize sentiment distribution
                st.subheader('Sentiment Distribution')
                sns.countplot(x='sentiment', data=df)
                st.pyplot()

                # Create and display word cloud
                st.subheader('Word Cloud')
                text = ' '.join(df['tweets'])
                wordcloud = WordCloud(stopwords=stop_words_en, background_color='white').generate(text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot()

                @st.cache
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

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
