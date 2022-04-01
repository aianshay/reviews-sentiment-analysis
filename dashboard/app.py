from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np
import spacy
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def load_data():

    data = pd.read_csv('../data/reviews_final.csv')

    return data

data = load_data()
st.cache(persist = True)    


st.sidebar.title('Sentiment Analysis of Smartphone Reviews')
st.sidebar.subheader('by Aian Shay and Tassiane Lima')

page = st.sidebar.selectbox("Navigation", ["Main", "Models Results", "Word Clouds"]) 


if page == "Main":
    st.title('Sentiment Analysis of Smartphone Reviews')

    select = st.sidebar.selectbox('Visualization type', ['Line Chart','Pie Chart'], key=1)

    monthly_count = pd.read_csv('../data/monthly_count.csv')

    line_plot = px.line(data_frame=monthly_count, 
                        x='new_date', 
                        y='monthly_perc', 
                        color='label',
                        markers=True,
                        labels={'new_date' : 'Month',
                                'monthly_perc' : 'Percentage of Reviews (%)'},
                        color_discrete_sequence=['blue', 'green', 'red'],
                        title='Monthly Sentiment Share',
                        )

    st.plotly_chart(line_plot) 

elif page == 'Word Clouds':
    st.title('Sentiment analysis of Smartphone Reviews')
    
    select = st.sidebar.selectbox('Evaluation', ['Good','Bad'], key=1)

    monthly_count = pd.read_csv('../data/monthly_count.csv')
    
    # Create stopword set
    nlp = spacy.load('pt_core_news_sm')
    stopwords = set(nlp.Defaults.stop_words)
    stopwords.update(['celular', 'aparelho', 'produto', 'dia', 'xiaomi', 'veio', 'telefone'])
    
    if(select == 'Good'):
        text_good = " ".join(str(review) for review in data[data['label'] == 'Good']['review'])
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(text_good)

    else:
        text_bad = " ".join(str(review) for review in data[data['label'] == 'Bad']['review'])
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(text_bad)
    
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wc, interpolation = 'bilinear')
    plt.axis('off')
    st.pyplot(fig)


elif page == "Models Results":
    st.title("Models Results")