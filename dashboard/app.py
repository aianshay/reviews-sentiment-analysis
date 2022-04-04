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
st.set_page_config(page_title='Sentiment Analysis Dashboard')


st.sidebar.title('Sentiment Analysis of Smartphone Reviews')
st.sidebar.subheader('by Aian Shay and Tassiane Lima')

page = st.sidebar.selectbox("Navigation", ["Main", "Models Results", "Word Clouds"]) 


if page == "Main":
    st.title('Sentiment Analysis of Smartphone Reviews')

    st.markdown("""This dashboard consists in a a sentiment analysis project where we analyze reviews from a Smartphone. We built a crawler to scrap the reviews from Amazon's [website](https://www.amazon.com.br/Smartphone-Xiaomi-Redmi-Note-4RAM/dp/B07Y9XWK4M?ref_=Oct_d_omwf_d_16243890011&pd_rd_w=WpQXH&pf_rd_p=ea8c9e98-cbc0-4a8a-857e-90f7489e5fb1&pf_rd_r=Z6X2BCKWQDBV75NFSFQV&pd_rd_r=4190a482-4da3-4ad9-a21e-b36f66437680&pd_rd_wg=1bzDv&pd_rd_i=B07Y9XWK4M&th=1) and built a dataset with 4.770 reviews.""")

    select = st.sidebar.selectbox('Visualization type', ['Line Chart','Pie Chart'], key=1)

    monthly_count = pd.read_csv('../data/monthly_count.csv')

    line_plot = px.line(data_frame=monthly_count, 
                        x='new_date', 
                        y='monthly_perc', 
                        color='label',
                        markers=True,
                        labels={'new_date' : 'Month',
                                'monthly_perc' : 'Percentage of Reviews (%)'},
                        color_discrete_sequence=['blue', 'green', 'red'])
                        
    st.markdown("## Monthly sentiment share")
    st.plotly_chart(line_plot) 
    st.markdown("""To plot this graph we considered 1 and 2 stars reviews as bad, 3 stars as 
                    neutral and 4 and 5 stars as good.""")


    st.markdown('## Reviews Summary')
    st.markdown("""To build a summary of the reviews, we used the KMeans clustering algorithm. 
                   The idea was to find key words that can summarize Good and Bad reviews, this 
                   way a manager does not need to read every single review to get a glimpse of what is being said about the product.""")

    st.markdown("### Good Reviews")
    st.markdown(""" - celular 
                    - produto 
                    - aparelho 
                    - bateria 
                    - chegou 
                    - excelente 
                    - câmera 
                    - ótimo 
                    - qualidade 
                    - entrega""")

    st.markdown("""With these words, we can notice that positive reviews of the smartphone
                   often mentioned battery and camera as strong caracteristics of the product. """)


    st.markdown('### Bad Reviews')
    st.markdown(""" - nota 
                    - fiscal
                    - produto
                    - veio
                    - carregador
                    - padrão
                    - garantia
                    - brasileiro
                    - vendedor
                    - chegou""")

    st.markdown("""In bad reviews, what is often mentioned is the lack of invoice,
                   missing power adapter in the Brazilian standard and problems while 
                   using the warranty.""")


elif page == 'Word Clouds':
    st.title('Word Clouds')
        
    # Create stopword set
    nlp = spacy.load('pt_core_news_sm')
    stopwords = set(nlp.Defaults.stop_words)
    stopwords.update(['celular', 'aparelho', 'produto', 'dia', 'xiaomi', 'veio', 'telefone'])
    
    
    st.markdown("## Positive Reviews")

    text_good = " ".join(str(review) for review in data[data['label'] == 'Good']['review'])
    wc = WordCloud(stopwords=stopwords, background_color="white").generate(text_good)

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wc, interpolation = 'bilinear')
    plt.axis('off')
    st.pyplot(fig)

    st.markdown("## Negative Reviews")

    text_bad = " ".join(str(review) for review in data[data['label'] == 'Bad']['review'])
    wc = WordCloud(stopwords=stopwords, background_color="white").generate(text_bad)

    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wc, interpolation = 'bilinear')
    plt.axis('off')
    st.pyplot(fig)


elif page == "Models Results":    
    st.title("LSTM")
    df_lstm = pd.read_csv('../results/lstm.csv')
    st.dataframe(df_lstm.iloc[:,1:]) 
    
    count_pred = pd.read_csv('../data/pred_lstm_count.csv')

    line_plot = px.line(data_frame=count_pred, 
                        x='new_date', 
                        y='monthly_perc', 
                        color='label_pred',
                        markers=True,
                        labels={'new_date' : 'Month',
                                'monthly_perc' : 'Percentage of Reviews (%)'},
                        color_discrete_sequence=['blue', 'green', 'red'],
                        title='Label Prediction',
                        )

    st.plotly_chart(line_plot)

    count_label = pd.read_csv('../data/label_test_count.csv')

    line_plot = px.line(data_frame=count_label, 
                        x='new_date', 
                        y='monthly_perc', 
                        color='label',
                        markers=True,
                        labels={'new_date' : 'Month',
                                'monthly_perc' : 'Percentage of Reviews (%)'},
                        color_discrete_sequence=['blue', 'green', 'red'],
                        title='Real Label',
                        )

    st.plotly_chart(line_plot)

    st.title("Random Forest")
    df_rf = pd.read_csv('../results/random_forest.csv')
    st.dataframe(df_rf.iloc[:,1:])