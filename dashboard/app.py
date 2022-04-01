from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np
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
    st.title('Sentiment analysis of Smartphone Reviews')

    select = st.sidebar.selectbox('Visualization type', ['Line Chart','Pie Chart'], key=1)

    monthly_count = pd.read_csv('../data/monthly_count.csv')

    fig = sns.lineplot()

    #line_plot = px.line(data_frame=monthly_count, 
     #             x='new_date', 
      #            y='monthly_perc', 
       #           color='label',
        #          title='Montlhy Sentiment Share')

    #st.plotly_chart(line_plot) 

    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly_count, x='new_date', y='monthly_perc', hue='label')
    plt.xticks(rotation=90)
    st.pyplot(fig)

elif page == 'Word Clouds':
    months = st.sidebar.slider('Months to plot', 1, 12, (1, 12))
    years = st.sidebar.slider('Year to plot', 2020, 2021)


elif page == "Models Results":
    st.title("Models Results")

