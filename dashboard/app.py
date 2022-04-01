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
    months = st.sidebar.slider('Months to plot', 1, 12, (1, 12))
    years = st.sidebar.slider('Year to plot', 2020, 2021)


elif page == "Models Results":
    st.title("Models Results")