#Stock Price Predict Web App

#Step1 Import the relavant libraries
import streamlit as st
import cufflinks as cf
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

#Step2: Create a widget with select box and slider using streamlit
START = '2017-07-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Stock Prediction Web App')
stocks = ['GOOGL','MSFT','JPM','GS','DAL','UAL','PFE','MRNA']
selected_stock = st.selectbox('Select a ticker for Prediction',stocks)
n_years = st.slider('Years of Prediction',1,4)
period = n_years*365

#Step3 Load Stock Data
@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data...')
data = load_data(selected_stock)
data_load_state.text('Loading data...done')

#Step4 Raw Data Visualization
st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data

#Step5 Forecasting with prophet library
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date':'ds','Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('forecast data')
st.write(forecast.tail())

#Step6 Plotting Forecast data and forecast components
st.write('Forecast Data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
