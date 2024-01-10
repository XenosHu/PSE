#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import feedparser
import string

raw = pd.read_csv("PSE_info.csv")
pse_tickers = raw['symbol'].tolist()

available_indexes = ["Philippines Stock Exchange"]

# Sidebar: Index Selection
index_selection = st.sidebar.selectbox("Select an Index", available_indexes)

# Correctly define index_mapping as a dictionary
index_mapping = {
    "Philippines Stock Exchange": "PSEi"
}

index_symbol = index_mapping.get(index_selection)


@st.cache_data
def get_stock_list(index_selection):
    
    if index_selection == "Philippines Stock Exchange":
        stock_list = pse_tickers
    else:
        stock_list = []
    return stock_list

ticker_list = get_stock_list(index_selection)


st.markdown("<h1 style='text-align: center;'>Stock Market Analysis Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def get_stock_data_pse(keyword,start_date,end_date):
    headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    'Content-Type': 'application/json',  # This is typically set automatically when using json parameter
    'Origin':'https://www.investagrams.com',
    'Referer':'https://www.investagrams.com/'
    }
    url_search = "https://webapi.investagrams.com/InvestaApi/Stock/SearchStockSnippet"
    params_search = {'keyword': keyword, 'userDefaultExchangeType': '4', 'selectedExchangeType': '0', 'limit': '0', 'cv': '1704729600-0-v3'}
    response_search = requests.get(url=url_search,headers=headers,params=params_search)
    stock_id = json.loads(response_search.text)[0]['StockId']
    
    url_getstock = "https://webapi.investagrams.com/InvestaApi/Stock/GetStockHistoricalTableByStockIdAndDate"
    params_getstock = {
    'stockId': stock_id,
    'timeRange': '12M',
    'irt': 'a'}
    response_getstock = requests.get(url=url_getstock,headers=headers,params=params_getstock)
    df = pd.DataFrame(json.loads(response_getstock.text))
    df['D'] = pd.to_datetime(df['D'], utc=True)

    # Convert the constraints to datetime in UTC
    start_date = pd.to_datetime(start_date, utc=True)
    end_date = pd.to_datetime(end_date, utc=True)

    df['D'] = pd.to_datetime(df['D'])
    # Filter the DataFrame based on the date constraints
    filtered_df = df[(df['D'] >= start_date) & (df['D'] <= end_date)]
    # Display the filtered data
    return filtered_df


@st.cache_data
def get_index_data(index_symbol, timeframe):
    
    # Define end date as today
    end_date = datetime.now()
        
    # Calculate start date based on the selected timeframe
    if timeframe == '5 Day':
        start_date = end_date - timedelta(days=5)
    elif timeframe == '1 Week':
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == '1 Month':
        start_date = end_date - timedelta(weeks=4)
    elif timeframe == '6 Months':
        start_date = end_date - timedelta(weeks=26)
    elif timeframe == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif timeframe == '1 Year':
        start_date = end_date - timedelta(weeks=52)
    elif timeframe == '5 Year':
        start_date = end_date - timedelta(weeks=260)
    
    index_data = yf.Ticker(index_symbol).history(start=start_date, end=end_date)

    return index_data#


datetime.today()
st.sidebar.title("Select Parameters")

selected_stock = st.sidebar.selectbox("Select a stock symbol", ticker_list)
default_start_date = datetime.today() - timedelta(weeks=52)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date")
selected_stock_name = raw[raw['symbol'] == 'WLCON']['company_name'].iloc[0]

if start_date >= end_date:
    st.error("Error: Start date must be before end date.")
    
# Fetch data and display price chart
if index_selection == "Philippines Stock Exchange":
    stock_data = get_stock_data_pse(selected_stock, start_date, end_date)
#other_data = get_currency(selected_stock)

#-------------------------------------------------------------------------------------------------------------------------------
# with st.expander(f"**({selected_stock}) Stock Price**"):
#     st.write(stock_data)

if stock_data.empty:
    st.write("Data not available for this stock symbol in the specified date range.")
else:
    fig = go.Figure(data=go.Candlestick(x=stock_data['D'],
                                       open=stock_data['O'],
                                       high=stock_data['H'],
                                       low=stock_data['L'],
                                       close=stock_data['C']))

    fig.update_layout(yaxis_title=f'Price ₱',
                      xaxis_title='Date')

    st.plotly_chart(fig)

st.subheader(f"{selected_stock} Stock Price Data")
if not stock_data.empty:
    # Calculate 1-Year Change
    one_year_change = ((stock_data["C"].iloc[-1] / stock_data["C"].iloc[0]) - 1) * 100
    
    average_vol_3m = stock_data["V"].tail(63).mean()

    prev_close = stock_data["C"].iloc[-2] if len(stock_data["C"]) > 1 else None
    open_price = stock_data["O"].iloc[-1] if len(stock_data["O"]) > 0 else None
    volume = stock_data["V"].iloc[-1] if len(stock_data["V"]) > 0 else None
    day_range = f"{stock_data['L'].iloc[-1]:,.2f}-{stock_data['H'].iloc[-1]:,.2f}" if len(stock_data["L"]) > 0 else "N/A"
    fifty_two_week_range = f"{stock_data['L'].min():,.2f}-{stock_data['H'].max():,.2f}" if not stock_data['L'].empty else "N/A"

    
    updated_data = stock_data
    updated_data["% Change"] = stock_data["C"] / stock_data["C"].shift(1) - 1
    st.write(updated_data)
    
    annual_return = updated_data["% Change"].mean()*252*100
    # annual_return_color = "green" if annual_return >= 0 else "red"
    # st.markdown(f"Annual Return: <span style='color:{annual_return_color}'>{round(annual_return, 2)}%</span>", unsafe_allow_html=True)
    
    stdev = np.std(updated_data["% Change"]) * np.sqrt(252)
    # stdev_color = "green" if stdev >= 0 else "red"
    # st.markdown(f"Standard Deviation is: <span style='color:{stdev_color}'>{round(stdev * 100, 2)}%</span>", unsafe_allow_html=True)
    
    stock_summary_data = {
        "Prev. Close": [f"{prev_close:,.2f}" if prev_close is not None else "N/A"],
        "Open": [f"{open_price:,.2f}" if open_price is not None else "N/A"],
        "1-Year Change": [f"{one_year_change:.2f}%" if one_year_change is not None else "N/A"],
        "Volume": [f"{volume:,.0f}" if volume is not None else "N/A"],
        "Average Vol. (3m)": [f"{average_vol_3m:,.0f}" if average_vol_3m is not None else "N/A"],
        "Day's Range": [day_range],
        "52 wk Range": [fifty_two_week_range],
        "Annual Return": [annual_return],
        "Standard Deviation": [stdev]
    }
    # Convert the dictionary to a DataFrame
    df_stock_summary = pd.DataFrame.from_dict(stock_summary_data)
    df_transposed = df_stock_summary.T

    st.subheader(f"{selected_stock} Stock Summary")
    # Display in Streamlit
    st.table(df_transposed)
else:
    st.write("Stock data is not available. Please select a valid stock.")

#-------------------------------------------------------------------------------------------------------------------------------

# @st.cache_data
# def get_fundamental_metrics(stock_symbol):
#     stock = yf.Ticker(stock_symbol)
#     info = stock.info

#     # Map API response fields to desired metrics
#     fundamental_metrics = {
#         "Market Cap": info.get("marketCap"),
#         "Forward P/E": info.get("forwardPE"),
#         "Trailing P/E": info.get("trailingPE"),
#         "Dividend Yield": info.get("dividendYield") * 100 if info.get("dividendYield") else None,
#         "Earnings Per Share (EPS)": info.get("trailingEps"),
#         "Beta": info.get("beta")
#     }

#     return fundamental_metrics
    
# with st.expander("Definitions of Fundamental Data"):
#     st.write("Market Cap: Market capitalization is the total value of a company's outstanding shares of stock. It is calculated by multiplying the stock's current market price by its total number of outstanding shares.")
#     st.write("Forward P/E: Forward price-to-earnings (P/E) ratio is a valuation ratio that measures a company's current share price relative to its estimated earnings per share for the next year.")
#     st.write("Trailing P/E: Trailing price-to-earnings (P/E) ratio is a valuation ratio that measures a company's current share price relative to its earnings per share over the past 12 months.")
#     st.write("Dividend Yield: Dividend yield is a financial ratio that indicates how much a company pays out in dividends each year relative to its share price. It is usually expressed as a percentage.")
#     st.write("Earnings Per Share (EPS): Earnings per share is a measure of a company's profitability. It represents the portion of a company's profit allocated to each outstanding share of common stock.")
#     st.write("Beta: Beta measures a stock's volatility in relation to the overall market. A beta greater than 1 indicates the stock is more volatile than the market, while a beta less than 1 indicates lower volatility.")

# st.subheader(f"Fundamental Data for {selected_stock}")

# fundamental_metrics = get_fundamental_metrics(selected_stock)

# for metric, value in fundamental_metrics.items():
#     st.write(f"{metric}: {value}")   
    
#-------------------------------------------------------------------------------------------------------------------------------

def plot_sma_vs_closing_price(stock_symbol, start_date, end_date):

    stock_data = get_stock_data_pse(stock_symbol, start_date, end_date)
    
    # Calculate Simple Moving Average (SMA)
    sma_period = 20
    stock_data['SMA'] = stock_data['C'].rolling(window=sma_period).mean()
    
   # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['D'], y=stock_data['C'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=stock_data['D'], y=stock_data['SMA'], mode='lines', name=f'SMA {sma_period}'))
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title= f'Price ₱',
        title=f'{stock_symbol} Closing Price vs. SMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

def plot_ema_vs_closing_price(stock_symbol, start_date, end_date):

    stock_data = get_stock_data_pse(stock_symbol, start_date, end_date)
    
    ema_period = 20
    
    if stock_data.empty or 'C' not in stock_data.columns:
        st.write("No data available or missing 'C' column.")
        return
            
    # Calculate Exponential Moving Average (EMA)
    stock_data['EMA'] = stock_data['C'].ewm(span=ema_period).mean()
    
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['D'], y=stock_data['C'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=stock_data['D'], y=stock_data['EMA'], mode='lines', name=f'EMA {ema_period}'))
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title= f'Price ₱',
        title=f'{stock_symbol} Closing Price vs. EMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

st.subheader("Trend Analysis Using Indicators")
indicator_type = st.selectbox("**Select Indicator Type**", ["sma", "ema"])

if indicator_type == "sma":
    sma_plot = plot_sma_vs_closing_price(selected_stock, start_date, end_date)
    
elif indicator_type == 'ema':
    ema_plot = plot_ema_vs_closing_price(selected_stock, start_date, end_date)       

#-------------------------------------------------------------------------------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def get_news(selected_stock_name):
    translator = str.maketrans('', '', string.punctuation)
    # Remove all punctuation from the stock name
    selected_stock_name = selected_stock_name.translate(translator)
    news_url = f'https://news.google.com/rss/search?hl=en-PH&gl=PH&ceid=PH:en&q={selected_stock_name}'
    feed = feedparser.parse(news_url)
    news_items = []

    for entry in feed.entries:
        source_url = entry.source.get('url') if entry.get('source') else 'Unknown Source'
        sentiment = analyzer.polarity_scores(entry.title)
        compound_score = sentiment['compound']  # Extract the compound score
        news_items.append({
            'title': entry.title,
            'pub_date': entry.published,
            'link': entry.link,
            'source_url': source_url,  # Extract source URL from the source tag
            'sentiment': compound_score
        })

    news = pd.DataFrame(news_items)
    return news
    
st.subheader(f"{selected_stock} Top News")
news = get_news(selected_stock_name)

if not news.empty:
    # Display the most recent 5 news items
    for index, row in news.head(5).iterrows():
        st.markdown(f"[{row['title']}]({row['link']})")
        st.write(f"Published Date: {row['pub_date']}")
        sentiment_score = row['sentiment']
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "grey"
        st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        st.write("---")  # Separator
else:
    st.write("No news found for the selected stock.")
    
