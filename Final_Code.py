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


# @st.cache_data
# def get_index_data(index_symbol, timeframe):
    
#     # Define end date as today
#     end_date = datetime.now()
        
#     # Calculate start date based on the selected timeframe
#     if timeframe == '5 Day':
#         start_date = end_date - timedelta(days=5)
#     elif timeframe == '1 Week':
#         start_date = end_date - timedelta(weeks=1)
#     elif timeframe == '1 Month':
#         start_date = end_date - timedelta(weeks=4)
#     elif timeframe == '6 Months':
#         start_date = end_date - timedelta(weeks=26)
#     elif timeframe == 'YTD':
#         start_date = datetime(end_date.year, 1, 1)
#     elif timeframe == '1 Year':
#         start_date = end_date - timedelta(weeks=52)
#     elif timeframe == '5 Year':
#         start_date = end_date - timedelta(weeks=260)
    
#     index_data = yf.Ticker(index_symbol).history(start=start_date, end=end_date)

#     return index_data


# @st.cache_data
# def get_currency(stock_symbol):
#     stock_data_other = yf.Ticker(stock_symbol)
#     info = stock_data_other.info['longName']
#     currency = stock_data_other.info['currency']
#     return info,currency

datetime.today()
st.sidebar.title("Select Parameters")

selected_stock = st.sidebar.selectbox("Select a stock symbol", ticker_list)
default_start_date = datetime.today() - timedelta(weeks=52)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date")

if start_date >= end_date:
    st.error("Error: Start date must be before end date.")
    
# Fetch data and display price chart
if index_selection == "Philippines Stock Exchange":
    stock_data = get_stock_data_pse(selected_stock, start_date, end_date)
#other_data = get_currency(selected_stock)

#-------------------------------------------------------------------------------------------------------------------------------
with st.expander(f"**({selected_stock}) Stock Price**"):
    st.write(stock_data)

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

st.subheader(f"{selected_stock} Stock Summary")
if not stock_data.empty:
    # Calculate 1-Year Change
    one_year_change = ((stock_data["C"].iloc[-1] / stock_data["C"].iloc[0]) - 1) * 100
    
    average_vol_3m = stock_data["V"].tail(63).mean()

    prev_close = stock_data["C"].iloc[-2] if len(stock_data["C"]) > 1 else None
    open_price = stock_data["O"].iloc[-1] if len(stock_data["O"]) > 0 else None
    volume = stock_data["V"].iloc[-1] if len(stock_data["V"]) > 0 else None
    day_range = f"{stock_data['L'].iloc[-1]:,.2f}-{stock_data['H'].iloc[-1]:,.2f}" if len(stock_data["L"]) > 0 else "N/A"
    fifty_two_week_range = f"{stock_data['L'].min():,.2f}-{stock_data['H'].max():,.2f}" if not stock_data['L'].empty else "N/A"
    
    stock_summary_data = {
        "Prev. Close": [f"{prev_close:,.2f}" if prev_close is not None else "N/A"],
        "Open": [f"{open_price:,.2f}" if open_price is not None else "N/A"],
        "1-Year Change": [f"{one_year_change:.2f}%" if one_year_change is not None else "N/A"],
        "Volume": [f"{volume:,.0f}" if volume is not None else "N/A"],
        "Average Vol. (3m)": [f"{average_vol_3m:,.0f}" if average_vol_3m is not None else "N/A"],
        "Day's Range": [day_range],
        "52 wk Range": [fifty_two_week_range]
    }
    # Convert the dictionary to a DataFrame
    df_stock_summary = pd.DataFrame.from_dict(stock_summary_data)

    # Display the summary information in a table
    st.table(df_stock_summary)
else:
    st.write("Stock data is not available. Please select a valid stock.")

analyzer = SentimentIntensityAnalyzer()

def get_news(selected_stock):
    news_url = f'https://news.google.com/rss/search?hl=en-PH&gl=PH&ceid=PH:en&q={selected_stock}'
    feed = feedparser.parse(news_url)
    news_items = []

    for entry in feed.entries:
        source_url = entry.source.get('url') if entry.get('source') else 'Unknown Source'
        sentiment = analyzer.polarity_scores(entry.title)
        news_items.append({
            'title': entry.title,
            'pub_date': entry.published,
            'link': entry.link,
            'source_url': source_url,  # Extract source URL from the source tag
            'sentiment': sentiment
        })

    news = pd.DataFrame(news_items)
    news['sentiment'].astype('int')
    return news
    

st.subheader(f"{selected_stock} Top News")
news = get_news(selected_stock)

if not news.empty:
    # Display the most recent 5 news items
    for index, row in news.head(5).iterrows():
        st.markdown(f"[{row['title']}]({row['link']})")
        st.write(f"Published Date: {row['pub_date']}")
        sentiment_score = news['sentiment']
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "white"
        st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        st.write("---")  # Separator
else:
    st.write("No news found for the selected stock.")



        
        


# pricing_data, fundamental_data, news  = st.tabs(["Pricing Data", "Fundamental Data", "Top News"])

# with pricing_data:
#     st.subheader(f'Price Movements for {selected_stock}')
#     updated_data = stock_data
#     updated_data["% Change"] = stock_data["Adj Close"] / stock_data["Adj Close"].shift(1) - 1
#     st.write(updated_data)
    
#     annual_return = updated_data["% Change"].mean()*252*100
#     annual_return_color = "green" if annual_return >= 0 else "red"
#     st.markdown(f"Annual Return: <span style='color:{annual_return_color}'>{round(annual_return, 2)}%</span>", unsafe_allow_html=True)
    
#     stdev = np.std(updated_data["% Change"]) * np.sqrt(252)
#     stdev_color = "green" if stdev >= 0 else "red"
#     st.markdown(f"Standard Deviation is: <span style='color:{stdev_color}'>{round(stdev * 100, 2)}%</span>", unsafe_allow_html=True)
    
#     fig = go.Figure()
    
#     st.subheader(f"{other_data[0]} % Price Change")

#     # Create a condition to determine the color of bars (green for positive and red for negative)
#     positive_mask = updated_data['% Change'] >= 0
#     negative_mask = updated_data['% Change'] < 0

#     # Add bars for positive values
#     fig.add_trace(go.Bar(
#         x=updated_data.index[positive_mask],
#         y=updated_data['% Change'][positive_mask],
#         name=f"{selected_stock} % Change (Positive)",
#         marker_color='rgb(34, 139, 34)'
#     ))

#     # Add bars for negative values (inverted)
#     fig.add_trace(go.Bar(
#         x=updated_data.index[negative_mask],
#         y=updated_data['% Change'][negative_mask],
#         name=f"{selected_stock} % Change (Negative)",
#         marker_color='rgb(220, 20, 60)'
#     ))

#     # Customize the chart layout
#     fig.update_layout(xaxis_title='Date',
#                       yaxis_title='% Price Change',
#                       barmode='relative',
#                       legend=dict(x=0, y=1.2))

#     # Display the chart in the Streamlit app
#     st.plotly_chart(fig)


# analyzer = SentimentIntensityAnalyzer()

#-------------------------------------------------------------------------------------------------------------------------------

# @st.cache_data
# def print_stock_news(stock_symbol):
#     stock = yf.Ticker(stock_symbol)
#     news = stock.news
#     top_news = []
#     for item in news[:5]:
#         title = item['title']
#         link = item['link']
#         publish_date = item['providerPublishTime']
        
#         # Analyze sentiment of the news title
#         sentiment = analyzer.polarity_scores(title)
        
#         news_info = {
#             "title": title,
#             "link": link,
#             "published_date": publish_date,
#             "sentiment": sentiment['compound']  # Compound sentiment score
#         }
#         top_news.append(news_info)
#     return top_news

# if selected_stock:
#     top_5_news = print_stock_news(selected_stock)


# with news:
#     st.subheader(f'Top News for {selected_stock}')
#     for i, news_item in enumerate(top_5_news):

#         st.subheader(f'News {i+1}')
#         st.write("Title:", news_item['title'])
#         st.write("Link:", news_item['link'])

#         sentiment_score = news_item['sentiment']
#         sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "white"

#         st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        
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


# with fundamental_data:
    
#     with st.expander("Definitions of Fundamental Data"):
#         st.write("Market Cap: Market capitalization is the total value of a company's outstanding shares of stock. It is calculated by multiplying the stock's current market price by its total number of outstanding shares.")
#         st.write("Forward P/E: Forward price-to-earnings (P/E) ratio is a valuation ratio that measures a company's current share price relative to its estimated earnings per share for the next year.")
#         st.write("Trailing P/E: Trailing price-to-earnings (P/E) ratio is a valuation ratio that measures a company's current share price relative to its earnings per share over the past 12 months.")
#         st.write("Dividend Yield: Dividend yield is a financial ratio that indicates how much a company pays out in dividends each year relative to its share price. It is usually expressed as a percentage.")
#         st.write("Earnings Per Share (EPS): Earnings per share is a measure of a company's profitability. It represents the portion of a company's profit allocated to each outstanding share of common stock.")
#         st.write("Beta: Beta measures a stock's volatility in relation to the overall market. A beta greater than 1 indicates the stock is more volatile than the market, while a beta less than 1 indicates lower volatility.")

#     st.subheader(f"Fundamental Data for {selected_stock}")

#     fundamental_metrics = get_fundamental_metrics(selected_stock)

#     for metric, value in fundamental_metrics.items():
#         st.write(f"{metric}: {value}")   

# st.header("Stock Price Comparison")

# # Get user input for two stock symbols from the list of tickers

# selected_stock2 = st.selectbox("Select the second stock symbol", ticker_list)

# # Fetch data for second stock
# stock_data2 = get_stock_data(selected_stock2, start_date, end_date)



# if stock_data.empty or stock_data2.empty:
#     st.write("Data not available for one or more selected stock symbols in the specified date range.")

# else:
#     # Create an area chart for the first stock
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=stock_data.index,
#                              y=stock_data['Close'],
#                              mode='lines',
#                              fill='tozeroy',  # Create an area chart
#                              name=f"{selected_stock} Closing Price"))

#     # Create an area chart for the closing prices of the second stock
#     fig.add_trace(go.Scatter(x=stock_data2.index,
#                             y=stock_data2['Close'],
#                             mode='lines',
#                             fill='tozeroy',  # Create an area chart
#                             name=f"{selected_stock2} Closing Price"))

#     fig.update_layout(title='Stock Price Comparison',
#                         xaxis_title='Date',
#                         yaxis_title=f'Price ({other_data[1]})',
#                         legend=dict(x=0, y=1.2))

#     st.plotly_chart(fig)    

# st.header(f"{index_selection} Index Performance")

# index_timeframe = ['5 Day', '1 Week', '1 Month', '6 Months', 'YTD', '1 Year', '5 Year']
# selected_index_timeframe = st.selectbox("Select a timeframe for Index Price", index_timeframe)

# # Fetch historical price data for the selected index
# index_his_data = get_index_data(index_symbol, selected_index_timeframe)


# if index_his_data.empty:
#     st.write("Data not available for the selected index.")
# else:
#     # Create a line chart for the index price
#     fig = go.Figure(data=go.Candlestick(x=index_his_data.index,
#                                        open=index_his_data['Open'],
#                                        high=index_his_data['High'],
#                                        low=index_his_data['Low'],
#                                        close=index_his_data['Close']))

#     # Customize the chart layout
#     fig.update_layout(xaxis_title='Date',
#                       yaxis_title=f'Price ({other_data[1]})',
#                       legend=dict(x=0, y=1.2))

#     # Display the chart in the Streamlit app
#     st.plotly_chart(fig)

# def fetch_market_cap_data(index_tickers):
#     market_cap_data = {}
#     sector_data = {}
    
#     for ticker in index_tickers:
#         try:
#             stock_info = yf.Ticker(ticker).info
#             if "marketCap" in stock_info and stock_info["marketCap"]:
                
#                 market_cap_data[ticker] = {
#                     "MarketCap": float(stock_info["marketCap"]),
#                     "Ticker": (ticker)
#                 }
                
#                 if "sector" in stock_info and stock_info["sector"]:
#                     sector_data[ticker] = {
#                         "Sector": stock_info["sector"]
#                     }
#         except requests.exceptions.HTTPError as e:
#             pass
        
#     market_cap_df = pd.DataFrame(market_cap_data.values(), index=market_cap_data.keys())
#     sector_df = pd.DataFrame(sector_data.values(), index=sector_data.keys())
    
#     # Merge the two DataFrames on the index (ticker symbol)
#     merged_df = market_cap_df.merge(sector_df, left_index=True, right_index=True)
    
#     return merged_df

# if index_selection == "S&P 500":
#     ticker_list_2 = new_tickerlist_sp500
#     market_cap_df = fetch_market_cap_data(ticker_list_2)
#     top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

# elif index_selection == "NASDAQ 100":
#     ticker_list_2 = new_tickerlist_nasdaq
#     market_cap_df = fetch_market_cap_data(ticker_list_2)
#     top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

# elif index_selection == "DOWJONES":
#     ticker_list_2 = new_tickerlist_dowjones
#     market_cap_df = fetch_market_cap_data(ticker_list_2)
#     top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

# elif index_selection == "FTSE 100":
#     ticker_list_2 = new_tickerlist_ftse100
#     market_cap_df = fetch_market_cap_data(ticker_list_2)
#     top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

# elif index_selection == "BSE SENSEX":
#     ticker_list_2 = new_tickerlist_bse
#     market_cap_df = fetch_market_cap_data(ticker_list_2)
#     top_10_stocks = market_cap_df.nlargest(10, "MarketCap")
    
# else:
#     ticker_list_2 = []

# st.header(f'Top Stocks by Market Cap - {index_selection}')
# # Plot a treemap using Plotly Express
# fig = px.treemap(top_10_stocks, path= ['Ticker'], values='MarketCap',
#                  color='MarketCap', color_continuous_scale='Viridis')

# st.plotly_chart(fig)

# index_data = fetch_market_cap_data(ticker_list_2)


# sector_performance = index_data.groupby('Sector')['MarketCap'].sum().reset_index()

# st.subheader(f'Sector-wise Performance for Top stocks - {index_selection}')
# fig = px.bar(sector_performance, x='Sector', y='MarketCap',
#              labels={'Sector': ' ', 'MarketCap': 'Total Market Cap'},
#              color='Sector'
#             )

# fig.update_xaxes(categoryorder='total descending')

# st.plotly_chart(fig)

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

st.header("Trend Analysis using Indicators")
indicator_type = st.selectbox("Select Indicator Type", ["sma", "ema"])

if indicator_type == "sma":
    sma_plot = plot_sma_vs_closing_price(selected_stock, start_date, end_date)
    
elif indicator_type == 'ema':
    ema_plot = plot_ema_vs_closing_price(selected_stock, start_date, end_date)       

