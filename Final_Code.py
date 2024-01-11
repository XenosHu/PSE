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
import re
import os
import tempfile
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import langchain_community.agent_toolkits
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup



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
selected_stock_name = raw[raw['symbol'] == selected_stock]['company_name'].iloc[0]

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

st.subheader(f"{selected_stock_name}({selected_stock}) Stock Price Data")
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

    st.subheader(f"{selected_stock_name}({selected_stock}) Stock Summary")
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
        title=f'{selected_stock_name}({selected_stock}) Closing Price vs. SMA',
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
        title=f'{selected_stock_name}({selected_stock}) Closing Price vs. EMA',
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

def get_annual_report(keyword):
    words = re.split(r'[^\w-]+', keyword)
    if len(words) >= 2:
        # Join the first two words with '%20' for URL encoding
        modified_keyword = "%20".join(words[:2])
    else:
        # Use the whole keyword if it's less than two words
        modified_keyword = "%20".join(words)

    headers_getid = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        'Content-Type': 'application/json',
        'Referer': 'https://edge.pse.com.ph/companyDisclosures/form.do?cmpy_id=665'
    }

    url_getid = f"https://edge.pse.com.ph/autoComplete/searchCompanyNameSymbol.ax?term={modified_keyword}"
    response = requests.get(url=url_getid, headers=headers_getid)
    
    if response.status_code != 200 or not response.json():
        st.error("No company found for the given keyword.")
        return None
        
    id = json.loads(response.text)[0]['cmpyId']
    url_getedge_no ="https://edge.pse.com.ph/companyDisclosures/search.ax"
    headers_getedge_no = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        'Content-Type': 'application/json'
    }
    params_getedge_no = {
        'keyword': id,
        'tmplNm': ''}
    response = requests.get(url=url_getedge_no,headers=headers_getedge_no,params=params_getedge_no)
        
    if response.status_code != 200:
        st.error("Failed to retrieve the annual report.")
        return None
            
    res = response.text
    pattern = r"openPopup\('([^']+)'\);return false;\"\>Annual Report"
    match = re.search(pattern, res)
    
    if not match:
        st.error("No annual report link found.")
        return None
        
    edge_no = match.group(1)
    res_url = f"https://edge.pse.com.ph/openDiscViewer.do?edge_no={edge_no}"
    
    response = requests.get(url=res_url, headers=headers_getedge_no)
    pattern1 = r'<iframe src="([^"]+)" id=\"viewContents\"'
    match = re.search(pattern1, response.text)

    if not match:
        st.error("Failed to update the annual report link.")
        return None
            
    download_idurl = match.group(1)
    res_url = f"https://edge.pse.com.ph{download_idurl}"
    return res_url



st.subheader(f"{selected_stock_name} ({selected_stock})Most Recent Financial Report")
fin_url = get_annual_report(selected_stock_name)
if fin_url:
    st.markdown(f"[{selected_stock_name} Link to the report]({fin_url})")

#-------------------------------------------------------------------------------------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

def get_news(selected_stock_name):
    translator = str.maketrans('', '', string.punctuation)
    # Remove all punctuation from the stock name
    selected_stock_name = selected_stock_name.translate(translator)
    selected_stock_name = selected_stock_name.replace(" ", "%20")
    news_url = f'https://news.google.com/rss/search?hl=en-PH&gl=PH&ceid=PH:en&q={selected_stock_name}'
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        'Content-Type': 'application/json',
        'Cookie' : 'ADS_VISITOR_ID=00000000-0000-0000-0000-000000000000; S=billing-ui-v3=mwWKsavKfFRVJCdtMr0LkBabkWInEe5c:billing-ui-v3-efe=mwWKsavKfFRVJCdtMr0LkBabkWInEe5c; SEARCH_SAMESITE=CgQIjZoB; HSID=A7tl0A4zKGwLDWBZD; SSID=AV_YykyS8U4fqrTsV; APISID=zXTQ1qnjzsdViyvR/A2G9zM7iw-kd3dj2A; SAPISID=xDK3H4bSpYhkaMm1/AIZNeSZsBJevrayRy; __Secure-1PAPISID=xDK3H4bSpYhkaMm1/AIZNeSZsBJevrayRy; __Secure-3PAPISID=xDK3H4bSpYhkaMm1/AIZNeSZsBJevrayRy; GN_PREF=W251bGwsIkNBSVNEQWlwM2ZPc0JoQ1FrWm13QWciXQ__; _ga=GA1.1.227433637.1704783527; SID=fAi3ME1qoAVp3WWOgrHyQd6SFmBkCl9edLWlAFTm9yTzIPuw6zXixzgKx8I-IRYRxEheiA.; __Secure-1PSID=fAi3ME1qoAVp3WWOgrHyQd6SFmBkCl9edLWlAFTm9yTzIPuwgbzNitE3za6hbwyheS_FPQ.; __Secure-3PSID=fAi3ME1qoAVp3WWOgrHyQd6SFmBkCl9edLWlAFTm9yTzIPuwGIABrkwi1Adt28gCBLcKTw.; AEC=Ae3NU9Mauigh2jELiazdCFSWhDcmL0ChGqGYMg4T1HLdWRapg9v0K_1oag; 1P_JAR=2024-01-11-01; OTZ=7377205_76_76_104100_72_446760; _ga_SYGF1G18MM=GS1.1.1704936314.3.0.1704936315.0.0.0; NID=511=J2p2L9i0lCjuzszZ6Rc4CDORTkAIPMWA0GDFyRtv2dlFtJtwmy_Suna_0cuLLkbkGbHXmGuGCyIprMx-j74BYm-CdqEl1Ckhfub0CWyE-BuQGGuI25c11WHYCac3sru9HxSdCm7yg8wlCrVevLguKx2uIAtX30LYv-lDh3W1Nfyruup2xZThikaj730FS5pDbXI8ch3u1P_5F1IZI652IpJrKKIRaTjGC2Q6OSOiSB0YfIlYN3GTZasXbhQjPf7lyomId1fvzI8xevdmKmLXwa-kTo3vR5to4niWSsQsmdlAyFWbJPVJFRopDsUtN5rBqVTfJzIcZxyfrsHhS1oIEjZ23ek0-pQXqWfcfy01hNBPizcFFKntjII3jfBgfjtrzbKO0-39wRY8GPn-7FktB89OmL4Kp_Jr8O1lhhTEPGHuXaFf8he9Gny8yYXl6q23Ak1uj4VrVYwCPAkO44kRCm2hZpJnIkVj4QNqllKEh2ZSiFjODf5t4ksMPwRk43wxpDV1v920JoapcsBsK-01TNus351Q09Y-KpBigOIK4ez0Pa7huz1Alkhht6sYBVdGLXC7mVK1IdpxsB9-ixkwZg2_FSIZ8aj0mNdveZXPUx81dTBoSEjsfoIwcIKjQketykTaeM4EMqzJ5Md6b4FGg2-duWLFJBeVMWHdP61ghOUal1XvjRGKO5L1UFqTC210ityefXCuMbMeXOhQg38Q8WDceHBszIak_Rl21iRGDH4m8olZ_f_ZCt6OmIPvqjoXTmcCi_TpFS6jm3D2hvA0zmqnXMXSVRddtxRfmQiEOQ4qY-BnTp3vjeOdfLAkAdTRidIg9_bkvE2gawgqUaMuI2savikY-kTXVY2FHXtQd3PnkxXwv2O-sCDlaBQ2jgBIpDNIJklEHco8d1Jl4wAj9x1-UdZWkjddn5Kcczt4bVlfYciqmtJw7Lp-4-24nII5kECZuT-oPs9dmuG330MFMpKLvjUXPiX2Z-PK-wEOJmAUyn9nqLlFR9dHcTC-dMSEF3HS_stLwPBhDmcIsH2WPlpqLjqs_iWsL507Hobbtf2mXw; __Secure-1PSIDTS=sidts-CjIBPVxjShncpCDh30nav9iKdTscy491iUyDaPCrQYxT4bVcC6rlATJUXScDxhbelohfOxAA; __Secure-3PSIDTS=sidts-CjIBPVxjShncpCDh30nav9iKdTscy491iUyDaPCrQYxT4bVcC6rlATJUXScDxhbelohfOxAA; SIDCC=ABTWhQEin-VyM5JrtgziHqVcw4vEkqKn8hz_Ak8mdUW0yMUGFElNF7BSBuJlSqNw_esxhZTyqdUu; __Secure-1PSIDCC=ABTWhQFKPjqta4w_Q1nawmXpUuS2oHlhkNvYFoHE4D2u_TfIqF0Si59p8-7fIM8mw-ucL5-CwfQ; __Secure-3PSIDCC=ABTWhQGwZZMlriobXZawp0zAdDE0TzHcqw7K0ZM8zvKJtQ4lrUZ0wR9ErsFQuwlUg2NHccTtoN7z',
        'Sec-Fetch-Dest':'document',
        'Sec-Fetch-Mode':'navigate',
        'Sec-Fetch-Site':'none',
        'Sec-Fetch-User':'?1',
        'Upgrade-Insecure-Requests':'1'}
    response = requests.get(url=news_url, headers=headers)
    response.text

    root = ET.fromstring(response.text)
    # Adjust the loop according to your XML structure
    data = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        link = item.find('link').text
        pubDate = item.find('pubDate').text
        sourceUrl = item.find('source').get('url')
        source = item.find('source').text
        data.append({'title': title, 'link': link, 'pubDate': pubDate, 'sourceUrl':sourceUrl,'source':source})
    news = pd.DataFrame(data)

    return news

# def get_news(keyword):
#     url = f"https://news.google.com/search?q={keyword}&hl=en-PH&gl=PH&ceid=PH:en"
#     news_url = f'https://news.google.com/rss/search?hl=en-PH&gl=PH&ceid=PH:en&q={selected_stock_name}'
#     headers = {
#             'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#             'Content-Type': 'application/json'}
#     response = requests.get(url=url, headers=headers)
#     res = response.text

#     soup = BeautifulSoup(res, 'html.parser')
#     article = soup.find_all('article')
#     url_list = []
#     for a in article:
#         url = "https://news.google.com"+a.div.div.a.get('href')[1:]
#         url_list.append(url)
#     return url_list

    
st.subheader(f"{selected_stock_name}({selected_stock}) Top News")
news = get_news(selected_stock_name)

if not news.empty:
    # Display the most recent 5 news items
    for index, row in news.head(5).iterrows():
        st.markdown(f"[{row['title']}]({row['link']})")
        st.write(f"Published Date: {row['pubDate']}")
        sentiment_score = row['sentiment']
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "grey"
        st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        st.write("---")  # Separator
else:
    st.write("No news found for the selected stock.")
    
#-------------------------------------------------------------------------------------------------------------------------------


# Use Streamlit's secrets for the API key
api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = api_key


# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()



# # Check if 'link' column exists
# if 'link' in news.columns:
#     news['link'] = news['link'].astype(str)
# else:
#     st.error("'link' column not found in news data.")

# Check if URLs are available
# if fin_url and not news.empty:
#     # Combine URLs from the annual report and news articles
# urls = [fin_url] + news['link'].tolist()

urls = ['https://news.google.com/articles/CBMidWh0dHBzOi8vd3d3LnBoaWxzdGFyLmNvbS9idXNpbmVzcy9zdG9jay1jb21tZW50YXJ5LzIwMjMvMTAvMzEvMjMwNzkyNC93aWxjb24tcmVwb3J0cy1xMy1uZXQtaW5jb21lLXA5MDgtbS1kb3duLTE3Ny15edIBAA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMiTmh0dHBzOi8vbWIuY29tLnBoLzIwMjMvMTAvMjUvc3RvY2tzLXNsaWdodGx5LXJlY292ZXItYW1pZC0zLXEtZWFybmluZ3MtcmVsZWFzZdIBAA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMicWh0dHBzOi8vd3d3LmJ3b3JsZG9ubGluZS5jb20vY29ycG9yYXRlLzIwMjMvMDcvMzEvNTM2Nzg1L3dlYWstZWFybmluZ3MtZGlzbWFsLW1hcmtldC1zZW50aW1lbnQtd2VpZ2gtZG93bi13aWxjb24v0gEA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMiTWh0dHBzOi8vYnVzaW5lc3MuaW5xdWlyZXIubmV0LzMzMjEwOC93aWxjb24tZ2V0cy1pbnRvLXBzZWktcmVwbGFjZXMtZmlyc3QtZ2Vu0gEA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMijgFodHRwczovL3d3dy5tYXJrZXRzY3JlZW5lci5jb20vcXVvdGUvc3RvY2svV0lMQ09OLURFUE9ULUlOQy00MjU5MTk1My9uZXdzL1dpbGNvbi1EZXBvdC1JbnRlZ3JhdGVkLUFubnVhbC1Db3Jwb3JhdGUtR292ZXJuYW5jZS1SZXBvcnQtNDA1NjA0Mzgv0gEA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMiWGh0dHBzOi8vd3d3LnBoaWxzdGFyLmNvbS9idXNpbmVzcy8yMDIxLzEwLzA2LzIxMzIyMjUvd2lsY29uLWRlcG90LXJlcGxhY2UtZmlyc3QtZ2VuLXBzZWnSAV1odHRwczovL3d3dy5waGlsc3Rhci5jb20vYnVzaW5lc3MvMjAyMS8xMC8wNi8yMTMyMjI1L3dpbGNvbi1kZXBvdC1yZXBsYWNlLWZpcnN0LWdlbi1wc2VpL2FtcC8?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMiXWh0dHBzOi8vbmV3cy5hYnMtY2JuLmNvbS9idXNpbmVzcy8wNi8xOC8xOC93aWxjb24tc2F5cy1wcmljZXMtdXAtMi0zLXBlcmNlbnQtZHVlLXRvLXdlYWstcGVzb9IBAA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMiQWh0dHBzOi8vYnVzaW5lc3MuaW5xdWlyZXIubmV0LzIyNzExNi93aWxjb24tZ2FpbnMtNS01LWlwby1saXN0aW5n0gEA?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMingFodHRwczovL3d3dy5waGlsc3Rhci5jb20vYnVzaW5lc3Mvc3RvY2stY29tbWVudGFyeS8yMDIyLzA2LzIxLzIxODk5MjAvY29zY28tY2FwaXRhbC1zaWducy1wNTAwLW0tam9pbnQtdmVudHVyZS1nZXQtY29uc3RydWN0aW9uLXN1cHBseS1hbmQtaG91c2V3YXJlcy1idXNpbmVzc9IBowFodHRwczovL3d3dy5waGlsc3Rhci5jb20vYnVzaW5lc3Mvc3RvY2stY29tbWVudGFyeS8yMDIyLzA2LzIxLzIxODk5MjAvY29zY28tY2FwaXRhbC1zaWducy1wNTAwLW0tam9pbnQtdmVudHVyZS1nZXQtY29uc3RydWN0aW9uLXN1cHBseS1hbmQtaG91c2V3YXJlcy1idXNpbmVzcy9hbXAv?hl=en-PH&gl=PH&ceid=PH%3Aen',
 'https://news.google.com/articles/CBMicWh0dHBzOi8vd3d3LnBoaWxzdGFyLmNvbS9idXNpbmVzcy9zdG9jay1jb21tZW50YXJ5LzIwMjEvMDcvMDgvMjExMTA1MC9pa2VhLXBoaWxpcHBpbmVzLWFwb2xvZ2l6ZXMtc2VydmVyLW1lbHRkb3du0gF2aHR0cHM6Ly93d3cucGhpbHN0YXIuY29tL2J1c2luZXNzL3N0b2NrLWNvbW1lbnRhcnkvMjAyMS8wNy8wOC8yMTExMDUwL2lrZWEtcGhpbGlwcGluZXMtYXBvbG9naXplcy1zZXJ2ZXItbWVsdGRvd24vYW1wLw?hl=en-PH&gl=PH&ceid=PH%3Aen']

# Prepare the query for OpenAI model
query = (
    "Access all the links of the news and generate a comprehensive report summary to: "
    "1. Summarize the highlights and insights based on the given materials and the recent challenges or achievements the company faces. "
    "2. In conclusion, recommend to the investor to invest or not to invest in this stock. "
    "3. In no more than 300 words, use a professional tone."
)

# Analyze the URLs with the query using OpenAI model
response = llm.run(query, context=urls)
st.write(response)
    
#     # Attempt to download PDF for the past five years
#     for year in range(current_year, current_year - 5, -1):  # Try the last five years
#         pdf_content = download_pdf(ticker_input, year)
#         if pdf_content:
#             break

#     if not pdf_content:
#         st.error("No report found for the given ticker in the last 5 years.")
#     else:
#         # Process the PDF content
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(pdf_content)
#             temp_file_path = tmp_file.name

#         loader = PyPDFLoader(temp_file_path)
#         pages = loader.load_and_split()

#         store = Chroma.from_documents(pages, embeddings, collection_name='uploaded_document')
#         vectorstore_info = VectorStoreInfo(
#             name="uploaded_document",
#             description="A document uploaded by the user",
#             vectorstore=store
#         )

#         toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
#         agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

#         # Set the default query
#         default_query = (
#             "Access all the links of the news and generate a comprehensive report summary to: "
#             "1. Summarize the highlights and insights based on the given materials and the recent challenges or achievements the company faces. "
#             "2. In conclusion, recommended to the investor to invest or not to invest in this stock. "
#             "3. In no more than 300 words, use a professional tone."
#         )

#         response = agent_executor.run(default_query)
#         st.write(response)


