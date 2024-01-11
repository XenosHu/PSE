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
import openai
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import langchain_community.agent_toolkits
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import time


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

def parse_and_format_date(date_str):
    # Extract the year if available
    year_match = re.search(r', (\d{4})', date_str)
    year = year_match.group(1) if year_match else str(datetime.now().year-1)

    # Remove the year part from the date string
    date_str = re.sub(r', \d{4}', '', date_str)

    # Convert the month abbreviation to a numerical month
    datetime_obj = datetime.strptime(date_str, "%b %d")

    # Set the year
    datetime_obj = datetime_obj.replace(year=int(year))

    # Format the date as 'YYYY-MM-DD' for sorting
    return datetime_obj.strftime("%Y-%m-%d")


def get_googlenews(keyword):
    url = f"https://news.google.com/search?q={keyword}&hl=en-PH&gl=PH&ceid=PH:en"
    headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            'Content-Type': 'application/json'}
    response = requests.get(url=url, headers=headers)
    res = response.text

    res_list = []
    soup = BeautifulSoup(res, 'html.parser')
    article = soup.find_all('article')
    url_list = []
    for a in article:
        elist = []
        for element in a.find_all('div'):
            if element.parent is a:
                elist.append(element)
        name = elist[0].text
        date = parse_and_format_date(elist[1].next.text)
        url = "https://news.google.com"+elist[0].a.get('href')[1:]
        sentiment = analyzer.polarity_scores(name)
        res_list.append({
            "name":name,
            "url":url,
            "date":date,
            "sentiment": sentiment['compound'] 
        })

    sorted_data = sorted(res_list, key=lambda x: x["date"], reverse=True)
    if len(sorted_data)>3:
        return sorted_data[:1]
    else:
        return sorted_data

st.subheader(f"{selected_stock_name}({selected_stock}) Top News")

news_url = get_googlenews(selected_stock_name)
news_url_df = pd.DataFrame(news_url)

if not news_url_df.empty:
    # Display the most recent 5 news items
    for index, row in news_url_df.iterrows():
        st.markdown(f"[{row['name']}]({row['url']})")
        st.write(f"Published Date: {row['date']}")
        sentiment_score = row['sentiment']
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "grey"
        st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        st.write("---")  # Separator
else:
    st.write("No news found for the selected stock.")
    
def get_rdcontent(ul):
    content = []
    headers = {
                    'User-Agent': "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
                    'Content-Type': 'application/json'}
    for u in ul:
        response = requests.get(url=u['url'], headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            link = soup.find_all('a')[-1].get('href')
            response1 = requests.get(url=link, headers=headers)
            if response1.status_code == 200:
                soup1 = BeautifulSoup(response1.text, 'html.parser')
                content.append(soup1.find('body').text.replace("\n","").replace("\t","").replace("\r","")) if soup.find('body')!=None else ""
        time.sleep(1)
    return content


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

    
#-------------------------------------------------------------------------------------------------------------------------------


api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = api_key

# Initialize the OpenAI model from Langchain
llm = OpenAI(api_key=api_key, temperature=0.1)

# Fetch the content from the news URLs
content = get_rdcontent(news_url)
combined_content = ' '.join(content)  # Join all contents into a single string

# Prepare the query for the model
query = (
    "Analyze following text and generate a comprehensive report: "
    f"{combined_content} "
    "Summarize insights based on given materials and the recent challenges or achievements that company faces. "
    "In conclusion, recommend to investors to invest or not to invest in this stock. "
    "Provide report in no more than 300 words, using a clear and professional tone."
)


# Generate the response using the OpenAI model from Langchain
try:
    # Wrapping the query in a list
    response = llm.generate([query], max_tokens=1000)
    st.write(response)
except Exception as e:
    st.error(f"Error: {e}")



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


