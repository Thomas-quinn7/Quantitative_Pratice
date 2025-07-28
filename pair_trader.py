import pandas as pd
from datetime import datetime
import yfinance as yf
import gs_quant as gq
import statsmodels.tsa.stattools as ts
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

def data_fetcher(tickers):
    data = pd.DataFrame()
    names = list()
    for ticker in tickers:
        data = pd.concat([data,pd.DataFrame(yf.download(ticker,start=datetime(2022,7,14),
                                                        end=datetime(2025,7,14))['Close'])],axis=1)
        names.append(ticker)
    data.columns = names
    return data

stock_tickers = ['AAPL','GOOG','TSLA','MSFT','NVDA','JPM','AMD','META','AMZN',
                 'BRK-B','PLTR','^SPX','BA','KO','SMCI','RTX','^IXIC']
data = data_fetcher(stock_tickers)

def heatmap(tickers):
    d=data_fetcher(tickers)
    corr_matrix = d.corr()
    plt.figure(figsize = (10,10), dpi = 100)
    sn.heatmap(corr_matrix,annot=True)
    plt.savefig('heatmap.png')
    plt.close()
    return d

def coint_tester(tickers,corr_threshold=0.9,Output_adfuller=True,stat_significant=0.05):
    data = data_fetcher(tickers)
    corr_matrix = data.corr()
    pairs = list(combinations(tickers, 2))
    results_list=[]

    for stock1, stock2 in pairs:
        corr = corr_matrix.loc[stock1, stock2]
        if abs(corr) > corr_threshold:
            spread = data[stock1] - data[stock2]
            ratio = data[stock1]/data[stock2]
            spread.dropna(inplace=True)
            ratio.dropna(inplace=True)
            p_value_S = adfuller(spread)[1]
            p_value_R = adfuller(ratio)[1]

            if Output_adfuller == False:
                print(f"\nPair: {stock1} & {stock2}")
                print(f"Correlation: {corr:.2f}")
                print(f"p-value for spread: {p_value_S:.4f}")
                print(f"p-value for ratio: {p_value_R:.4f}")
                results_list.append([stock1,stock2,p_value_S,p_value_R])
            else:
                if p_value_S < stat_significant or p_value_R < stat_significant:
                    print(f"\nPair: {stock1} & {stock2}")
                    print(f"Correlation: {corr:.2f}")
                    print(f"p-value for spread: {p_value_S:.4f}")
                    print(f"p-value for ratio: {p_value_R:.4f}")
                    results_list.append([stock1,stock2,p_value_S,p_value_R])
    all_pairs = pd.DataFrame(results_list,columns=
                             ['Stock 1', 'Stock 2', 'p_value for Spread', 'p_value for Ratio'])
    return all_pairs

a=coint_tester(stock_tickers)
print(a)
