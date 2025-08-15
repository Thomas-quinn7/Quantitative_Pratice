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
        data = pd.concat([data,pd.DataFrame(yf.download(ticker,start=datetime(2022,7,29),
                                                        end=datetime(2025,7,29))['Close'])],axis=1)
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
                             ['s1', 's2', 'pvs', 'pvr'])
    return all_pairs

a=coint_tester(stock_tickers)
print(len(a))
b=yf.Ticker(a.iloc[2].iloc[0])
hist_data = b.history(period='1y')
print(hist_data)

def strat_stats(tickers,item=0,stat_sig=0.05):
    n1=a.iloc[item].iloc[0]
    n2=a.iloc[item].iloc[1]
    s1=data_fetcher(n1)
    s2=data_fetcher(n2)
    if a.iloc[item].iloc[3]<0.05:
        ratio = s1/s2
        z_score = (ratio-ratio.mean())/(ratio.std())
        try:
            mean_val = z_score.mean().item()
        except ValueError:
            # If .item() fails, try other approaches
            mean_val = float(z_score.mean().iloc[0]) if hasattr(z_score.mean(), 'iloc') else float(z_score.mean())

        plt.figure(figsize=(8,6),dpi=200)
        plt.plot(ratio.index,z_score.values,label="z scores")
        plt.axhline(mean_val,color="black")
        plt.axhline(1,color="red")
        plt.axhline(1.25,color="red")
        plt.axhline(-1,color="green")
        plt.axhline(-1.25,color="green")
        plt.legend(loc="best")
        plt.title(f"Ratio between {n1} and {n2}")
        plt.show()
        None

    else:
        None

strat_stats(a)    