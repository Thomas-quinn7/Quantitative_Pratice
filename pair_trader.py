import pandas as pd
from datetime import datetime
import yfinance as yf
import gs_quant as gq
import statsmodels.tsa.stattools as ts
import seaborn as sn
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import numpy as np

def data_fetcher(tickers):
    data = pd.DataFrame()
    names = list()

    if isinstance(tickers, str):
        tickers = [tickers]
        
    for ticker in tickers:
        data = pd.concat([data,pd.DataFrame(yf.download(ticker,start=datetime(2022,7,29),
                                                        end=datetime(2025,7,29))['Close'])],axis=1)
        names.append(ticker)
    data.columns = names
    return data

stock_tickers = ['AAPL','GOOG','TSLA','MSFT','NVDA','JPM','AMD','META','AMZN',
                 'BRK-B','PLTR','^SPX','BA','KO','SMCI','RTX','^IXIC']

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
                results_list.append([stock1,stock2,p_value_S,p_value_R,data[stock1],data[stock2]])
            else:
                if p_value_S < stat_significant or p_value_R < stat_significant:
                    print(f"\nPair: {stock1} & {stock2}")
                    print(f"Correlation: {corr:.2f}")
                    print(f"p-value for spread: {p_value_S:.4f}")
                    print(f"p-value for ratio: {p_value_R:.4f}")
                    results_list.append([stock1,stock2,p_value_S,p_value_R,data[stock1],data[stock2]])
    all_pairs = pd.DataFrame(results_list,columns=
                             ['s1', 's2', 'pvs', 'pvr','stock_1_data','stock_2_data'])
    return all_pairs

def strat_stats(pairs_df,item=0,stat_sig=0.05):
    """This function is the base function to understand 
    if you want to proceed to implement the strategy"""
    if len(pairs_df)==0:
        print("No cointegration found")
        return

    n1=pairs_df.iloc[item].iloc[0]
    n2=pairs_df.iloc[item].iloc[1]
    s1=data_fetcher(n1)
    s2=data_fetcher(n2)
    if pairs_df.iloc[item].iloc[3]<stat_sig:
        common_dates = s1.index.intersection(s2.index)
        s1_aligned = s1.loc[common_dates, n1]
        s2_aligned = s2.loc[common_dates, n2]

        ratio = s1_aligned/s2_aligned
        ratio.dropna(inplace=True)

        z_score = (ratio-ratio.mean())/(ratio.std())
        mean_val= z_score.mean()


        plt.figure(figsize=(12,6),dpi=200)
        plt.plot(z_score.index,z_score.values,label="z scores",linewidth=1)
        plt.axhline(mean_val,color="black")
        plt.axhline(1,color="red")
        plt.axhline(1.25,color="red")
        plt.axhline(1.96,color="red")
        plt.axhline(-1,color="green")
        plt.axhline(-1.25,color="green")
        plt.axhline(-1.96,color="red")
        plt.xlabel("Date")
        plt.ylabel("Z Scores of ratio")
        plt.grid(True,alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(loc="best")
        plt.title(f"Ratio between {n1} and {n2}")
        plt.show()
        return ratio, z_score

    else:
        print(f"Ratio for {n1}/{n2} is not statistically significant (p-value: {pairs_df.iloc[item].iloc[3]:.4f})")
        return None, None


def moving_average_strategy(pairs_df, item=0, ma_short=5, ma_long=15, 
                          z_entry=0.5, z_exit=0.1, initial_capital=10000, 
                          transaction_cost=0.001, stop_loss_z=3.0, stop_loss_ratio=0.1,
                          Performance="Y", Graphs="Y"):
    """
    Parameters:
    - pairs_df: DataFrame with cointegrated pairs
    - item: which pair to analyze
    - ma_short: short moving average window
    - ma_long: long moving average window  
    - z_entry: z-score threshold for entry
    - z_exit: z-score threshold for exit
    - initial_capital: starting capital
    - transaction_cost: cost per trade as decimal (0.001 = 0.1% per trade)
    - stop_loss_z: z-score threshold for emergency exit (3.0 = 3 std devs)
    - stop_loss_ratio: ratio change threshold for emergency exit (0.05 = 5% adverse move)
    """
    
    if len(pairs_df) == 0:
        print("No pairs found to analyze")
        return None
        
    n1 = pairs_df.iloc[item]['s1']
    n2 = pairs_df.iloc[item]['s2']
    s1 = pairs_df.iloc[item]['stock_1_data']
    s2 = pairs_df.iloc[item]['stock_2_data']

    common_dates = s1.index.intersection(s2.index)
    s1_aligned = s1.loc[common_dates]
    s2_aligned = s2.loc[common_dates]

    ratio = s1_aligned / s2_aligned
    ratio.dropna(inplace=True)
    z_score = (ratio - ratio.mean()) / ratio.std()

    ma_short_series = z_score.rolling(window=ma_short).mean()
    ma_long_series = z_score.rolling(window=ma_long).mean()

    signals = pd.DataFrame(index=z_score.index)
    signals['z_score'] = z_score
    signals['ma_short'] = ma_short_series
    signals['ma_long'] = ma_long_series
    signals['signal'] = 0
    signals['position'] = 0
    signals['trade_reason'] = ''
    signals['ratio'] = ratio
    signals['entry_ratio'] = np.nan  
    signals['max_favorable_ratio'] = np.nan  
    signals['stop_loss_triggered'] = False
    
    #Strategy logic for more frequent trading with stop losses:
    # 1. Thresholds for entry/exit
    # 2. Multiple entry conditions (MA crossover/momentum/threshold breach)
    # 3. Exit conditions
    # 4. Stop loss protection for black swan events
    
    for i in range(ma_long, len(signals)):
        current_z = signals['z_score'].iloc[i]
        prev_z = signals['z_score'].iloc[i-1]
        current_ma_short = signals['ma_short'].iloc[i]
        current_ma_long = signals['ma_long'].iloc[i]
        prev_ma_short = signals['ma_short'].iloc[i-1]
        prev_ma_long = signals['ma_long'].iloc[i-1]
        prev_position = signals['position'].iloc[i-1]
        current_ratio = signals['ratio'].iloc[i]

        prev_entry_ratio = signals['entry_ratio'].iloc[i-1] if i > 0 else np.nan
        prev_max_favorable = signals['max_favorable_ratio'].iloc[i-1] if i > 0 else np.nan

        ma_cross_up = (prev_ma_short <= prev_ma_long) and (current_ma_short > current_ma_long)
        ma_cross_down = (prev_ma_short >= prev_ma_long) and (current_ma_short < current_ma_long)

        ma_diverging_up = current_ma_short > current_ma_long and (current_ma_short - current_ma_long) > (prev_ma_short - prev_ma_long)
        ma_diverging_down = current_ma_short < current_ma_long and (current_ma_short - current_ma_long) < (prev_ma_short - prev_ma_long)
        
        stop_loss_hit = False
        
        if prev_position != 0 and not np.isnan(prev_entry_ratio):
            z_stop_loss = abs(current_z) > stop_loss_z
            
            if prev_position == 1:
                ratio_stop_loss = (current_ratio / prev_entry_ratio - 1) < -stop_loss_ratio
            else:
                ratio_stop_loss = (current_ratio / prev_entry_ratio - 1) > stop_loss_ratio
            
            trailing_stop = False
            if not np.isnan(prev_max_favorable):
                if prev_position == 1:  
                    trailing_stop = (current_ratio / prev_max_favorable - 1) < -stop_loss_ratio/2
                else: 
                    trailing_stop = (current_ratio / prev_max_favorable - 1) > stop_loss_ratio/2
            
            stop_loss_hit = z_stop_loss or ratio_stop_loss or trailing_stop
            
            if stop_loss_hit:
                signals.loc[signals.index[i], 'signal'] = -prev_position  
                signals.loc[signals.index[i], 'position'] = 0
                signals.loc[signals.index[i], 'stop_loss_triggered'] = True
                
                if z_stop_loss:
                    signals.loc[signals.index[i], 'trade_reason'] = f'STOP LOSS: Z-Score {current_z:.2f} > {stop_loss_z}'
                elif ratio_stop_loss:
                    ratio_change = (current_ratio / prev_entry_ratio - 1) * 100
                    signals.loc[signals.index[i], 'trade_reason'] = f'STOP LOSS: Ratio moved {ratio_change:.1f}% against position'
                elif trailing_stop:
                    signals.loc[signals.index[i], 'trade_reason'] = f'TRAILING STOP: Gave back profits'
                
                signals.loc[signals.index[i], 'entry_ratio'] = np.nan
                signals.loc[signals.index[i], 'max_favorable_ratio'] = np.nan
                continue  
        

        if prev_position == 0:
            long_condition1 = ma_cross_up and current_z < -z_entry*0.8  
            long_condition2 = ma_diverging_up and current_z < -z_entry*0.6 
            long_condition3 = current_z < -z_entry and current_z < prev_z 
            long_condition4 = (current_ma_short > current_ma_long) and current_z < -z_entry*0.4 and (current_z - prev_z) < -0.1 
            
            short_condition1 = ma_cross_down and current_z > z_entry*0.8 
            short_condition2 = ma_diverging_down and current_z > z_entry*0.6 
            short_condition3 = current_z > z_entry and current_z > prev_z 
            short_condition4 = (current_ma_short < current_ma_long) and current_z > z_entry*0.4 and (current_z - prev_z) > 0.1 
            
            if long_condition1 or long_condition2 or long_condition3 or long_condition4:
                signals.loc[signals.index[i], 'signal'] = 1
                signals.loc[signals.index[i], 'position'] = 1
                signals.loc[signals.index[i], 'entry_ratio'] = current_ratio  
                signals.loc[signals.index[i], 'max_favorable_ratio'] = current_ratio  
                
                if long_condition1:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Entry: MA Cross + Oversold'
                elif long_condition2:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Entry: MA Diverging + Oversold'
                elif long_condition3:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Entry: Strong Oversold + Momentum'
                else:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Entry: Bullish MA + Z-Score Drop'
            
            elif short_condition1 or short_condition2 or short_condition3 or short_condition4:
                signals.loc[signals.index[i], 'signal'] = -1
                signals.loc[signals.index[i], 'position'] = -1
                signals.loc[signals.index[i], 'entry_ratio'] = current_ratio 
                signals.loc[signals.index[i], 'max_favorable_ratio'] = current_ratio
                
                if short_condition1:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Entry: MA Cross + Overbought'
                elif short_condition2:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Entry: MA Diverging + Overbought'
                elif short_condition3:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Entry: Strong Overbought + Momentum'
                else:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Entry: Bearish MA + Z-Score Rise'
            else:
                signals.loc[signals.index[i], 'position'] = prev_position
                signals.loc[signals.index[i], 'entry_ratio'] = prev_entry_ratio
                signals.loc[signals.index[i], 'max_favorable_ratio'] = prev_max_favorable
        
        elif prev_position == 1:
            if current_ratio > prev_max_favorable or np.isnan(prev_max_favorable):
                new_max_favorable = current_ratio
            else:
                new_max_favorable = prev_max_favorable
                
            exit_condition1 = ma_cross_down  
            exit_condition2 = abs(current_z) < z_exit  
            exit_condition3 = current_z > z_entry*0.3  
            exit_condition4 = (current_z > prev_z) and current_z > -z_entry*0.3  
            exit_condition5 = current_ma_short < current_ma_long and current_z > -z_entry*0.2  
            
            if exit_condition1 or exit_condition2 or exit_condition3 or exit_condition4 or exit_condition5:
                signals.loc[signals.index[i], 'signal'] = -1
                signals.loc[signals.index[i], 'position'] = 0
                signals.loc[signals.index[i], 'entry_ratio'] = np.nan  
                signals.loc[signals.index[i], 'max_favorable_ratio'] = np.nan
                
                if exit_condition1:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Exit: MA Cross Down'
                elif exit_condition2:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Exit: Mean Reversion'
                elif exit_condition3:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Exit: Moved Overbought'
                elif exit_condition4:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Exit: Z-Score Momentum Reversal'
                else:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Long Exit: Bearish MA Setup'
            else:
                signals.loc[signals.index[i], 'position'] = prev_position
                signals.loc[signals.index[i], 'entry_ratio'] = prev_entry_ratio
                signals.loc[signals.index[i], 'max_favorable_ratio'] = new_max_favorable
        
        elif prev_position == -1:
            if current_ratio < prev_max_favorable or np.isnan(prev_max_favorable):
                new_max_favorable = current_ratio
            else:
                new_max_favorable = prev_max_favorable
                
            exit_condition1 = ma_cross_up  
            exit_condition2 = abs(current_z) < z_exit  
            exit_condition3 = current_z < -z_entry*0.3  
            exit_condition4 = (current_z < prev_z) and current_z < z_entry*0.3  
            exit_condition5 = current_ma_short > current_ma_long and current_z < z_entry*0.2  
            
            if exit_condition1 or exit_condition2 or exit_condition3 or exit_condition4 or exit_condition5:
                signals.loc[signals.index[i], 'signal'] = 1
                signals.loc[signals.index[i], 'position'] = 0
                signals.loc[signals.index[i], 'entry_ratio'] = np.nan  
                signals.loc[signals.index[i], 'max_favorable_ratio'] = np.nan
                
                if exit_condition1:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Exit: MA Cross Up'
                elif exit_condition2:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Exit: Mean Reversion'
                elif exit_condition3:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Exit: Moved Oversold'
                elif exit_condition4:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Exit: Z-Score Momentum Reversal'
                else:
                    signals.loc[signals.index[i], 'trade_reason'] = 'Short Exit: Bullish MA Setup'
            else:
                signals.loc[signals.index[i], 'position'] = prev_position
                signals.loc[signals.index[i], 'entry_ratio'] = prev_entry_ratio
                signals.loc[signals.index[i], 'max_favorable_ratio'] = new_max_favorable
    
    signals['ratio'] = ratio
    signals['ratio_returns'] = ratio.pct_change()
    
    signals['position_change'] = signals['position'].diff().fillna(0)
    signals['trade_occurred'] = (signals['position_change'] != 0).astype(int)
    
    signals['transaction_costs'] = signals['trade_occurred'] * transaction_cost * 2  
    
    signals['gross_strategy_returns'] = signals['position'].shift(1) * signals['ratio_returns']
    
    signals['net_strategy_returns'] = signals['gross_strategy_returns'] - signals['transaction_costs']
    
    signals['gross_cumulative_returns'] = (1 + signals['gross_strategy_returns']).cumprod()
    signals['net_cumulative_returns'] = (1 + signals['net_strategy_returns']).cumprod()
    
    signals['gross_portfolio_value'] = initial_capital * signals['gross_cumulative_returns']
    signals['net_portfolio_value'] = initial_capital * signals['net_cumulative_returns']
    
    gross_total_return = (signals['gross_portfolio_value'].iloc[-1] / initial_capital - 1) * 100
    net_total_return = (signals['net_portfolio_value'].iloc[-1] / initial_capital - 1) * 100
    
    gross_annual_return = ((signals['gross_portfolio_value'].iloc[-1] / initial_capital) ** (252/len(signals)) - 1) * 100
    net_annual_return = ((signals['net_portfolio_value'].iloc[-1] / initial_capital) ** (252/len(signals)) - 1) * 100
    
    gross_volatility = signals['gross_strategy_returns'].std() * np.sqrt(252) * 100
    net_volatility = signals['net_strategy_returns'].std() * np.sqrt(252) * 100
    
    gross_sharpe_ratio = gross_annual_return / gross_volatility if gross_volatility > 0 else 0
    net_sharpe_ratio = net_annual_return / net_volatility if net_volatility > 0 else 0
    
    gross_max_dd = ((signals['gross_portfolio_value'] / signals['gross_portfolio_value'].cummax()) - 1).min() * 100
    net_max_dd = ((signals['net_portfolio_value'] / signals['net_portfolio_value'].cummax()) - 1).min() * 100
    
    trades = signals[signals['signal'] != 0]
    stop_loss_trades = signals[signals['stop_loss_triggered'] == True]
    num_trades = len(trades)
    num_stop_losses = len(stop_loss_trades)
    total_transaction_costs = signals['transaction_costs'].sum() * initial_capital
    cost_impact = (gross_total_return - net_total_return)
    
    if Performance=="Y":
        print(f"\n=== Moving Average Strategy Results for {n1}/{n2} ===")
        print(f"Strategy Parameters:")
        print(f"  - Short MA: {ma_short} days")
        print(f"  - Long MA: {ma_long} days") 
        print(f"  - Entry Z-Score: ±{z_entry}")
        print(f"  - Exit Z-Score: ±{z_exit}")
        print(f"  - Stop Loss Z-Score: ±{stop_loss_z}")
        print(f"  - Stop Loss Ratio: {stop_loss_ratio*100:.1f}%")
        print(f"  - Transaction Cost: {transaction_cost*100:.2f}% per trade")
        print(f"\nTrading Activity:")
        print(f"  - Total Trades: {num_trades}")
        print(f"  - Stop Loss Exits: {num_stop_losses} ({num_stop_losses/num_trades*100:.1f}% of trades)")
        print(f"  - Normal Exits: {num_trades - num_stop_losses}")
        print(f"  - Total Transaction Costs: ${total_transaction_costs:.2f}")
        print(f"  - Cost Impact on Returns: -{cost_impact:.2f}%")
        print(f"\nPerformance Metrics (Gross vs Net):")
        print(f"  - Total Return: {gross_total_return:.2f}% → {net_total_return:.2f}%")
        print(f"  - Annualised Return: {gross_annual_return:.2f}% → {net_annual_return:.2f}%")
        print(f"  - Volatility: {gross_volatility:.2f}% → {net_volatility:.2f}%")
        print(f"  - Sharpe Ratio: {gross_sharpe_ratio:.2f} → {net_sharpe_ratio:.2f}")
        print(f"  - Max Drawdown: {gross_max_dd:.2f}% → {net_max_dd:.2f}%")
        print(f"  - Number of Trades: {num_trades}")      
        print(f"  - Total Transaction Costs: ${total_transaction_costs:.2f}")
        print(f"  - Cost Impact on Returns: -{cost_impact:.2f}%")

    if Graphs=="Y":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
        ax1.plot(signals.index, signals['z_score'], label='Z-Score', alpha=0.7)
        ax1.plot(signals.index, signals['ma_short'], label=f'MA {ma_short}', linewidth=2)
        ax1.plot(signals.index, signals['ma_long'], label=f'MA {ma_long}', linewidth=2)
        ax1.axhline(z_entry, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(-z_entry, color='green', linestyle='--', alpha=0.7)
        ax1.axhline(z_exit, color='orange', linestyle=':', alpha=0.7)
        ax1.axhline(-z_exit, color='orange', linestyle=':', alpha=0.7)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax1.set_title(f'Z-Score and Moving Averages: {n1}/{n2}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
    
        ax2.plot(signals.index, signals['z_score'], label='Z-Score', alpha=0.7)
        ax2.scatter(buy_signals.index, buy_signals['z_score'], color='green', 
               marker='^', s=100, label='Buy Signal', zorder=5)
        ax2.scatter(sell_signals.index, sell_signals['z_score'], color='red',
               marker='v', s=100, label='Sell Signal', zorder=5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Trading Signals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(signals.index, signals['gross_portfolio_value'], linewidth=2, 
             color='blue', label='Gross Returns', alpha=0.8)
        ax3.plot(signals.index, signals['net_portfolio_value'], linewidth=2, 
             color='red', label='Net Returns (After Costs)', alpha=0.8)
        ax3.axhline(initial_capital, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
        ax3.set_title('Portfolio Value: Gross vs Net Returns')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        cumulative_costs = (signals['transaction_costs'] * initial_capital).cumsum()
        ax4.plot(signals.index, cumulative_costs, linewidth=2, color='purple', label='Cumulative Transaction Costs')
        ax4.fill_between(signals.index, cumulative_costs, alpha=0.3, color='purple')
        ax4.set_title('Cumulative Transaction Costs Over Time')
        ax4.set_ylabel('Cumulative Costs ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    return signals, {
        'gross_total_return': gross_total_return,
        'net_total_return': net_total_return,
        'gross_annual_return': gross_annual_return,
        'net_annual_return': net_annual_return,
        'gross_volatility': gross_volatility,
        'net_volatility': net_volatility,
        'gross_sharpe_ratio': gross_sharpe_ratio,
        'net_sharpe_ratio': net_sharpe_ratio,
        'gross_max_drawdown': gross_max_dd,
        'net_max_drawdown': net_max_dd,
        'num_trades': num_trades,
        'total_transaction_costs': total_transaction_costs,
        'cost_impact': cost_impact
    }



def optimisation_parms(pairs_df,pair_index,params=None, optimisation_metric='net_sharpe_ratio'):
    if params is None:
        params= {
            'ma_short' : [3, 5, 7, 10],
            'ma_long' : [10, 15, 20, 30],
            'exit_z': [0, 0.05, 0.1, 0.25],
            'entry_z': [0.5, 0.75, 1, 1.5]
        }

        best_result = None
        best_score = -float('inf')

        for Short in params['ma_short']:
            for Long in params['ma_long']:
                if Long <= Short:
                    continue
                for Exit in params['exit_z']:
                    for Entry in params['entry_z']:
                        if Entry <= Exit:
                            continue
                        try:
                            signals, performance = moving_average_strategy(
                                pairs_df, item = pair_index, 
                                ma_short = Short,
                                ma_long = Long,
                                z_entry = Entry,
                                z_exit = Exit,
                                Performance="N",
                                Graphs="N"
                            )

                            if optimisation_metric == 'net_sharpe_ratio':
                                score = performance['net_sharpe_ratio']
                            elif optimisation_metric == 'net_total_return':
                                score = performance['net_total_return']
                            elif optimisation_metric == 'net_annual_return':
                                score = performance['net_annual_return']
                            elif optimisation_metric == 'return_to_drawdown':
                                if performance['net_max_drawdown'] != 0:
                                    score = performance['net_total_return'] / abs(performance['net_max_drawdown'])
                                else:
                                    score = performance['net_total_return']
                            elif optimisation_metric == 'profit_factor':
                                trades = signals[signals['signal'] != 0]
                                if len(trades) > 0:
                                    returns = signals['net_strategy_returns']
                                    gross_profit = returns[returns > 0].sum()
                                    gross_loss = abs(returns[returns < 0].sum())
                                    score = gross_profit / gross_loss if gross_loss > 0 else gross_profit
                                else:
                                    score = 0
                            else:
                                raise ValueError(f"Unknown optimiation metric: {optimisation_metric}")
                            
                            if score > best_score:
                                best_score = score
                                best_result = {
                                    'ma_short': Short,
                                    'ma_long': Long,
                                    'z_entry': Entry,
                                    'z_exit': Exit,
                                    'performance': performance,
                                    'optimisation_score':score,
                                    'optimisated metric': optimisation_metric
                                }
                        except:
                            continue
    
    if best_result:
        print(f"Best parameters found (optimised for {optimisation_metric}):")
        print(f"  MA Short: {best_result['ma_short']}")
        print(f"  MA Long: {best_result['ma_long']}")
        print(f"  Z Entry: {best_result['z_entry']}")
        print(f"  Z Exit: {best_result['z_exit']}")
        print(f"  {optimisation_metric}: {best_result['optimisation_score']:.3f}")
        print(f"  Net Total Return: {best_result['performance']['net_total_return']:.2f}%")
        print(f"  Net Sharpe Ratio: {best_result['performance']['net_sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {best_result['performance']['net_max_drawdown']:.2f}%")
    
    return best_result



if __name__ == "__main__":
    stock_tickers = ['AAPL','GOOG','TSLA','MSFT','NVDA','JPM','AMD','META','AMZN',
                     'BRK-B','PLTR','^SPX','BA','KO','SMCI','RTX','^IXIC','RYA.IR',
                     'A5G.IR','BIRG.IR','KRZ.IR','GL9.IR']
    
    pairs = coint_tester(stock_tickers)
    print(f"Found {len(pairs)} cointegrated pairs")
    all_results=[]
    if len(pairs) > 0:
        for i in range(len(pairs)):
            pair_info = pairs.iloc[i]
            try:
                signals, performance = moving_average_strategy(
                    pairs, 
                    item=i,                   
                    ma_short=5,                
                    ma_long=15,                
                    z_entry=0.5,              
                    z_exit=0.1,               
                    initial_capital=10000,
                    transaction_cost=0.001    
                )
                if signals is not None and performance is not None:
                    result = {
                        'pair_index': i,
                        'stock1': pair_info['s1'],
                        'stock2': pair_info['s2'],
                        'p_value_spread': pair_info['pvs'],
                        'p_value_ratio': pair_info['pvr'],
                        'signals': signals,
                        'performance': performance,
                        'success': True
                    }
                    all_results.append(result)
            except Exception as e:
                print(f"Error analysising pair {pair_info["s1"]}/{pair_info["s2"]}:{str(e)}")
                all_results.append({
                    'pair_index': i,
                    'stock1': pair_info['s1'],
                    'stock2': pair_info['s2'],
                    'success': False,
                    'Error': str(e)
                })

    successful_results = [r for r in all_results if r.get('success', False)]
    if len(successful_results)>0:
        summary_data=[]
        for result in successful_results:
            perf = result['performance']
            summary_data.append({
                'Pair': f"{result['stock1']}/{result['stock2']}",
                'Net Return (%)': perf['net_total_return'],
                'Net Annual (%)': perf['net_annual_return'],
                'Net Sharpe': perf['net_sharpe_ratio'],
                'Max Drawdown (%)': perf['net_max_drawdown'],
                'Num Trades': perf['num_trades'],
                'Total Costs ($)': perf['total_transaction_costs'],
                'Volatility (%)': perf['net_volatility']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Net Return (%)', ascending=False)

        print("\nAll successful strategies by Net returns(%) :")
        print(summary_df.to_string(index=False, float_format='%.2f'))

        best_return = summary_df.iloc[0]
        best_sharpe = summary_df.iloc[summary_df['Net Sharpe'].idxmax()]
        best_risk_adj = summary_df.iloc[(summary_df['Net Return (%)'] / summary_df['Max Drawdown (%)'].abs()).idxmax()]
        risk_adj_ratio = best_risk_adj['Net Return (%)'] / abs(best_risk_adj['Max Drawdown (%)'])

        print(f"\nBest net Return {best_return['Pair']}({best_return['Net Return (%)']:.2f}%)")
        print(f"Best Sharpe ratio {best_sharpe['Pair']}({best_sharpe['Net Sharpe']:.2f})")
        print(f"Best Risk-Adjusted: {best_risk_adj['Pair']} (Return/MaxDD: {risk_adj_ratio:.2f})")
    optimisation_parms(pairs, 1)
