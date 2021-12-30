import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm, gmean, cauchy
import seaborn as sns
from datetime import datetime
import time
import pickle
import os

#%matplotlib inline
results = {}

def import_stock_data(tickers, start = '2010-1-1', end = datetime.today().strftime('%Y-%m-%d')):
    data = pd.DataFrame()
    if len([tickers]) ==1:
        data[tickers] = wb.get_data_yahoo(tickers, start = start)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = wb.get_data_yahoo(t, start = start)['Adj Close']
    return(data)


def drift_calc(data, return_type='log'):
    if return_type=='log':
        lr = log_returns(data)
    elif return_type=='simple':
        lr = simple_returns(data)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)
    try:
        return drift.values
    except:
        return drift


def daily_returns(data, days, iterations, return_type='log'):
    ft = drift_calc(data, return_type)
    if return_type == 'log':
        try:
            stv = log_returns(data).std().values
        except:
            stv = log_returns(data).std()
    elif return_type=='simple':
        try:
            stv = simple_returns(data).std().values
        except:
            stv = simple_returns(data).std()    
    #Oftentimes, we find that the distribution of returns is a variation of the normal distribution where it has a fat tail
    # This distribution is called cauchy distribution
    dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
    return dr


def probs_find(predicted, higherthan, on = 'value'):
    """
    This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value (final stock price) or return rate (percentage change)
    Input: 
    1. predicted: dataframe with all the predicted prices (days and simulations)
    2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
    3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
    """
    if on == 'return':
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
        less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higherthan]
        less = [i for i in predList if i < higherthan]
    else:
        print("'on' must be either value or return")
    if len(over) > 0 or len(less) > 0:
        return (len(over)/(len(over)+len(less)))
    else:
        return 0


def log_returns(data):
    return (np.log(1+data.pct_change()))


def simulate_mc(data, days, iterations, return_type='log', plot=False):
    returns = daily_returns(data, days, iterations, return_type)
    price_list = np.zeros_like(returns)
    price_list[0] = data.iloc[-1]
    for t in range(1,days):
        price_list[t] = price_list[t-1]*returns[t]
    
    if plot == True:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        sns.distplot(x, ax=ax[0])
        sns.distplot(x, hist_kws={'cumulative':True},kde_kws={'cumulative':True},ax=ax[1])
        plt.xlabel("Stock Price")
        plt.show()

    try:
        [print(nam) for nam in data.columns]
    except:
        print(data.name)

    days_zero = days - 1
    expected_value = round(pd.DataFrame(price_list).iloc[-1].mean(),2)
    expected_return = round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)
    breakeven_probability = probs_find(pd.DataFrame(price_list),0, on='return')
    print(f"Days: {days_zero}")
    print(f"Expected Value: ${expected_value}")
    print(f"Return: {expected_return}%")
    print(f"Probability of Breakeven: {breakeven_probability}")
   
          
    return pd.DataFrame(price_list)

def monte_carlo(tickers, days_forecast, iterations, start_date = '2000-1-1', return_type = 'log', plotten=False):
    data = import_stock_data(tickers, start=start_date)
    #inform = beta_sharpe(data, mark_ticker="CHR.TO", start=start_date)
    simulatedDF = []
    for t in range(len(tickers)):
        y = simulate_mc(data.iloc[:,t], (days_forecast+1), iterations, return_type)
        if plotten == True:
            forplot = y.iloc[:,0:10]
            forplot.plot(figsize=(15,4))
        #print(f"Beta: {round(inform.iloc[t,inform.columns.get_loc('Beta')],2)}")
        #print(f"Sharpe: {round(inform.iloc[t,inform.columns.get_loc('Sharpe')],2)}") 
        #print(f"CAPM Return: {round(100*inform.iloc[t,inform.columns.get_loc('CAPM')],2)}%")
        y['ticker'] = tickers[t]
        cols = y.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        y = y[cols]
        simulatedDF.append(y)
    simulatedDF = pd.concat(simulatedDF)
    return simulatedDF

start = "2015-1-1"
days_to_forecast= 90
simulation_trials= 10000
symbols = open('symbols.txt', 'r')
Lines = symbols.readlines()

# plz don't ban me yahoo
if os.path.isfile('./results.pkl'):
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

for symbol in Lines:
    symbol = symbol.strip()
    if symbol not in results.keys():
        try:
            results[symbol] = monte_carlo([symbol], days_forecast= days_to_forecast, iterations=simulation_trials,  start_date=start, plotten=False)
        except:
            with open("results.pkl", "wb") as outfile:
                pickle.dump(results, outfile)
        # plz plz
        time.sleep(5)
    print()

with open("results.pkl", "wb") as outfile:
    pickle.dump(results, outfile)
