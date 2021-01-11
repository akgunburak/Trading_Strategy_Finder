# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 18:47:52 2021

@author: burak
"""


import pandas as pd
import yfinance as yf 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from itertools import combinations

#Initialize a class named stock
class stock:
    def __init__(self, ticker, start_date = "2015-1-1", end_date = dt.date.today(), interval = "1d"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = yf.download(self.ticker, self.start_date, self.end_date, interval = self.interval)
    
    
    #Calculate the moving averages and buy & sell signals
    def mavg(self, shortmavg = 20, longmavg = 200):
        #Data
        data = self.data["Close"].dropna()
        df = pd.DataFrame()
        
        #Calculating long and short moving averages
        df["short"] = np.round(data.rolling(window = shortmavg).mean(), 2)
        df["long"] = np.round(data.rolling(window = longmavg).mean(), 2)
        
        #Strategy:Buy if the short mavg is greter than the long mavg as the treshold value, sell vice-versa 
        df["max"] = np.round(data.rolling(window = shortmavg).max(), 2)
        df["min"] = np.round(data.rolling(window = shortmavg).min(), 2)
        df["signal"] = np.where(np.logical_and(df["short"] - df["long"] > 0, data > df["max"] ), 1, 0)
        df["signal"] = np.where(np.logical_and(df["short"] - df["long"] < 0, data < df["min"] ), -1, df["signal"])     
        return(df["signal"])
    
    
    #Calculate stochastic and buy & sell signals
    def stochastic(self):
        #Data
        data = self.data["Close"].dropna()
        df = pd.DataFrame()
        
        #Calculating 14 day max and min values 
        df["max14"] = data.rolling(window = 14).max()
        df["low14"] = data.rolling(window = 14).min()
        
        #Calculating the stochastic
        df["stochastic"] = (data - df["low14"]) / (df["max14"] - df["low14"]) * 100
        df["smoothed"] = df["stochastic"].rolling(window = 3).mean()
         
        #Strategy:While under the 20, buy if the stochastic crosses the smoothed from downside to up,
        #while above the 80, sell if the stochastic crosses the smoothed from upside to down
        df["signal"] = np.where(np.logical_and(df["stochastic"] > df["smoothed"], df["stochastic"] < 20), 1, 0)
        df["signal"] = np.where(np.logical_and(df["stochastic"] < df["smoothed"], df["stochastic"] > 80), -1, df["signal"])
        return(df["signal"])


    #Calculate bollinger bands and buy & sell signals
    def bollinger_bands(self, mavg = 20, std = 2):
        #Data
        data = self.data["Close"].dropna()
        df = pd.DataFrame()
        
        #Calculating bollinger bands
        df["middle"] = data.rolling(window = mavg).mean()
        df["upper"] = df["middle"] + (std * (data.rolling(window = mavg).std()))
        df["lower"] = df["middle"] - (std * (data.rolling(window = mavg).std()))
        
        #Strategy:Buy if the price crosses to lower band, sell if it crosses the upper band
        df["signal"] = np.where(data < df["lower"], 1, 0)
        df["signal"] = np.where(data > df["upper"], -1, df["signal"])
        return(df["signal"])
    
    
    #Calculate CCI and buy & sell signals
    def cci(self):
        #Data
        df = pd.DataFrame()
        
        #Calculating CCI
        df["typical_prices"] = (self.data["Close"] + self.data["High"] + self.data["Low"]) / 3
        df["tp_mean"] = df["typical_prices"].rolling(window = 20).mean()      
        df["deviations"] = (abs(df["typical_prices"] - df["tp_mean"])).rolling(window = 20).sum() / 20
        df["cci"] = (df["typical_prices"] - df["tp_mean"]) / (0.015 * df["deviations"])
        
        #Strategy: Buy if CCI < -100 , sell if CCI > 100
        df["signal"] = np.where(df["cci"] < -100, 1, 0)
        df["signal"] = np.where(df["cci"] > 100, -1, df["signal"])
        return(df["signal"])
    
    
    #Calculate MACD and buy & sell signals
    def macd(self):
        #Data
        data = self.data["Close"].dropna()
        df = pd.DataFrame()
        
        #Calculating MACD
        df["ema12"] = data.ewm(span = 12).mean()
        df["ema26"] = data.ewm(span = 26).mean()
        df["macd"] = df["ema12"] - df["ema26"]
        df["signal_line"] = df["macd"].ewm(span = 9).mean()
        
        #Strategy: Buy if macd > signal line, sell if macd < signal line
        df["signal"] = np.where(df["macd"] > df["signal_line"], 1, 0)
        df["signal"] = np.where(df["macd"] < df["signal_line"], -1, df["signal"])
        return(df["signal"])
    
    
    #Calculate RSI and buy & sell signals
    def rsi(self):
        #Data
        data = self.data["Close"].dropna()
        df = pd.DataFrame()
        
        #Calculating the change
        df["change"] = data.diff(1)
        #Determining the positive and negative changes
        df["pos_chg"] = df["change"][df["change"] > 0]
        df["neg_chg"] = -(df["change"][df["change"] < 0])
        #Converting Nan to "0" 
        df["pos_chg"].fillna(0, inplace = True)
        df["neg_chg"].fillna(0, inplace = True)
        #Calculating the averages of positive and negative days
        df["avg_gain"] = df['pos_chg'].rolling(window = 14).mean()
        df["avg_loss"] = df['neg_chg'].rolling(window = 14).mean()
        #Calculating RS
        df["rs"] = df["avg_gain"] / df["avg_loss"]
        #Calculating RSI
        df["rsi"] = 100 - (100 / (1 + df["rs"]))
        #Strategy: Buy if RSI < 30, sell if RSI > 70
        df["signal"] = np.where(df["rsi"] < 30, 1, 0)
        df["signal"] = np.where(df["rsi"] > 70, -1, df["signal"])
        return(df["signal"])
    
    
    #Calculate ADX and buy & sell signals
    def adx(self):
        #Data
        data = self.data
        df = pd.DataFrame()
        
        #Calculating true range and average true range
        df["h-l"] = data["High"] - data["Low"]
        df["h-c"] = abs(data["High"] - data["Close"].shift())
        df["l-c"] = abs(data["Low"] - data["Close"].shift())
        df["tr"] = float()
        x = 0
        for i in df["tr"]:
            if df["h-l"][x] > df["h-c"][x] and  df["h-l"][x] > df["l-c"][x]:
                df["tr"][x] = df["h-l"][x]
            if df["h-c"][x] > df["h-l"][x] and df["h-c"][x] > df["l-c"][x]:
                df["tr"][x] = df["h-c"] [x]
            if df["l-c"][x] > df["h-l"][x] and df["l-c"][x] > df["h-c"][x]:
                df["tr"][x] = df["l-c"][x]
            x += 1
        a = 15
        df["atr"] = float()
        df["atr"][14] = df["tr"][:14].mean()
        for i in df["atr"][15:]:
            df["atr"][a] = (((df["atr"][a-1] * 13) + df["tr"][a]) / 14)
            a += 1
        
        #Calculating directional movements and smoothed 
        df["h-h"] = float()
        y = 0
        for i in df["h-h"]:
            if y == 0:
                y += 1
                continue
            if data["High"][y] - data["High"][y-1] > 0:
                df["h-h"][y] = data["High"][y] - data["High"][y-1]
            y += 1
        df["+dx"] = np.where(data["High"] - data["High"].shift() > data["Low"].shift() - data["Low"], df["h-h"], 0)
        df["l-l"] = float()
        z = 0
        for i in df["l-l"]:
            if z == 0:
                z += 1
                continue
            if data["Low"][z-1] - data["Low"][z] > 0:
                df["l-l"][z] = data["Low"][z-1] - data["Low"][z]
            z += 1
        df["-dx"] = np.where(data["High"] - data["High"].shift() < data["Low"].shift() - data["Low"], df["l-l"], 0)
        b = 15
        df["+dx_smoothed"] = float(0)
        df["+dx_smoothed"][14] = df["+dx"][:14].mean()
        df["-dx_smoothed"] = float(0)
        df["-dx_smoothed"][14] = df["-dx"][:14].mean()
        for i in df["+dx_smoothed"][15:]:
            df["+dx_smoothed"][b] = (((df["+dx_smoothed"][b-1] * 13) + df["+dx"][b]) / 14)
            df["-dx_smoothed"][b] = (((df["-dx_smoothed"][b-1] * 13) + df["-dx"][b]) / 14)
            b += 1
        df["+dmi"] = (df["+dx_smoothed"] / df["atr"]) * 100
        df["-dmi"] = (df["-dx_smoothed"] / df["atr"]) * 100
        df["dx"] = (abs(df["+dmi"] - df["-dmi"]) / (df["+dmi"] + df["-dmi"])) * 100
        df["adx"] = df["dx"].rolling(window = 14).mean()
        #Strategy:While adx > 25, buy if +DMI > -DMI , sell vice-versa
        df["signal"] = np.where(np.logical_and(df["adx"] > 25, df["+dmi"] > df["-dmi"]), 1, 0)
        df["signal"] = np.where(np.logical_and(df["adx"] > 25, df["-dmi"] > df["+dmi"]), -1, df["signal"])
        return(df["signal"])
        
        
    #Calculate Williams %R and buy & sell signals
    def williamsr(self):
        #Data
        df = pd.DataFrame()
        
        #Calculating 14 day high and low
        df["max14"] = self.data["High"].rolling(window = 14).max()
        df["low14"] = self.data["Low"].rolling(window = 14).min()
        # %R
        df["r"] = ((df["max14"] - self.data["Close"]) / (df["max14"] - df["low14"])) * -100
        
        #Strategy: buy if r < -80 , sell if r > -20
        df["signal"] = np.where(df["r"] > -20, 1, 0)
        df["signal"] = np.where(df["r"] < -80, -1, df["signal"])
        return(df["signal"])
     
    
    #Determine the strategy;
    #Buy if half of the selected indicators give buy signal, and vice-versa 
    def strategy(self, stop = 9999, take_profit = 9999, *t):
        df = pd.DataFrame(columns = t)
        
        #Converting the parameters to function
        for i in t:
            df[i] = getattr(self, i)()
        
        #Generating signals according to signals that generated from half of the indicators
        result = 0
        x = 0
        df["aggregate"] = 0
        for indicator in t:
            result += df[indicator] 
            df["aggregate"] = result
            x += 1
        df["signal"] = np.where(df["aggregate"] >= x/2, 1, 0)
        df["signal"] = np.where(df["aggregate"] <= -(x/2), -1, df["signal"])        
        
        #Calculating the market return and strategy return
        df["price"] = self.data["Close"].dropna()
        df["market_return"] = np.log(df["price"] / df["price"].shift(1))
        df["strategy_return"] = df["market_return"] * df["signal"].shift(1)
        
        #Rearrange the signals according to stoploss and calculating new returns
        df["account"] = float(0)
        x = 0
        for i in df["signal"].iloc[:-1]:
            if i != 0:
                df["account"].iloc[x+1] = df["strategy_return"].iloc[x+1] + df["account"].iloc[x]                                      
                if df["account"].iloc[x] < -stop:
                    df["signal"].iloc[x] = 0
                    df["strategy_return"].iloc[x+1] = df["market_return"].iloc[x+1] * df["signal"].iloc[x]
                    df["account"].iloc[x+1] = 0
                elif df["account"].iloc[x] > take_profit:
                    df["signal"].iloc[x] = 0
                    df["strategy_return"].iloc[x+1] = df["market_return"].iloc[x+1] * df["signal"].iloc[x]
                    df["account"].iloc[x+1] = 0    
            x += 1
            
        #Cumulative return
        df["sumreturn"] = df["strategy_return"].cumsum()
        return(df)

    
    #Calculate alternative strategy;
    #If we would have buyed at the beginning of term and keep it during the term
    def alternative(self, initial_balance = 10000):   
        df = pd.DataFrame()
        df["price"] = self.data["Close"].dropna()
        number_of_shares = initial_balance / df["price"][0]
        df["cash"] = df["price"] * number_of_shares
        print("If the initial balance was:", initial_balance, "the period-end balance would be:", df["cash"][-1] )
        return(df)
    
    
    
    #Calculate how much money does strategy make
    def simulate_strategy(self, strategy, transaction_amount, leverage_rate = 0): 
        df = pd.DataFrame()
        strat = strategy
        df["number_of_shares"] = transaction_amount / strat["price"]
        number_of_shares = float(0)
        df["cash"] = float(0)
        z = 1
        for i in strat["signal"].iloc[1:]:
            if i == 1:
                if strat["signal"].iloc[z-1] == 1:
                    z += 1
                    continue
                number_of_shares = df["number_of_shares"].iloc[z]
            else:
                if strat["signal"].iloc[z-1] == 1:
                    df["cash"].iloc[z] = number_of_shares * strat["price"].iloc[z]
            if i == -1:
                if strat["signal"].iloc[z-1] == -1:
                    z += 1
                    continue
                number_of_shares = df["number_of_shares"].iloc[z]
            else:
                if strat["signal"].iloc[z-1] == -1:
                    df["cash"].iloc[z] = number_of_shares * strat["price"].iloc[z]
                    df["cash"].iloc[z] = (transaction_amount - df["cash"].iloc[z]) + transaction_amount
            z += 1
        df["total"] = df["cash"].loc[df["cash"] != 0] - transaction_amount
        
        #Calculating leveraged return if leverage included 
        if leverage_rate > 0:
            df["profit"] = df["total"][df["total"] > 0] * leverage_rate
            df["loss"] = df["total"][df["total"] < 0] * leverage_rate
            df["leverage_total"] = (df["profit"].fillna(0) + df["loss"].fillna(0)).cumsum()    
            print("Number of transaction:",  len(strat["signal"][strat["signal"] != 0]))
            print("If we would have traded", transaction_amount, "with leverage rate", leverage_rate, "at every signal, period-end profit and loss status:", df["leverage_total"][-1])
        else:
            #Total profit-loss
            df["total"] = df["total"].fillna(0).cumsum()
            print("Number of transaction:",  len(strat["signal"][strat["signal"] != 0]))
            print("If we would have traded", transaction_amount, "at every signal, period-end profit and loss status:", df["total"][-1])
            df["total2"] = (transaction_amount * strat["strategy_return"]) + transaction_amount
        return(df)


    #Calculate various ratios for evaluating the strategy
    def ratios(self, strategy, prnt = "no"):
        strat = strategy
        ratio = ["Total Market Return", "Average Market Return", "Total Strategy Return", "Average Strategy Return",
                  "Days Number", "Positive Return Days", "Hit Ratio", "CAGR", "Daily Volatility", 
                  "Daily Sharpe", "Max Drawdown", "Calmar Ratio", "Volatility/MaxDrawdown", 
                  "Maximum Daily Return", "Minimum Daily Return", "Ulcer Index", "Martin Ratio",
                  "Sortino Ratio"
                  ] 
        df = pd.DataFrame(index = ratio)
        df[self.ticker] = float()
        
        #Cumulative market return
        df[self.ticker].iloc[0] = strat["market_return"].cumsum()[-1]
        
        #Average market return
        df[self.ticker].iloc[1] = strat["market_return"].mean()
        
        #Total strategy return
        df[self.ticker].iloc[2] = strat["sumreturn"].iloc[-1]
        
        #Average strategy return
        df[self.ticker].iloc[3] = strat["strategy_return"].mean()
        
        df[self.ticker].iloc[4] = len(strat["signal"][strat["signal"] != 0])
        
        df[self.ticker].iloc[5] = len(strat["strategy_return"].loc[strat["strategy_return"] > 0])
        
        #Hit ratio
        df[self.ticker].iloc[6] = df[self.ticker].iloc[5] / df[self.ticker].iloc[4]
        
        #Compound annual growth rate
        days = (strat.index[-1] - strat.index[0]).days
        strat["strategy_equity"] = strat['sumreturn'] + 1
        df[self.ticker].iloc[7] = (((strat["strategy_equity"].iloc[-1] / strat["strategy_equity"].iloc[1])) ** (365.0 / days)) - 1
        
        #Volatility
        df[self.ticker].iloc[8] = (strat['strategy_return'].std()) * sqrt(252)
        
        #Sharpe ratio
        df[self.ticker].iloc[9] = df[self.ticker].iloc[7] / df[self.ticker].iloc[8]
        
        #Maxdrawdown
        def max_drawdown(Y):
            mdd = 0
            peak = Y[1]
            for y in Y:
                if y > peak: 
                    peak = y
                dd = (peak - y) / peak
                if dd > mdd:
                    mdd = dd
            return mdd  
        
        df[self.ticker].iloc[10] = max_drawdown(strat["strategy_equity"])
        
        #Calmar Ratio
        df[self.ticker].iloc[11] = df[self.ticker].iloc[7] / df[self.ticker].iloc[10]
        
        #Volatility / Max Drawdown
        df[self.ticker].iloc[12] = df[self.ticker].iloc[8] / df[self.ticker].iloc[10]
        
        #Maximum return
        df[self.ticker].iloc[13] = np.nanmax(strat["strategy_return"])
        
        #Minimum return

        df[self.ticker].iloc[14] = np.nanmin(strat["strategy_return"])
        
        #Ulcer index
        max14 = strat["price"].rolling(window = 14).max()
        percent_drawdown = ((strat["price"] - max14) / max14) * 100
        percent_drawdown_squared = np.square(percent_drawdown)
        dd_squared_average = percent_drawdown_squared.rolling(window = 14).mean()
        ulcer_index = np.sqrt(dd_squared_average)
        df[self.ticker].iloc[15] = ulcer_index[-1]
        
        #Martin ratio (Ulcer Performance Index)
        martin = df[self.ticker].iloc[7] / ulcer_index
        df[self.ticker].iloc[16] = martin[-1]
        
        #Sortino Ratio
        neg_ret_sqr_avrg = (np.square(strat['strategy_return'][strat['strategy_return'] < 0])).mean()
        squareroot = np.sqrt(neg_ret_sqr_avrg)
        df[self.ticker].iloc[17] = df[self.ticker].iloc[7] / squareroot
        
        #Returns by months
        months = pd.DataFrame(strat["strategy_return"].resample('M').sum())

        #Returns by years
        years = pd.DataFrame(strat["strategy_return"].resample('Y').sum())
        
        if prnt == "yes":
            print("Total market return:", str(round(df[self.ticker].iloc[0] * 100, 4 )) + "%")
            print("Average market return:", str(round(df[self.ticker].iloc[1] * 100, 4 )) + "%")
            print("Total strategy return:", str(round(df[self.ticker].iloc[2] *100, 4 )) + "%")
            print("Average strategy return:", str(round(df[self.ticker].iloc[3] * 100, 4 )) + "%")
            print("Number of days:", df[self.ticker].iloc[4])
            print("Number of days with positive return:", df[self.ticker].iloc[5])
            print("Hit Ratio:", str(round(df[self.ticker].iloc[6] * 100, 2)) + "%")
            print ("CAGR:", str(round(df[self.ticker].iloc[7] * 100, 2)) + "%")
            print ("Annualised volatility using daily data:", str(round(df[self.ticker].iloc[8] * 100, 2)) + "%")
            print ("Daily sharpe:", round(df[self.ticker].iloc[9], 2))
            print ("Max drawdown daily data:", str(round(df[self.ticker].iloc[10] * 100, 2)) + "%")
            print ("Calmar ratio:", round(df[self.ticker].iloc[11], 2))
            print ("Volatility / Max Drawdown:", round(df[self.ticker].iloc[12], 2))
            print("Maximum daily return:", str(round(df[self.ticker].iloc[13] * 100, 2)) + "%")
            print("Minimum daily return:", str(round(df[self.ticker].iloc[14] * 100, 2)) + "%")
            print("Ulcer Index 14 Day:", round(df[self.ticker].iloc[15], 4))
            print("Martin Ratio:", round(df[self.ticker].iloc[16], 4))
            print("Sortino Ratio:", round(df[self.ticker].iloc[17], 4))
            
        return(months, years, df)


    #Plot the graphics
    def graphics(self, strategy):
        df = pd.DataFrame(strategy)
        price = df["price"].plot(title = "Price")
        signals = plt.scatter(df.loc[df.signal == 1].index, df.price[df.signal == 1], c="g", marker = "^")
        signals2 = plt.scatter(df.loc[df.signal == -1].index, df.price[df.signal == -1], c="r", marker = "v")
        
        returnsum = df[["strategy_return", "market_return"]].cumsum().plot(title = "Cumulative Returns")
        
        years = stock.ratios(self, strategy)[1].plot(title="Yearly average returns")
        
        months = stock.ratios(self, strategy)[0].plot(title="Monthly average returns")
        
        plt.figure()
        days = plt.plot(df["strategy_return"])
        plt.title("Daily returns")
        
        fig , axs = plt.subplots(1, sharey = True, tight_layout = True)
        axs.hist(df["strategy_return"], bins = 5)
        plt.title("Daily returns histogram")
        
        return(price, signals , signals2, returnsum, days, months, years)


#Try different stoploss levels between lower and upper bounds
def optimize_stop(lower_bound, upper_bound, step, ticker, *indicators): 
    
    stck = stock(ticker)
    
    #Try only stoploss levels
    if upper_bound > 99:
        bound = np.arange(lower_bound, 0.01, step)
        df = pd.DataFrame(index = bound)
        df[upper_bound] = ""
        z = 0
        for i in bound:   
            strat = stck.strategy(i, upper_bound, *indicators)
            rate_of_return =  strat["strategy_return"].mean()
            df[upper_bound].iloc[z] = rate_of_return
            z += 1
    
    #Try both stoploss and take-profit levels
    else:
        bounds = np.arange(lower_bound, upper_bound, step)
        df = pd.DataFrame(columns = bounds, index = bounds)    
        
        for y in bounds:
            z = 0
            for i in bounds:   
                strat = stck.strategy(i, y, *indicators)
                rate_of_return = strat["strategy_return"].mean()
                df[y].iloc[z] = rate_of_return
                z += 1
    return(df)


#Try all combination of selected indicators on BIST30 stocks
def strategy_finder(start_date, end_date, desired_return, excel = False): 
    
    tickers =["AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "CCOLA.IS", "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
              "FROTO.IS",  "GARAN.IS", "HALKB.IS", "ISCTR.IS", "KCHOL.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS",
              "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SODA.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
              "TKFEN.IS", "TRKCM.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "YKBNK.IS"]
    df = pd.DataFrame(index = range(0, 255))#Upper band must be equal to (number of combination - 1)
                                            #Because null set does not included --> 2^n - 1
                                            
    chosen = pd.DataFrame()  #Dataframe for adding the successful strategies 
    
    
    for ticker in tickers:
        df[ticker] = float(0)
        stck = stock(ticker, start_date, end_date)
        indicators = ["mavg", "rsi", "stochastic", "bollinger_bands", "cci", "macd", "williamsr", "adx"]
        data = pd.DataFrame()
        results = pd.DataFrame()
        names = []
        
        for indicator in indicators:
            data[indicator] = getattr(stck, indicator)()
        
        price = stck.data["Close"].dropna()
        results["market_return"] = np.log(price / price.shift(1))
        
        x = 0
        for z in range(1,9): #Upper band must be equal to number of indicator + 1
            comb = combinations(indicators, z)
            

            for sub in comb:
                if ticker == tickers[-1]:
                    names.append(sub)
                aggregate = 0
                for ind in sub:
                    aggregate += data[ind]
                results[sub] = np.where(aggregate >= len(sub) / 2, 1, 0)
                results[sub] = np.where(aggregate <= -(len(sub) / 2), -1, results[sub])
                
                results["strategy_return", sub] = results["market_return"] * results[sub].shift(1)
                average_strategy = results["strategy_return",sub].mean()
                df[ticker][x] = average_strategy
                
                
                if average_strategy > desired_return:
                    chosen = pd.concat([stck.ratios(stck.strategy(99, 99, *sub))[2], chosen], axis = 1)
                    chosen.rename(columns={ticker: ticker + str(sub)}, inplace = True)
                x += 1    
    df.index = names
    
    #Saving the results to Excel file (Sheet1: Returns, Sheet2: Ratios)
    if excel == True:
        writer = pd.ExcelWriter('strategyfinder.xlsx', engine = 'xlsxwriter')
        df.to_excel(writer, "Returns")
        
        chosen.to_excel(writer, "Ratios")
        writer.save()
        
    return(df, chosen)
    
