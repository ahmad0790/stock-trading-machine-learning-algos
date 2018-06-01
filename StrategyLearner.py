"""
Name: Ahmad Khan
GT ID: akhan361
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import indicators as ind
import numpy as np
from scipy import stats
import marketsimcode as msim
import BagLearner as bl
import RTLearner as rt
import DTLearner as dl
import ManualStrategy as ms

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.qLearner = False

    # this method should create a RandomForest, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        # add your code to do learning here
        self.start_date = sd - dt.timedelta(days=30)
        self.end_date = ed
        self.sv = sv
        self.symbol = []
        self.symbol.append(symbol)

        price = self.createIndicators()
        #print price.ix[:, 1:(price.shape[1]-1)]

        trainX = np.array(price.ix[:, 1:(price.shape[1]-1)])
        trainY = np.array(price.ix[:,(price.shape[1]-1)])

        leaf = 5
        bag = 25
        learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":leaf}, bags = bag, boost = False, verbose = False) # constructor
        #learner = dl.DTLearner(leaf_size = 5, verbose = False)
        learner.addEvidence(trainX, trainY)

        self.learner = learner
        ###implement strategy
            

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        self.start_date = sd - dt.timedelta(days=30)
        self.end_date = ed
        self.sv = sv
        self.symbol = []
        self.symbol.append(symbol)
        
        price = self.createIndicators()
        
        testX = np.array(price.ix[:,1:(price.shape[1]-1)])
        Y_pred = self.learner.query(testX)

        predicted_actions = pd.DataFrame(0, index = price.index, columns=['Action'], dtype = int)
        predicted_actions['Action'] =Y_pred.astype(int)

        holdings = 0
        portval = sv
        cash = sv
        commission = 0

        df_trades = pd.DataFrame(0, index = predicted_actions.index, columns=['Shares'])    

        for index_predict,row_predict in predicted_actions.iterrows():            
            if row_predict['Action'] == 1 and holdings == 0:
                df_trades.ix[index_predict, 'Shares'] = 1000
                holdings = 1000

            elif row_predict['Action'] == -1  and holdings == 0:
                df_trades.ix[index_predict, 'Shares'] = -1000
                holdings = -1000


            elif row_predict['Action'] == -1 and holdings == 1000:
                df_trades.ix[index_predict, 'Shares'] = -2000
                holdings = -1000
            
            elif row_predict['Action'] == 1 and holdings == -1000:
                df_trades.ix[index_predict, 'Shares'] = 2000
                holdings = 1000

            else:
                df_trades.ix[index_predict, 'Shares'] = 0

        return df_trades

    def createIndicators(self):
         ##mycode
        dates = pd.date_range(self.start_date, self.end_date)
        price = ut.get_data(self.symbol, dates, addSPY = True)
        price_SPY = price['SPY']
        price = price.drop(['SPY'], axis=1)

        ##get data for all the indicators including T - 30 trading days of data (to account for creation of historical indicators)
        sma_ind = ind.get_price_sma_ind(price, 20)
        momentum_ind = ind.getMomentumInd(price, 14)
        bb_ind = ind.getBBInd(price, 14)
        macd_ind = ind.getMACDInd(price)
        vol_ind = ind.volatility(price, 14)

        #we now remove the last 30 days of data (December 2007) after creating the indicators so that we only have training period data.
        price = price.loc[price.index >= self.start_date + dt.timedelta(days=30)]
        sma_ind = sma_ind.loc[sma_ind.index >= self.start_date + dt.timedelta(days=30)]
        vol_ind = vol_ind.loc[vol_ind.index >= self.start_date + dt.timedelta(days=30)]
        momentum_ind = momentum_ind.loc[momentum_ind.index >= self.start_date + dt.timedelta(days=30)]
        bb_ind = bb_ind.loc[bb_ind.index >= self.start_date + dt.timedelta(days=30)]
        macd_ind = macd_ind.loc[macd_ind.index >= self.start_date + dt.timedelta(days=30)]

        
        ##create cross over signals for each day
        price_sma_crossover = pd.DataFrame(0, index=sma_ind.index, columns=sma_ind.columns)
        price_sma_crossover[sma_ind > 0] = 1
        price_sma_crossover = price_sma_crossover.diff()
        price_sma_crossover[price_sma_crossover != 0] = 1

        macd_sigal_diff = ind.getMACDHistogramInd(price)    
        
        #macd cross below signal = sell
        macd_cross_below_signal = pd.DataFrame(0, index=macd_ind.index, columns=macd_ind.columns)
        macd_cross_below_signal[macd_sigal_diff < 0] = 1
        macd_cross_below_signal[1:] = macd_cross_below_signal.diff()
        macd_cross_below_signal.ix[0] = 0
        #print(macd_cross_above_signal)

        #macd cross above signal = buy
        macd_cross_above_signal = pd.DataFrame(0, index=macd_ind.index, columns=macd_ind.columns)
        macd_cross_above_signal[macd_sigal_diff > 0] = 1
        macd_cross_above_signal[1:] = macd_cross_above_signal.diff()
        macd_cross_above_signal.ix[0] = 0
        #print(macd_cross_above_signal)

        #bollinger crossovers
        ##this is a sell signal
        bb_upper_cross_signal = ind.getBBUpperCross(price,20)

        #this is a buy signal
        bb_lower_cross_signal = ind.getBBLowerCross(price,20)

        ##create and discretize states for Q Learner
        #print sma_ind
        daily_rets = ((price.shift(-5)/price) - 1)
        daily_rets.ix[-1] = 0
        

        price['Price_Sma'] = sma_ind
        price['Volatility'] = vol_ind
        price['Momentum'] = momentum_ind
        price['BB_Ind'] = bb_ind
        #price['MACD_Ind'] = macd_ind

        #price['BB_Upper_Cross'] = bb_upper_cross_signal
        #price['BB_Lower_Cross'] = bb_lower_cross_signal
        price['MACD_Cross_Below'] = macd_cross_below_signal
        price['MACD_Cross_Above'] = macd_cross_above_signal
        #price['Price_SMA_Crossover'] = price_sma_crossover

        daily_ret_classes = pd.DataFrame(0, index=daily_rets.index, columns=daily_rets.columns, dtype = int)
        Y_buy = 0.005
        Y_sell = -0.005
        daily_ret_classes[daily_rets > Y_buy] = 1
        daily_ret_classes[daily_rets < Y_sell] = -1

        price['Action'] = daily_ret_classes
        #print price.iloc[:,1:].sum(axis=1)
        #print price
        return price
        

        #convert df_trades to ordesr dataframe
    def createOrdersDataFrame(self, df_trades):
        df_orders = pd.DataFrame(columns = ['Date','Symbol','Order','Shares'])

        for day in df_trades.index:
            if df_trades.ix[day,'Shares'] > 0:
                df_orders = df_orders.append({'Date': day, 'Symbol': self.symbol[0], 'Order': 'BUY', 'Shares': abs(df_trades.ix[day,'Shares'])}, ignore_index=True)
    
            elif df_trades.ix[day,'Shares'] < 0:
                df_orders = df_orders.append({'Date': day, 'Symbol': self.symbol[0], 'Order': 'SELL', 'Shares': abs(df_trades.ix[day,'Shares'])}, ignore_index=True)

        df_orders = df_orders.set_index('Date')
        return df_orders


        #benchmark
    def createBenchMarkDataFrame(self, df_trades, sd):
        sd = sd + dt.timedelta(days=1)
        day_of_week  = sd.weekday()

        if(day_of_week == 5):
            sd = sd + dt.timedelta(days=2)
        if(day_of_week == 6):
            sd = sd + dt.timedelta(days=1)

        benchmark = pd.DataFrame(columns = ['Date','Symbol','Order','Shares'])
        benchmark = benchmark.append({'Date': sd, 'Symbol': self.symbol[0], 'Order': 'BUY', 'Shares':1000},ignore_index=True)
        benchmark = benchmark.set_index('Date')
        return benchmark


if __name__=="__main__":
    print "One does not simply think up a strategy"

    def generatePerformanceChart(df_trades, strategy, benchmark):
        portfolios = np.column_stack((strategy, benchmark))
        portfolios = pd.DataFrame(portfolios, columns =['Random Forest Strategy', 'Benchmark'], index = strategy.index)
        portfolios = portfolios/portfolios.ix[0]
        portfolios = pd.merge(portfolios, df_trades, left_index = True, right_index = True)
        portfolios['Holdings'] = portfolios['Shares'].cumsum()
        portfolios['Long Entry Point'] = (portfolios['Shares'] > 0) & (portfolios['Holdings'] == 1000)
        portfolios['Short Entry Point'] = (portfolios['Shares'] < 0) & (portfolios['Holdings'] == -1000)

        long_entry_dates = portfolios[portfolios['Long Entry Point']==True].index
        short_entry_dates = portfolios[portfolios['Short Entry Point']==True].index

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for d in long_entry_dates:
            ax.axvline(d, color = 'g', lw=0.5)

        for d in short_entry_dates:
            ax.axvline(d, color = 'r', lw=0.5)

        ax.plot(portfolios['Random Forest Strategy'], color = 'k')
        ax.plot(portfolios['Benchmark'], color = 'b')
        #ax.plot(portfolios['Total'], color = 'g')
        #ax.plot(portfolios['Shares'], color = 'g')

        ax.set_xlabel("Date", fontsize =12)
        ax.set_ylabel("Portfolio Value")
        ax.set_title("JPM Stock Portfolio Value Over Time")
        ax.legend(loc='best')
        
        plt.xlim(portfolios.index[0], portfolios.index[-1])
        plt.show()


    def run_strategy(start_date, end_date):
        print('')
        #ms = manualStrategy(symbol = 'JPM', sd=start_date, ed=end_date, sv = 100000)
        np.random.seed(106347)
        learner = StrategyLearner()
        learner.addEvidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
        df_trades = learner.testPolicy(symbol = "JPM", sd=start_date, ed=end_date, sv = 100000)
        #df_trades = ms.testPolicy()
        
        df_orders = learner.createOrdersDataFrame(df_trades)
        benchmark = learner.createBenchMarkDataFrame(df_trades, sd=start_date)

        portvals = msim.compute_portvals(orders_file = df_orders, start_val = 100000, commission=0, impact=0.005, end_date = end_date, start_date = start_date)
        portvals_benchmark = msim.compute_portvals(orders_file = benchmark, start_val = 100000, commission=0, impact=0, end_date = end_date, start_date = start_date)
        print 'Statistics for Random Forest Strategy' 
        msim.computeStatistics(portvals)
        print '' 
        print 'Statistics for Benchmark Strategy'
        msim.computeStatistics(portvals_benchmark)
        generatePerformanceChart(df_trades, portvals, portvals_benchmark)

    print ''
    #run the trading strategy on the training developmentperiod
    print 'RUNNING THE Random Forest STRATEGY FOR THE IN SAMPLE DEVELOPMENT PERIOD'
    run_strategy(dt.datetime(2008,1,1), dt.datetime(2009,12,31))

    #run the trading strategy on the testing out of sample period
    print 'RUNNING THE Random Forest STRATEGY FOR THE OUT OF SAMPLE TESTING PERIOD'
    run_strategy(dt.datetime(2010,1,1), dt.datetime(2011,12,31))


   
