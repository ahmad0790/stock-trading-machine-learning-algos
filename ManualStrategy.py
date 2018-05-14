##Name: Ahmad Khan
#GT ID: akhan361

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math
import marketsimcode as msim
import indicators as ind

class manualStrategy(object):

	def __init__(self, symbol = 'AAPL', sd=dt.datetime(2008,1,01), ed=dt.datetime(2009,12,31), sv = 100000):
		self.sv = sv
		self.symbol = []
		self.symbol.append(symbol)
		self.start_date = sd - dt.timedelta(days=30)
		self.end_date = ed

	def testPolicy(self):
		dates = pd.date_range(self.start_date, self.end_date)
		price = get_data(self.symbol, dates, addSPY = True)
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


		#creaete tradings strategy
		df_trades = pd.DataFrame(0, index = price.index, columns=['Shares'])
		overbought_or_oversold = pd.DataFrame(index = price.index, columns=['Action'])

		##entry conditions -different combinations
		
		#macd + price/sma + bb % + volatility = final trading strategy
		short_ind = (macd_cross_below_signal==1) | ((sma_ind > 0.05) & (bb_ind > 1))
		buy_ind = (macd_cross_above_signal==1) | ((sma_ind< -0.05) & (bb_ind < 0))
		exit_ind = (price_sma_crossover==1) & ((vol_ind > 0.04))


		#only overbought/oversold (price/sma + bb%)
		#short_ind = ((sma_ind>0.05) & (bb_ind > 1))
		#buy_ind =  ((sma_ind< -0.05) & (bb_ind < 0))

		#either crossovers (MACD or BB)
		#buy_ind = (bb_lower_cross_signal==1) | (macd_cross_above_signal==1)
		#short_ind = (bb_upper_cross_signal==1) | (macd_cross_below_signal==1)


		#both crossovers (MACD and BB)
		#buy_ind = (bb_lower_cross_signal==1) & (macd_cross_above_signal==1)
		#short_ind = (bb_upper_cross_signal==1) & (macd_cross_below_signal==1)

		#only bollinger crossover
		#buy_ind = (bb_lower_cross_signal==1) 
		#short_ind = (bb_upper_cross_signal==1)

		#only MACD signal line crossover
		#short_ind = (macd_cross_below_signal==1)
		#buy_ind = (macd_cross_above_signal==1)


		##exit conditions
		overbought_or_oversold.iloc[exit_ind == True] = 0 #exit
		#overbought
		overbought_or_oversold.iloc[short_ind==True]  = -1 #short
		#oversold
		overbought_or_oversold.iloc[buy_ind==True] = 1 #buy

		#buy or sell on above if holding constraint is met 
		holdings = 0
		for index_order,row_order in overbought_or_oversold.iterrows():

			if row_order['Action'] == 1 and holdings == 0:
				df_trades.ix[index_order, 'Shares'] = 1000
				holdings = 1000	

			elif row_order['Action'] == 1 and holdings == -1000:
				df_trades.ix[index_order,'Shares'] = 2000
				holdings = 1000

			elif row_order['Action'] == -1 and holdings == 0:
				df_trades.ix[index_order, 'Shares'] = -1000
				holdings = -1000

			elif row_order['Action'] == -1 and holdings == 1000:
				df_trades.ix[index_order,'Shares'] = -2000
				holdings = -1000

			elif row_order['Action'] == 0 and holdings == 1000:
				df_trades.ix[index_order,'Shares'] = -1000
				holdings = 0

			elif row_order['Action'] == 0 and holdings == -1000:
				df_trades.ix[index_order,'Shares'] = 1000
				holdings = 0

			else:
				df_trades.ix[index_order, 'Shares'] = 0
		
		return df_trades

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


	def generatePerformanceChart(df_trades, strategy, benchmark):
		portfolios = np.column_stack((strategy, benchmark))
		portfolios = pd.DataFrame(portfolios, columns =['Manual Strategy', 'Benchmark'], index = strategy.index)
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

		ax.plot(portfolios['Manual Strategy'], color = 'k')
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
		ms = manualStrategy(symbol = 'JPM', sd=start_date, ed=end_date, sv = 100000)
		df_trades = ms.testPolicy()
		df_orders = ms.createOrdersDataFrame(df_trades)
		benchmark = ms.createBenchMarkDataFrame(df_trades, sd=start_date)
		portvals = msim.compute_portvals(orders_file = df_orders, start_val = 100000, commission=9.95, impact=0.005, end_date = end_date, start_date = start_date)
		portvals_benchmark = msim.compute_portvals(orders_file = benchmark, start_val = 100000, commission=0, impact=0, end_date = end_date, start_date = start_date)
		print('Statistics for Manual Strategy')
		msim.computeStatistics(portvals)
		print('')
		print('Statistics for Benchmark Strategy')
		msim.computeStatistics(portvals_benchmark)
		generatePerformanceChart(df_trades, portvals, portvals_benchmark)

	print('')
	#run the trading strategy on the training developmentperiod
	print('RUNNING THE STRATEGY FOR THE IN SAMPLE DEVELOPMENT PERIOD')
	run_strategy(dt.datetime(2008,1,1), dt.datetime(2009,12,31))

	#run the trading strategy on the testing out of sample period
	print('RUNNING THE STRATEGY FOR THE OUT OF SAMPLE TESTING PERIOD')
	run_strategy(dt.datetime(2010,1,1), dt.datetime(2011,12,31))








