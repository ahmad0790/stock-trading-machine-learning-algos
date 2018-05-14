#NAME: AHMAD KHAN
#GT ID: akhan361

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math
import matplotlib.pyplot as plt


def price_over_sma(price, n):
	sma = price.rolling(window=n).mean()
	price_sma = price/sma - 1
	return price_sma, sma, price


def momentum(price, n):
	mom = price/price.shift(n) - 1
	return mom


def bollingerBand(price, n):

	##bollinger bands
	std_dev = (price.rolling(window = n, center=False).std())
	bb_upper = price.rolling(window=n, center=False).mean() + std_dev*2
	bb_lower = price.rolling(window=n, center=False).mean() - std_dev*2
	bb_middle = price.rolling(window=n, center=False).mean()
	#bb_indicator = (price - price.rolling(window=n, center=False).mean())/(std_dev*2)

	##bb %
	bb_indicator = (price - bb_lower)/(bb_upper - bb_lower)
	
	#crossovers
	bb_upper_cross = pd.DataFrame(0, index=price.index, columns=price.columns)
	bb_upper_cross[(price - bb_upper) < 0] = 1
	bb_upper_cross[1:] = bb_upper_cross.diff()
	bb_upper_cross.ix[0] = 0

	bb_lower_cross = pd.DataFrame(0, index=price.index, columns=price.columns)
	bb_lower_cross[(price - bb_lower) > 0] = 1
	bb_lower_cross[1:] = bb_lower_cross.diff()
	bb_lower_cross.ix[0] = 0

	return bb_indicator, bb_upper, bb_lower, bb_middle, bb_upper_cross, bb_lower_cross

def ema(price, n):
	return price.ewm(span=n).mean()

def macd(price, macd_lower, macd_upper, signal):
	ema_upper = ema(price, macd_upper)
	ema_lower = ema(price, macd_lower)
	macd_line = ema_lower - ema_upper
	signal_line = ema(macd_line, signal)
	macd_histogram = macd_line - signal_line
	return macd_line, signal_line, macd_histogram, ema_upper, ema_lower

def volatility(price, n):	
	price = ((price/price.shift(1)) - 1) 
	price.ix[0] = 0
	sddr = price.rolling(window = n, center=False).std()
	return sddr

def get_ema_ind(price, n):
	return ema(price,n)

def get_price_sma_ind(price, n):
	return price_over_sma(price,n)[0]

def getMomentumInd(price, n):
	return momentum(price,n)

def getBBInd(price, n):
	return bollingerBand(price,n)[0]

def getBBUpperCross(price, n):
	return bollingerBand(price,n)[4]

def getBBLowerCross(price, n):
	return bollingerBand(price,n)[5]

def getMACDInd(price):
	return macd(price,12,26,9)[0]

def getMACDHistogramInd(price):
	return macd(price,12,26,9)[2]

if __name__ == "__main__":
	symbols = ['JPM']
	#start_date = '2009-12-01'
	#end_date = '2011-12-31'
	start_date = '2007-12-01'
	end_date = '2009-12-31'

	dates = pd.date_range(start_date, end_date)

	price = get_data(symbols, dates, addSPY = True)
	price = price.drop(['SPY'],axis=1)

	#Momentum Chart
	#momentum_data = momentum(price, 7)
	#momentum_data.columns = ['Momentum']

	##PRICE/SMA % CHART
	def generatePriceChart(price, n):
		psma, sma, price = price_over_sma(price, n)
		price=pd.merge(price, sma, left_index=True,right_index=True)
		price = price.ix[20:]
		price = price/price.ix[0]
		price=pd.merge(price, psma.ix[20:], left_index=True,right_index=True)
		price.columns = ['Normalized Price', 'Normalized SMA', 'Price/SMA %']
		plot_data(price, title="Price/SMA % Over Time", xlabel="Date", ylabel="Price")

	##BOLLINGER BAND CHART
	def generateBollingerPriceChart(price, n):
		bb_indicator, bb_upper, bb_lower, bb_middle, bb_upper_cross, bb_lower_cross = bollingerBand(price, n)
		#price = price/price[0]
		price=pd.merge(price, bb_upper, left_index=True,right_index=True)
		price=pd.merge(price, bb_lower, left_index=True,right_index=True)
		price=pd.merge(price, bb_middle, left_index=True,right_index=True)
		price = price.ix[20:]
		bb_indicator = bb_indicator.ix[20:]
		price.columns = ['JPM Price','BB Upper Band', 'BB Lower Band', '20 Day SMA']
		plot_data(price, title="Bollinger Bands Over Time", xlabel="Date", ylabel="Price")
		plot_data(bb_indicator, title="Bollinger Band % Over Time", xlabel="Date", ylabel="Bollinger Band %")
		#plot_data_multiple(price, bb_indicator, title="Bollinger Band % Over Time", xlabel="Date", ylabel="Price", ylabel2= "Bollinger Band %")


	def generateMACDChart(price, a,b,c):
		#emas with price
		macd_line, signal_line, macd_histogram, ema_upper, ema_lower = macd(price, a,b,c)
		macd_data =pd.merge(price, ema_upper, left_index=True,right_index=True)
		macd_data =pd.merge(macd_data, ema_lower, left_index=True,right_index=True)
		macd_data = macd_data.ix[20:]
		macd_data = macd_data/macd_data.ix[0]
		macd_data.columns = ['Price Normalized','26 Day EMA Normalized Price', '12 Day EMA Normalized Price']

		#macd indicators
		macd_line=pd.merge(macd_line, signal_line, left_index=True,right_index=True)
		#macd_line=pd.merge(macd_line, macd_histogram, left_index=True,right_index=True)
		macd_line.columns = ['Signal Line', 'MACD Line']
		plot_data(macd_data, title="EMA Prices Over Time", xlabel="Date", ylabel="Price")
		plot_data(macd_line, title="MACD and Signal Line Over Time", xlabel="Date", ylabel="MACD or Signal Value")
		#plot_data_multiple(macd_data, macd_line, title="EMA Prices Over Time", xlabel="Date", ylabel="Price", ylabel2= "MACD or Signal Value")

	def generateVolatilityChart(price, n):
		psma, sma, price = price_over_sma(price, n)
		vol = volatility(price,n)
		price = pd.merge(price, vol, left_index=True,right_index=True)
		price.columns = ['Price', 'Volatility']
		plot_data(vol, title= "Volatility Over Time", xlabel="Date", ylabel="Volatility")



if __name__=="__main__":

	generatePriceChart(price,20)
	generateBollingerPriceChart(price, 20)
	generateMACDChart(price,12,26,9)
	generateVolatilityChart(price,14)

	#leaving out momentum
	#plot_data(momentum_data, title= "Momentum Over Time", xlabel="Date", ylabel="Momentum")



