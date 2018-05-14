"""MC2-P1: Market simulator.
Name: Ahmad Khan
GT ID: akhan361
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math
import marketsimcode as msim

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005, end_date = dt.datetime(2009,12,31)
    , start_date = dt.datetime(2008,01,02)):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio

    orders = orders_file
    symbols = np.ndarray.tolist(orders.Symbol.unique())

    #start_date = orders.index.min()
    #end_date = orders.index.max()
    dates = pd.date_range(start_date, end_date)

    prices = get_data(symbols, dates, addSPY = True)
    prices = prices[symbols]  # only portfolio symbols

    prices.fillna(method='ffill',inplace=True)
    prices.fillna(method='bfill', inplace=True)
    prices['Cash'] = start_val
    #print prices

    trades = pd.DataFrame(0, index = prices.index, columns = symbols)
    trades['Cash'] = 0

    holdings = pd.DataFrame(0, index = prices.index, columns = symbols)

    values = pd.DataFrame(0, index = prices.index, columns = symbols)
    
    for index_order,row_order in orders.iterrows():
    	for symbol in symbols:
    		if symbol == row_order['Symbol'] and row_order['Order'] == 'BUY':
    			trades.loc[index_order, symbol] = row_order['Shares'] + trades.loc[index_order, symbol]
    			trades.loc[index_order, 'Cash'] = trades.loc[index_order, 'Cash'] - 1 * (row_order['Shares'] * prices.loc[index_order,symbol]*(1+impact)) - commission
    	
    		if symbol == row_order['Symbol'] and row_order['Order'] == 'SELL':
    			trades.loc[index_order, symbol] = -1* row_order['Shares'] + trades.loc[index_order, symbol]
    			trades.loc[index_order, 'Cash'] = trades.loc[index_order, 'Cash'] + (row_order['Shares'] * prices.loc[index_order,symbol]*(1-impact)) - commission

    #print trades

    for sym in holdings:
    	holdings[sym] =  trades[sym].cumsum()

    holdings['Cash'] = trades['Cash'].cumsum()
    holdings['Cash'] = start_val + holdings['Cash']
    #sprint holdings

    values = holdings*prices
    values['Cash'] = holdings['Cash']
    #print values

    port_val = values.sum(axis=1)
    return port_val

def computeStatistics(port_val):
    rfr=0.0
    sf=252.0

    daily_return = ((port_val/port_val.shift(1)) - 1)
    daily_return[0] = 0

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_return[1:].mean()
    sddr = daily_return[1:].std()

    daily_rf = math.pow(1 +rfr, 1/sf) - 1
    sr = np.sqrt(sf) * ((daily_return[1:] - daily_rf).mean()/(daily_return[1:].std()))

    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [cr,adr,sddr,sr]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Final Portfolio Value: {}".format(port_val[-1])

def computeStatisticsReturnValues(port_val):
    rfr=0.0
    sf=252.0

    daily_return = ((port_val/port_val.shift(1)) - 1)
    daily_return[0] = 0

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_return[1:].mean()
    sddr = daily_return[1:].std()

    daily_rf = math.pow(1 +rfr, 1/sf) - 1
    sr = np.sqrt(sf) * ((daily_return[1:] - daily_rf).mean()/(daily_return[1:].std()))

    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [cr,adr,sddr,sr]
    return cum_ret, sharpe_ratio, std_daily_ret, avg_daily_ret


def author():
    return 'akhan361' # replace tb34 with your Georgia Tech username.
    
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    #of = "./orders/orders-02.csv"
    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv, commission=9.95,impact=0.005)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    computeStatistics(porvals)


if __name__ == "__main__":
    test_code()
    #print('test')
