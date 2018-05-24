# ML_Stock_Trading_Algos
Some Machine Learning algos from scratch for stock trading over historical stock data. 

## Reports
report.pdf explains machine learning trading strategy and performance over training and test data vs a benchmark and manual rules based strategy

Reports folder also has previous reports on manual trading strategy as well as analysis of performance of random forests and decision trees.

## Algorithms

DTLearner.py = Decision Tree Implemented from scratch

RTLearner.py = Random Tree implemented from scratch

BagLearner.py = Random Forest built using many bagged Random Trees (inspired by Adele Cutler's seminal paper)

QLearner.py = Q Learning Agent Built using Reinforcment Learning. Trades over historical data.

## Trading Strategies
manualStrategy.py = a manual strategy built using manual technical indicators (no machine learning)

strategyLearner.py = a Machine Learning based trading strategy using Random Forests based on Random Trees and Bagging.

indicators.py = class that has the functions for the different stock indicators feature engineered for prediction purposes. Inidicators used include:

  1) Price / Simple Moving Average
  2) Momentum
  3) Volatility
  4) MACD (including cross overs of MACD and Signal Line)
  5) Bollinger Bands (including Bollinger Bands % and Bollinger Band Cross Overs)

## Market Simulator
marketsimcode.py = market simulator class that allows us to buy and sell stocks and tracks performance of our portfolio

