# Trading Strategy Finder
&nbsp;&nbsp;&nbsp;&nbsp;  This is my bachelor's degree graduation project. In this work, I created a framework for both analyzing and finding a trading strategy. 
It can be used by first defining a class and picking a variety of methods optionally. Methods contain indicators, backtests, graphs, etc. The usage may found in 
Implementation.py file. Just in case, the same file is available [**here**](https://nbviewer.jupyter.org/github/akgunburak/Trading_Strategy_Finder/blob/master/Implementation.ipynb)

&nbsp;
&nbsp;
&nbsp;

## Requirements
* **Language and version:** Python - 3.7
* **Packages:** pandas, yfinance, numpy, matplotlib.pyplot, datetime, math, itertools

&nbsp;
&nbsp;
&nbsp;

## Improvable Aspects
* Indicators can be calculated by using a library (e.g Ta-Lib)
* The main strategy is based on the decision of the absolute majority of indicators. It can be improved by assigning different weights to each of them.
* The number of backtest ratios or graphs can be increased.
* The current version of strategy may lead to a variety of biases (e.g survivorship bias) since it picks stocks without testing. Incorporating cross-validation into the algorithm may be useful.
