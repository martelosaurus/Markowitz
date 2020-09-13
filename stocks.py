import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cleaning
import portfolio
import twostock
import beta_plots

font = {'family' : 'normal', 'size' : 12}

# parameters
digits = 4

# read CRSP data
D = pd.read_csv('dow30_data.csv')

# renames
D = D.rename(columns={'TICKER' : 'ticker'})

# clean-up
D = D[D.ticker!='C']                        # C not in the DIJA
D = D[D.RET!='C']							# V first obs
D.loc[D.ticker=='WAG','ticker'] = 'WBA' 	# WBA ticker change

# sort
D.date = D.date.map(lambda d: int(str(d)[0:6]))

# names of the stocks
N = pd.read_csv('dow30_name.csv')

# load Fama-French data
M = pd.read_csv('famafrench.csv')

# merge
D = D.merge(N) 
D = D.merge(M)

# change date format
D.date = pd.to_datetime(D.date,format='%Y%m')
M.date = pd.to_datetime(M.date,format='%Y%m')

D['ret'] = np.double(D.RET)
D['rf'] = D.rf

# round 
D = cleaning.round_columns(D) 

# build portfolio
portfolio1 = twostock.TwoStocks(D,'AAPL','PG',n=1000)
portfolio2 = twostock.TwoStocks(D,'CVX','XOM',n=1000)
portfolio3 = twostock.TwoStocks(D,'AAPL','MSFT',n=1000)
portfolio4 = twostock.TwoStocks(D,'WBA','WMT',n=1000)
#portfolio2 = portfolio.Portfolio(D,['AAPL','PG'],.5])
x = portfolio4.summary()
z = pd.DataFrame({'stat' : list(x.keys()), 'val' : list(x.values())})
z.to_csv('exam2_stats.csv',index=False)
