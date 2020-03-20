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
print(portfolio2.summary())
#portfolio2 = portfolio.Portfolio(D,['AAPL','PG'],.5])

# load wide 
W = pd.read_csv('canvas_wide.csv')

# betas
stats, beta, sharpe = beta_plots.beta_plots(D.loc[D['ticker']=='CAT',['mkt','ret','rf']])

# histogram
if False:
	X = W.corr().to_numpy()
	X = np.triu(X,1)
	X = X[X.nonzero()]
	plt.hist(X,density=True,rwidth=.9)
	plt.xlim([0.,1.])
	plt.xticks(**font)
	plt.yticks(**font)
	plt.xlabel('Correlation',**font)
	plt.ylabel('Frequency',**font)
	plt.title('Correlation Between DIJA 30 Stocks',**font)
	plt.show()
	print(X.mean())

# value of diversification
if False:
	X = W.cov().to_numpy()
	portvar = []
	N = 30
	for n in range(1,N+1):
		x = np.hstack([np.ones(n)/n,np.zeros(N-n)])
		portvar.append(100.*np.sqrt(X.dot(x).dot(x)))
	portvar = np.array(portvar)
	plt.plot(np.arange(1,N+1),portvar,linewidth=4)
	plt.xticks(**font)
	plt.yticks(**font)
	plt.xlabel('Number of Stocks',**font)
	plt.ylabel('Portfolio Standard Deviation (%)',**font)
	plt.title('Equally-Weighted Portfolio Risk (Adding Alphabetically)',**font)
	plt.show()
