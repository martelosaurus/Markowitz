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

# load wide 
W = pd.read_csv('canvas_wide.csv')

# betas
def betafun(ticker):
    stats, beta, sharpe = beta_plots.beta_plots(D.loc[D['ticker']==ticker,['mkt','ret','rf']])
    print(D.loc[D['ticker']==ticker,['date','mkt','ret','rf']])
    return stats, beta, sharpe

def beta_plots(df):
	cov = df.cov()
	corr = df.corr()
	std = df.std()
	avg = df.mean()
	stats = {
		'rf_rtrn' : avg['rf'],
		'mkt_rtrn' : avg['mkt'],
		'ret_rtrn' : avg['ret'],
		'mkt_risk' : std['mkt'],
		'ret_risk' : std['ret'],
		'ret_mkt_cov' : cov.loc['ret','mkt'],
		'ret_mkt_corr' : corr.loc['ret','mkt'],
	} 
	beta = {
		'beta' : stats['ret_mkt_cov']/stats['mkt_risk']**2.
	}
	perf = {
		'mkt_sharpe' : (stats['mkt_rtrn']-stats['rf_rtrn'])/stats['mkt_risk'])
		'ret_sharpe' : (stats['ret_rtrn']-stats['rf_rtrn'])/stats['ret_risk'])
		'ret_prem' : 100.*(stats['ret_rtrn']-stats['rf_rtrn']))
		'mkt_prem' : 100.*(stats['mkt_rtrn']-stats['rf_rtrn'])
	}
	stats = {	
		'rf_rtrn' : 100.*avg['rf'])
		'mkt_rtrn' : 100.*avg['mkt'])
		'ret_rtrn' : 100.*avg['ret'])
		'mkt_risk' : 100.*std['mkt'])
		'ret_risk' : 100.*std['ret'])
		'ret_mkt_corr' : corr.loc['ret','mkt'])
	}
	beta['alpha'] = perf['ret_prem']-beta['beta']*perf['mkt_prem']

	return stats, beta, perf
	
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
# this should go to another module
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
