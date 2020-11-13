# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve
from alpha_vantage.timeseries import TimeSeries

font = {'family' : 'normal', 'size' : 12}

class Portfolio:

	def __init__(self,key,tickers,rF=0.,hor=[1,5]):
		"""
		Initializes Portfolio object

		Parameters
		----------

		x : array like object
			Weights
		tickers : list
			A list of tickers
		dow : boolean
			If True, loads tickers from the DIJA
		rF : float
			Risk-free rate

		Examples
		--------
		"""	 

		# rates
		self.rF = rF
		self.hor = hor

		# load tickers and add SPY
		self.tickers = tickers
		tickers.append('SPY')
		tickers.sort()

		# load data
		ts = TimeSeries(key, output_format = "pandas")
		X = None

		# loop over tickers
		for ticker in tickers:

			# load stock data from Alpha Vantage
			tick_dat, _ = ts.get_monthly_adjusted(symbol=ticker)

			# old and new columns
			old_cols = ['5. adjusted close','7. dividend amount']
			new_cols = [ticker + '_PRC', ticker + '_DIV']

			# select tick data
			tick_dat = tick_dat[old_cols]
			col_dict = dict(zip(old_cols,new_cols))
			tick_dat = tick_dat.rename(columns=col_dict)

			# reformat date
			tick_dat.index = 100*tick_dat.index.year+tick_dat.index.month

			# meger to X list
			opts = {'how' : 'outer', 'left_index' : True, 'right_index' : True}
			if X is None:
				X = tick_dat.iloc[::-1]
			else:
				X = X.merge(tick_dat,**opts)

		# drop
		X.to_csv('_'.join(tickers) + '.csv')		

		# compute returns 
		for t in tickers:
			X[t+'_RET'] = (X[t+'_PRC']+X[t+'_DIV'])/X[t+'_PRC'].shift()-1.
			X[t+'_DY'] = X[t+'_DIV']/X[t+'_PRC'].shift()
			X[t+'_CG'] = X[t+'_PRC']/X[t+'_PRC'].shift()-1.

		# kill the first row (with the NAs)
		X = X.loc[X.index[1:],]

		# store data frame
		self.X = X

		# column names
		self.columns = [ticker + '_RET' for ticker in self.tickers]

	def summary(self,h=60):
		"""
		Parameters
		----------
		m : int
			Month
		h : int
			Horizon (in years) over which to compute stats

		Returns
		-------
		_stat : pandas.DataFrame
			DataFrame indexed by tickers containing various statistics
		_corr :
			Correlation table 
		"""

		# ----------------------------------------------------------------------
		# PORTFOLIO STATISTICS

		# subset
		idx = self.X.index[-h-2:-2]
		_X = self.X.loc[idx]

		# drop
		_X.to_csv('_'.join(self.tickers) + '_JORDAN.csv')		

		# statistics
		self.mu = np.matrix(_X[self.columns].mean().to_numpy()).T
		self.Sigma = _X[self.columns].cov().to_numpy()

		# TODO: use .cov()
		_corr = _X[self.columns].corr()
		_corr.index = self.tickers
		
		# pre-processing for efficient frontier
		self.n = len(self.tickers)
		self.z = np.zeros((self.n,1))
		self.o = np.ones((self.n,1))

		# matrix and its lu decomposition
		A = np.block([
			[self.Sigma,self.mu,self.o],
			[self.mu.transpose(),0.,0.],
			[self.o.transpose(),0.,0.]])
		self.lu, self.piv = lu_factor(A)

		# efficient frontier
		@np.vectorize
		def _ef(mu_port):
			"""
			Efficient frontier
			
			Parameters
			----------
			mu_port : float
				Desired expected return
			"""
			rhs = np.block([self.o.T,mu_port,1.]).T
			x = lu_solve((self.lu,self.piv),rhs)[:-2]
			s = x.T@(self.Sigma@x)
			return np.sqrt(s[0])

		def _apply(v,s):
			"""
			Helper function 

			Parameters
			----------
			v : str
				Variable (e.g. '_RET')
			s : str
				Statistic; must be "applicable" using pandas.DataFrame.apply
			"""
			cols = [ticker + v for ticker in self.tickers]
			stat = _X[cols].apply(s)
			stat.name = s # TODO: figure out the correct way of doing this
			stat.index = self.tickers
			return stat

		# compute statistics
		_stat = {}
		_stat['sys'] = _corr['SPY_RET']						# %systematic
		_stat['exp'] = _apply('_RET','mean')				# expected return
		_stat['vol'] = _apply('_RET','std')					# volatility
		_stat['div'] = _apply('_DY','mean')					# dividend yield
		_stat['cap'] = _apply('_CG','mean')					# capital gain rate

		_stat = pd.DataFrame(_stat)

		# auxiliary
		_mvol = _stat.loc['SPY','vol']						# market volatility
		_stat['idi'] = 1.-_stat['sys']						# %idiosyncratic 
		_stat['bet'] = (_stat['vol']/_mvol)*_stat['sys'] 	# beta

		# alpha
		lm = self.X.index[-2]
		t1 = self.X.loc[lm,self.columns]
		t1.index = self.tickers
		t2 = _stat['bet']*self.X.loc[lm,'SPY_RET'] 
		_stat['alp'] = t1-t2

		# drop it
		return _stat, _corr

	def __str__(self):
		"""prints statistics about the portfolio"""
		pass
