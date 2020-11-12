# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from famafrench import FamaFrench
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

	def summary(self,h=5):
		"""
		Parameters
		----------
		h : int
			Horizon (in years) over which to compute stats
		"""

		indx = X.index[-h*12-2:-2]

		# ----------------------------------------------------------------------
		# PORTFOLIO STATISTICS

		# statistics
		self.mu = np.matrix(self.X[self.columns].mean().to_numpy()).T
		self.Sigma = self.X[self.columns].cov().to_numpy()

		# TODO: use .cov()
		self._corr = self.X[self.columns]
		
		# pre-processing for efficient frontier
		self.n = len(tickers)
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

		def _stat(v,s,V=None,p=True):
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
			stat = X.loc[indx,cols].apply(s)
			stat.name = s # TODO: figure out the correct way of doing this
			return stat

		# dividend yields and capital gain rates
		_df['div'] = _stat(h,'_DY','mean')		# dividend yield
		_df['cap'] = _stat(h,'_CG','mean')		# capital gain rate

		# compute statistics
		_df = {}
		_df['sys'] = _corr.loc['RET_SPY']		# %systematic
		_df['exp'] = _stat(h,'_RET','mean')		# expected return
		_df['vol'] = _stat(h,'_RET','std')		# volatility

		_df = pd.DataFrame(_df)

		# auxiliary
		_df['idi'] = 1.-_df['sys']				# %idiosyncratic 
		_df['bet'] = (_df['vol']/_mvol)*_corr 	# beta

		# drop it
		return _df

	def __str__(self):
		"""prints statistics about the portfolio"""
		pass
