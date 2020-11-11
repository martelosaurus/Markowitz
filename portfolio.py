# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from famafrench import FamaFrench
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve
from alpha_vantage.timeseries import TimeSeries

font = {'family' : 'normal', 'size' : 12}

class Stock:

	def __init__(self,ticker,hor=[1,5]):

		"""
		Parameters
		----------
		ticker : str
			The stock's ticker
		hor : (list of) int
			Horizons (in years) over which to compute stats
		"""
		self.ticker = ticker
		self.hor = hor
		self.stats 

class Portfolio:

	def __init__(self,key,tickers,rF=0.):
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

		# ----------------------------------------------------------------------
		# STOCK STATISTICS
		def _stat(t,v,stat,V=None,p=True):
			"""
			Helper function 

			Parameters
			----------
			t : str
				Stock ticker
			v : str
				Variable (e.g. '_RET')
			s : str
				Statistic; must be "applicable" using pandas.DataFrame.apply
			p : boolean
				If True, multiplies result by 100 
			V : str
				Second variable 
			"""
			indx =  X.index[-h*12-2:-2]
			stat = float(X.loc[indx,t+v].apply(s))
			dick = {h : 100.*stat if p else stat for h in self.hor}
			return pd.DataFrame(dick)

		# loop over stocks in portfolio
		self.stocks = {}
		_mvol = _stat('SPY','_RET','std')			# market volatility
		for t in tickers:

			# initialize a new stock
			_stock = Stock(t)

			# variance
			_vars = float(X[t+'_RET'].std())
			_corr = X[['SPY_RET',t+'_RET']].corr().loc['SPY_RET',t+'_RET']

			# compute statistics
			_stock.stats['sys'] = _corr				# percent sys risk
			_stock.stats['idi'] = 1.-_stock.sys		# percent idiosyncratic risk
			_stock.stats['div'] = _stat(t,'_DY','mean')# dividend yield
			_stock.stats['cap'] = _stat(t,'_CG','mean')# capital gain rate
			_stock.stats['exp'] = _stat(t,'_RET','mean')# expected return
			_stock.stats['vol'] = _stat(t,'_RET','std')# volatility
			_stock.stats['bet'] = np.sqrt(_vars/_varm)*_cor# beta

			_stock.stats = S

			# add stock to dictionary
			self.stocks[t] = _stock

		# ----------------------------------------------------------------------
		# PORTFOLIO STATISTICS

		# column names
		self.columns = [ticker + '_RET' for ticker in self.tickers]

		# correlation
		self.corr = self.X[self.columns].corr()

		# statistics
		self.mu = np.matrix(self.X[self.columns].mean().to_numpy()).T
		self.Sigma = self.X[self.columns].cov().to_numpy()
		
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
		self._ef = _ef

		# tangency portfolio
		self.sr_tan = 1.

	def __str__(self):
		"""prints statistics about the portfolio"""
		_str = [str(h)+'-year stats:\n\n'+str(self.stocks[h]) for h in self.hor]
		return '\n\n'.join(_str)

	def risk_return_plot(self,n_plot=100,cml=True,cal=True,sys_ido=True,
		e_f=True,mkt=False):
		"""
		Under construction
		
		Parameters
		----------
		cml : boolean
			If True, draws the Capital Market Line (CML)
		cal : boolean
			If True, draws the Capital Allocation Line (CAL)
		e_f : boolean
			If True, draws the efficient frontier
		sys : boolean
			If True, draws the systematic vs. idiosyncratic risks
		mkt : boolean
			If True, plots the mkt return
		"""

		if e_f:
			pass
		plt.plot(stdv_p,mean_p,'-')
		plt.plot(self.stdv1,self.mean1,'ok')	
		plt.plot(self.stdv2,self.mean2,'ok')	
		plt.plot(0.,0.,'ok')
		def sml(weights):
			t = np.linspace(-.25,2.25,n_plot)
			plt.plot(t*self.stdv_p,t*self.mean_p)
			plt.plot(self.stdv_p,self.mean_p,'ok')
			plt.annotate('Portfolio',(self.stdv_p,self.mean_p))
		for stock in stocks:
			plt.annotate(self.ticker1,(self.stdv1,self.mean1))
		plt.annotate('T-bill',(0.,self.rf))

		# axes
		plt.xticks([0.,self.stdv1,self.stdv2])
		plt.yticks([0.,self.mean1,self.mean2])
		plt.xlabel('Monthly Risk (%)')
		plt.ylabel('Monthly Return (%)')
		plt.grid()

		# plot
		plt.show()

	def capm_scatter(self):
		"""
		Scatter plot of stock and market
		"""
		pf = np.polyfit(dat1,dat2,1)
		plt.plot(dat1,dat2,'o')
		plt.plot(dat1,(self.alpha+self.beta*dat1),linewidth=2)
		plt.xlabel('Monthly ' + self.col1 + ' (%)')
		plt.ylabel('Monthly ' + self.col2 + ' (%)')
		title = '$\\alpha = $' + self.alpha 
		title += '; $\\beta = $' + self.beta
		title += '; $R^{2} = $' + self.rsqr
		plt.title(title)  
		plt.grid()
		plt.show()
