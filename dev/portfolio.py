import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Portfolio:

	def __init__(self,D,tickers,X=None,digits=4):
		"""
		Initializes Portfolio object

		Parameters
		----------

		D : pandas.self.DataFrame
			self.DataFrames with columns 'ticker', 'date', and 'ret'
		tickers : list
			A list of tickers
		x : array like object
			Weight on stock1 in the portfolio. The weight on stock2 is 1.-x1.
		n : int
			Number of observations (default is 3)
		digits : float
			Number of digits

		Examples
		--------
		"""	

		# fix weights
		x = np.array(x)
		x = x/x.sum()

		self.D = D
		self.n = n
		self.digits = 4
		self.tickers = tickers

		# column names
		self.columns = [ticker + ' Return' for ticker in self.tickers]
		
		# format and merge
		self.stock1 = self.D.loc[self.D['ticker']==self.ticker1,['date','ret']]

		# statistics
		Sigma = D.cov().to_numpy()
		
		# pre-processing for efficient frontier
		z = np.zeros((4,1))
		o = np.ones((4,1))
		A = np.block([
			[Sigma,mu,o],
			[mu.transpose(),z,z],
			[o.transpose(),z,z]])
	
		# tangency portfolio
		tangency_mean = 1.
		tangency_stdv = 1.

	def frontier(mu_port,efficient=False):
		return sigma, x_star

	def risk_return_plot(self,n_plot=100,cml=True,cal=True,sys_ido=True,
		efficient_frontier=True):
		"""
		Under construction
		
		Parameters
		----------
		cml : boolean
			If True, draws the Capital Market Line (CML)
		cal : boolean
			If True, draws the Capital Allocation Line (CAL)
		efficient_frontier : boolean
			If True, draws the efficient frontier
		"""
		x1 = np.linspace(-.5,1.5,n_plot)
		x2 = 1.-x1
		if efficient_frontier:
			mean_p = x1*self.mean1+x2*self.mean2
			stdv_p = np.sqrt((x1**2.)*self.var1+(x2**2.)*self.var2+2.*x1*x2*self.cov12)
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
		plt.annotate('T-bill',(0.,0.))
		plt.xticks([0.,self.stdv1,self.stdv2])
		plt.yticks([0.,self.mean1,self.mean2])
		plt.xlabel('Monthly Risk (%)')
		plt.ylabel('Monthly Return (%)')
		plt.grid()
		plt.show()
		L, U = np.array()

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

