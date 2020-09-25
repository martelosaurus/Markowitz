# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from famafrench import FamaFrench
from scipy.optimize import minimize_scalar
from scipy.linalg import lu, lu_solve
from alpha_vantage.timeseries import TimeSeries

font = {'family' : 'normal', 'size' : 12}

class Portfolio:

    def __init__(self,key,x,tickers,dow=False):
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

        Examples
        --------
        """     

        # fix weights
        x = np.array(x)
        x = x/x.sum()

        # load tickers
        if dow:
            pass
        self.tickers = tickers

        # load data
        ts = TimeSeries(key, output_format = "pandas")
        X = FamaFrench().get_monthly()

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
            opts = {'left_index' : True, 'right_index' : True}
            X = X.merge(tick_dat,**opts)

        # compute returns
        for t in [ticker1, ticker2]:
            X[t+'_RET'] = (X[t+'_PRC']+X[t+'_DIV'])/X[t+'_PRC'].shift()-1.

        # kill the first row (with the NAs)
        X = X.loc[X.index[1:],]

        # store
        self.X = X

        #---------------------------------------------

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
            [self.Sigma,self.mu,o],
            [mu.transpose(),z,z],
            [o.transpose(),z,z]])

        # tangency portfolio
        tangency_mean = 1.
        tangency_stdv = 1.

    def __str__(self):
        pass

    def frontier(self,mu_port,efficient=False):
        return sigma, x_star

    def risk_return_plot(self,n_plot=100,cml=True,cal=True,sys_ido=True,
        efficient_frontier=True,market=False):
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
        systematic : boolean
            If True, draws the systematic vs. idiosyncratic risks
        market : boolean
            If True, plots the market return
        """

        if efficient_frontier:
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

