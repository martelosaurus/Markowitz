# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from famafrench import FamaFrench
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve
from alpha_vantage.timeseries import TimeSeries

font = {'family' : 'normal', 'size' : 12}

class Stock:

    def __init__(self,ticker):
        self.ticker = ticker
        self.sys = None # fraction of risk that's systematic
        self.div = None # dividend yield
        self.cap = None # capital gain rate

class Portfolio:

    def __init__(self,key,tickers,dow=False,spy=True):
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
        #x = np.array(x)
        #x = x/x.sum()

        # load tickers
        if dow:
            pass
        self.tickers = tickers
        if spy:
            tickers.append('SPY')

        # load data
        ts = TimeSeries(key, output_format = "pandas")
        X = None
        if not spy:
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

            print(tick_dat)

            # meger to X list
            opts = {'left_index' : True, 'right_index' : True}
            if spy:
                if X is None:
                    X = tick_dat.iloc[::-1]
                else:
                    X = X.merge(tick_dat,**opts)
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

        # compute stats
        self.stocks = {}
        for t in tickers:
            _stock = Stock(t)
            #_stock.sys = X[['Mkt-RF',t+'_RET']].corr().loc['Mkt-RF',t+'_RET']
            _stock.sys = X[['SPY_RET',t+'_RET']].corr().loc['SPY_RET',t+'_RET']
            _stock.idi = 1.-_stock.sys
            _stock.div = float(X[t+'_DY'].mean())
            _stock.cap = float(X[t+'_CG'].mean())
            _stock.exp = float(X[t+'_RET'].mean())
            _stock.vol = float(X[t+'_RET'].std())
            self.stocks[t] = _stock

        # store
        self.X = X

        # column names
        self.columns = [ticker + '_RET' for ticker in self.tickers]
        
        # statistics
        self.mu = self.X[self.columns].mean().to_numpy()
        self.mu = np.matrix(self.mu).T
        self.Sigma = self.X[self.columns].cov().to_numpy()
        
        # pre-processing for efficient frontier
        self.n = len(tickers)
        self.z = np.zeros((self.n,1))
        self.o = np.ones((self.n,1))

        A = np.block([
            [self.Sigma,self.mu,self.o],
            [self.mu.transpose(),0.,0.],
            [self.o.transpose(),0.,0.]])

        self.lu, self.piv = lu_factor(A)

        @np.vectorize
        def _ef(mu_port):
            rhs = np.block([self.o.T,mu_port,1.]).T
            x = lu_solve((self.lu,self.piv),rhs)[:-2]
            s = x.T@(self.Sigma@x)
            return np.sqrt(s[0])

        self._ef = _ef
        self.sr_tan = 1.

    def __str__(self):
        pass

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

