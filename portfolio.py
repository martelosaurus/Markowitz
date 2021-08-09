# ------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve
from alpha_vantage.timeseries import TimeSeries

font = {'family' : 'normal', 'size' : 12}

# efficient frontier
@np.vectorize
def _ef(n,mu):
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
        hor : int
            Horizon

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

    def summary(self,h=60,to_csv=False):
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

        # -------------------------------------------------------------------
        # PORTFOLIO STATISTICS

        # subset
        idx = self.X.index[-h-2:-2]
        _X = self.X.loc[idx]

        # drop
        if to_csv:
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
        self.lu, self.piv = lu_factor(self.Sigma)

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
        _stat['sys'] = _corr['SPY_RET']        # %systematic
        _stat['exp'] = _apply('_RET','mean')   # expected return
        _stat['vol'] = _apply('_RET','std')    # volatility
        _stat['div'] = _apply('_DY','mean')    # dividend yield
        _stat['cap'] = _apply('_CG','mean')    # capital gain rate

        _stat = pd.DataFrame(_stat)

        # auxiliary
        _mvol = _stat.loc['SPY','vol']                      # market volatility
        _stat['idi'] = 1.-_stat['sys']                      # %idiosyncratic 
        _stat['bet'] = (_stat['vol']/_mvol)*_stat['sys']    # beta

        # alpha
        lm = self.X.index[-2]
        t1 = self.X.loc[lm,self.columns]
        t1.index = self.tickers
        t2 = _stat['bet']*self.X.loc[lm,'SPY_RET'] 
        _stat['alp'] = t1-t2

        # drop it
        return _stat, _corr

    def plot(self,n_plot=100,cml=True,cal=None,risk=None,ef=True):
        """
        Plots risk-return space
        
        Parameters
        ----------
        cml : boolean
            If True, draws the Capital Market Line (CML)
        cal : float
            If not None, draws a Capital Allocation Line (CAL) with mean <cal>
        risks : str
            If not None, delineates the systematic and idiosyncratic risks
        ef : boolean
            If True, draws the efficient frontier
        """
        mu_plt = np.linspace(-1,2,n_plt) 

        # efficient frontier
        if ef:
            plt.plot(_ef(mu_plt,self.lu,self.piv),mu_plt)

        # tickers to plot
        if ticks:
            for ticker in tickers:
                ticker_mean = P.D[ticker].mean()
                ticker_stdv = P.D[ticker].std()
                plt.annotate(ticker,(ticker_stdv,ticker_mean))
                plt.plot(ticker_stdv,ticker_mean,'ok')	

        # draws a capital allocation line
        def cal(weights):
            t = np.linspace(-.25,2.25,n_plot)
            plt.plot(t*P.stdv_p,t*P.mean_p)
            plt.plot(P.stdv_p,P.mean_p,'ok')
            plt.annotate('Portfolio',(P.stdv_p,P.mean_p))

        # capital allocation line for portfolio 'x'
        if cal: 
            cal(x)

        # capital market line
        if cml:
            cal(x_tan)

        # plot the portfolio
        if port:
            plt.plot(stdv_p,mean_p,'-')
        plt.plot(0.,0.,'ok')
        plt.annotate('T-bill',(0.,0.))

        # cosmetics
        plt.xlabel('Monthly Risk (%)')
        plt.ylabel('Monthly Return (%)')
        plt.grid()
        plt.show()
