# ------------------------------------------------------------------------------
# TODO find way of automatically downloading Fama French
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from famafrench import FamaFrench
from scipy.optimize import minimize_scalar
from alpha_vantage.timeseries import TimeSeries

@np.vectorize
def rnd(x,d=2):
	return round(100.*x,d)

def form(stuff,digits=4):
    if type(stuff) is float:
        return str(rnd(stuff,digits)) + '%'
    else:
        return str(stuff)

font = {'family' : 'normal', 'size' : 12}

class TwoStocks:

    def __init__(self,key,ticker1,ticker2,x1=.5,digits=4):
        """
        Initializes Twostocks object

        Parameters
        ----------

        ticker1, ticker2: str, str
        	Tickers of first and second stock
        x1 : float
            Weight on stock1 in the portfolio (the weight on stock2 is 1-x1)

        Examples
        --------
        """     

        # parameters
        self.x1 = x1
        self.x2 = 1.-self.x1
        self.digits = 4
        self.ticker1 = ticker1
        self.ticker2 = ticker2

        #---------------------------------------------

        # load data
        ts = TimeSeries(key, output_format = "pandas")

        #X = FamaFrench().get(start,end)
        X = pd.read_csv('trash.csv',index_col=0)

        # loop over tickers
        for ticker in [ticker1, ticker2]:

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

        self.X = X

        #---------------------------------------------

        # column names
        col1 = ticker1 + '_RET'
        col2 = ticker2 + '_RET'
        
        ## asic statistics
        self.mean1 = float(X[col1].mean())
        self.mean2 = float(X[col2].mean())
        self.stdv1 = float(X[col1].std())
        self.stdv2 = float(X[col2].std())
        self.cov12 = float(X[[col1,col2]].cov().loc[col1,col2])
        self.var1 = self.stdv1**2.
        self.var2 = self.stdv2**2.
        T1 = self.var1*x1**2.
        T2 = self.var2*(1.-x1)**2.
        T3 = 2.*x1*(1.-x1)*self.cov12
        stdv_p = np.sqrt(T1+T2+T3)
        self.corr = self.cov12/(self.stdv1*self.stdv2)
        self.sr1 = self.mean1/self.stdv1        
        self.sr2 = self.mean2/self.stdv2

        # portfolio object
        x1_tilde = self.stdv2*(self.sr1-self.sr2*self.corr)
        x2_tilde = self.stdv1*(self.sr2-self.sr1*self.corr)
        self.x1_star = x1_tilde/(x1_tilde+x2_tilde)
        self.x2_star = x2_tilde/(x1_tilde+x2_tilde)

        # tangency
        self.mean_tan = self.mean1*self.x1_star+self.mean2*self.x2_star
        self.stdv_tan = np.sqrt((self.x1_star**2.)*self.var1+(self.x2_star**2.)*self.var2+2.*self.x1_star*self.x2_star*self.cov12)

    def  __str__(self):
        return str({
            'mean1'     :   rnd(self.mean1,4),
            'mean2'     :   rnd(self.mean2,4),
            'stdv1'     :   rnd(self.stdv1,4),
            'stdv2'     :   rnd(self.stdv2,4),
            'corr'      :   np.round(self.corr,4),
            'x1_tan'    :   rnd(self.x1_star,4),
            'x2_tan'    :   rnd(self.x2_star,4)
        })

    def risk_return_plot(self,
		n_plot=1000,
		sml=True,
		cml=True,
		sys_ido=True,
		frontier=True,
		vertical=False
	):
        """
        Under construction
        
        Parameters
        ----------
        sml : boolean
                If True, draws the Security Market Line (SML)
        cml : boolean
                If True, draws the Capital Market Line (SML)
        """
#
        x1 = np.linspace(-.5,1.5,n_plot)
        x2 = 1.-x1
        mean_p = self.x1*self.mean1+self.x2*self.mean2
        def stdv_p_fun(x):
            return np.sqrt((x**2.)*self.var1+((1.-x)**2.)*self.var2+2.*x*(1.-x)*self.cov12)
        stdv_p_fun = np.vectorize(stdv_p_fun)
        stdv_p = stdv_p_fun(self.x1)
        minvarport = minimize_scalar(stdv_p_fun,[self.mean1,self.mean2])
        min_mean_p = self.mean1*minvarport.x+self.mean2*(1.-minvarport.x)
        leg = []
        if frontier:
            plt.plot(0,0,linewidth=3,color='tab:blue')
        if sml:
            plt.plot(0,0,linewidth=3,color='tab:orange')
        if cml:
            plt.plot(0,0,linewidth=3,color='tab:green')
        if frontier:
            leg.append('Efficient Frontier')
            I = mean_p>min_mean_p
            plt.plot(rnd(stdv_p,2),rnd(mean_p,2),'--',color='tab:blue',linewidth=3)
            plt.plot(rnd(stdv_p[I],2),rnd(mean_p[I],2),'-',color='tab:blue',linewidth=3)
        plt.plot(rnd(self.stdv1,2),rnd(self.mean1,2),'ok')  
        plt.plot(rnd(self.stdv2,2),rnd(self.mean2,2),'ok')  
        plt.plot(0.,0.,'ok')
        if vertical:
            t = np.linspace(-.25,1.25,n_plot)
            stdv_mid = rnd(.5*(stdv1+stdv2),2)
            plt.plot(stdv_mid*np.ones(n_plot),100.*np.linspace(0.,1.5*mean1,n_plot),'-',color='tab:orange')
        if sml:
            leg.append('Capital Allocation Line (CAL)')
            t = np.linspace(0.,1.5,n_plot)
            plt.plot(t*100.*stdv_p,t*100.*mean_p,color='tab:orange',linewidth=3)
            plt.plot(rnd(stdv_p,2),rnd(mean_p,2),'ok')
        if cml: 
            leg.append('CAL with Tangent Portfolio')
            t = np.linspace(-.25,2.25,n_plot)
            plt.plot(t*100.*self.stdv_tan,t*100.*self.mean_tan,color='tab:green',linewidth=3)
            plt.plot(100.*self.stdv_tan,100.*self.mean_tan,'ok')
            plt.annotate('Portfolio',(100.*self.stdv_tan,100.*self.mean_tan),**font)
        plt.annotate(self.ticker1,(rnd(self.stdv1,2),rnd(self.mean1)),**font)
        plt.annotate(self.ticker2,(rnd(self.stdv2,2),rnd(self.mean2)),**font)
        plt.annotate('T-bill',(0.,0.),**font)
        plt.legend(leg)
        plt.xticks([0.,rnd(self.stdv1,2),rnd(self.stdv2,2)],**font)
        plt.yticks([0.,rnd(self.mean1,2),rnd(self.mean2,2)],**font)
        plt.xlabel('Monthly Risk (%)',**font)
        plt.ylabel('Monthly Return (%)',**font)
        if sml:
            plt.title('Portfolio: ' + str(round(100.*self.x1,2)) + ' in ' + self.ticker1 + '%, ' + str(round(100.*(1.-self.x1),2)) + '% in ' + self.ticker2)
        if cml:
            plt.title('Portfolio: ' + str(round(100.*self.x1_star,2)) + ' in ' + self.ticker1 + '%, ' + str(round(100.*self.x2_star,2)) + '% in ' + self.ticker2)
        plt.grid()
        plt.show()
#
    def scatter_plot(self):
        """
        Under construction
        """
        dat1 = stock1[col1]
        dat2 = stock2[col2]
        pf = np.polyfit(dat1,dat2,1)
        plt.plot(100.*dat1,100.*dat2,'o')
        plt.plot(100.*dat1,100.*(alpha+beta*dat1),linewidth=3)
        plt.xlabel('Monthly ' + col1 + ' (%)')
        plt.ylabel('Monthly ' + col2 + ' (%)')
        title = '$a = $' + str(np.round(alpha,digits))
        title += '; $b = $' + str(np.round(beta,digits))
        title += '; $R^{2} = $' + form(rsqr)
        plt.title(title)  
        plt.grid()
        plt.show()
