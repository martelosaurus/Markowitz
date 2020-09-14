import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from alpha_vantage.timeseries import TimeSeries

def form(stuff,digits=4):
    if type(stuff) is float:
        return str(np.round(100.*stuff,digits)) + '%'
    else:
        return str(stuff)

font = {'family' : 'normal', 'size' : 12}

class TwoStocks:

    def __init__(self,key,ticker1,ticker2,x1=.5,digits=4):
        """
        Initializes Twostocks object

        Parameters
        ----------

        stock1, stock2: pandas.self.DataFrame
                self.DataFrames with columns 'ticker', 'date', and 'ret'
        x1 : float
                Weight on stock1 in the portfolio. The weight on stock2 is 1.-x1.

        Examples
        --------
        """     

        # parameters
        self.x1 = x1
        self.x2 = 1.-x1
        self.digits = 4
        self.ticker1 = ticker1
        self.ticker2 = ticker2

        #---------------------------------------------

        # load data
        ts = TimeSeries(key, output_format = "pandas")
        self.stock1 = ts.get_monthly_adjusted(ticker1)[0]
        self.stock2 = ts.get_monthly_adjusted(ticker2)[0]

        # call the API with symbol=ticker
        tick_dat, _ = ts.get_monthly_adjusted(symbol=ticker)

        # old and new columns
        old_cols = ['5. adjusted close','7. dividend amount']
        new_cols = [ticker + '_PRC', ticker + '_DIV']

        # select tick data
        tick_dat = tick_dat[old_cols]
        col_dict = dict(zip(old_cols,new_cols))
        tick_dat = tick_dat.rename(columns=col_dict)

        # meger to fund_data list
        opts = {'left_index' : True, 'right_index' : True}
        fund_data = fund_data.merge(tick_dat,**opts)

        # load Fama French factors
        ff = pd.read_csv('famafrench.csv')
        ff.index = tuple(zip(ff.Date//100,ff.Date%100))

        # TODO: check this out - want same index as for FF for merge
        self.X['Date'] = self.X.date.apply(date_fix)
        self.X.index = tuple(zip(self.X.Date//100,self.X.Date%100))
    
        # compute returns
        self.tickers = [x[:-4] for x in self.X.columns[1:] if x.endswith('_PRC')]
        for t in self.tickers:
            self.X[t+'_RET'] = (self.X[t+'_PRC']+self.X[t+'_DIV'])/self.X[t+'_PRC'].shift()-1.

        # kill the first row (with the NAs)
        self.X = self.X.loc[self.X.index[1:],]

        # merge-in FF data
        self.X = pd.merge(self.X,ff,left_index=True,right_index=True)
        self.X['Mkt'] = self.X['Mkt-RF']+self.X['RF']
        self.X[['Mkt','RF','Mkt-RF']] /= 100.

        #---------------------------------------------

        # column names
        self.col1 = self.ticker1 + ' Return'
        self.col2 = self.ticker2 + ' Return'
        
        # format and merge
        self.portfolio = pd.merge(self.stock1,self.stock2)
        self.portfolio['Portfolio Return'] = (x1*self.portfolio[self.col1]
                +(1.-x1)*self.portfolio[self.col2])

        ## asic statistics
        self.mean1 = float(self.stock1.mean())
        self.mean2 = float(self.stock2.mean())
        self.stdv1 = float(self.stock1.std())
        self.stdv2 = float(self.stock2.std())
        self.cov12 = float(self.portfolio.cov().loc[self.col1,self.col2])
        self.var1 = self.stdv1**2.
        self.var2 = self.stdv2**2.
        self.mean_p = x1*self.mean1+(1.-x1)*self.mean2
        T1 = self.var1*x1**2.
        T2 = self.var2*(1.-x1)**2.
        T3 = 2.*x1*(1.-x1)*self.cov12
        self.stdv_p = np.sqrt(T1+T2+T3)
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

    def __str__(self):
        return str({
            'mean1'     :   np.round(100.*self.mean1,4),
            'mean2'     :   np.round(100.*self.mean2,4),
            'stdv1'     :   np.round(100.*self.stdv1,4),
            'stdv2'     :   np.round(100.*self.stdv2,4),
            'corr'      :   np.round(self.corr,4),
            'mean_p'    :   np.round(100.*self.mean_p,4),
            'stdv_p'    :   np.round(100.*self.stdv_p,4),
            'sr_p'      :   np.round(self.mean_p/self.stdv_p,4),
            'sr_tan'    :   np.round(self.mean_tan/self.stdv_tan,4),
            'x1_tan'    :   np.round(100.*self.x1_star,4),
            'x2_tan'    :   np.round(100.*self.x2_star,4)
        })
#
    #def risk_return_plot(self,n_plot=1000,sml=True,cml=True,sys_ido=True,frontier=True,vertical=False):
        #"""
        #Under construction
        #
        #Parameters
        #----------
        #sml : boolean
                #If True, draws the Security Market Line (SML)
        #cml : boolean
                #If True, draws the Capital Market Line (SML)
        #"""
#
        #x1 = np.linspace(-.5,1.5,n_plot)
        #x2 = 1.-x1
        #mean_p = x1*self.mean1+x2*self.mean2
        #def stdv_p_fun(x):
            #return np.sqrt((x**2.)*self.var1+((1.-x)**2.)*self.var2+2.*x*(1.-x)*self.cov12)
        #stdv_p_fun = np.vectorize(stdv_p_fun)
        #stdv_p = stdv_p_fun(x1)
        #minvarport = minimize_scalar(stdv_p_fun,[self.mean1,self.mean2])
        #min_mean_p = self.mean1*minvarport.x+self.mean2*(1.-minvarport.x)
        #leg = []
        #if frontier:
            #plt.plot(0,0,linewidth=3,color='tab:blue')
        #if sml:
            #plt.plot(0,0,linewidth=3,color='tab:orange')
        #if cml:
            #plt.plot(0,0,linewidth=3,color='tab:green')
        #if frontier:
            #leg.append('Efficient Frontier')
            #I = mean_p>min_mean_p
            #plt.plot(np.round(100.*stdv_p,2),np.round(100.*mean_p,2),'--',color='tab:blue',linewidth=3)
            #plt.plot(np.round(100.*stdv_p[I],2),np.round(100.*mean_p[I],2),'-',color='tab:blue',linewidth=3)
        #plt.plot(np.round(100.*self.stdv1,2),np.round(100.*self.mean1,2),'ok')  
        #plt.plot(np.round(100.*self.stdv2,2),np.round(100.*self.mean2,2),'ok')  
        #plt.plot(0.,0.,'ok')
        #if vertical:
            #t = np.linspace(-.25,1.25,n_plot)
            #stdv_mid = np.round(100.*.5*(self.stdv1+self.stdv2),2)
            #plt.plot(stdv_mid*np.ones(n_plot),100.*np.linspace(0.,1.5*self.mean1,n_plot),'-',color='tab:orange')
        #if sml:
            #leg.append('Capital Allocation Line (CAL)')
            #t = np.linspace(0.,1.5,n_plot)
            #plt.plot(t*100.*self.stdv_p,t*100.*self.mean_p,color='tab:orange',linewidth=3)
            #plt.plot(np.round(100.*self.stdv_p,2),np.round(100.*self.mean_p,2),'ok')
        #if cml: 
            #leg.append('CAL with Tangent Portfolio')
            #t = np.linspace(-.25,2.25,n_plot)
            #plt.plot(t*100.*self.stdv_tan,t*100.*self.mean_tan,color='tab:green',linewidth=3)
            #plt.plot(100.*self.stdv_tan,100.*self.mean_tan,'ok')
            #plt.annotate('Portfolio',(100.*self.stdv_tan,100.*self.mean_tan),**font)
        #plt.annotate(self.ticker1,(np.round(100.*self.stdv1,2),np.round(100.*self.mean1)),**font)
        #plt.annotate(self.ticker2,(np.round(100.*self.stdv2,2),np.round(100.*self.mean2)),**font)
        #plt.annotate('T-bill',(0.,0.),**font)
        #plt.legend(leg)
        #plt.xticks([0.,np.round(100.*self.stdv1,2),np.round(100.*self.stdv2,2)],**font)
        #plt.yticks([0.,np.round(100.*self.mean1,2),np.round(100.*self.mean2,2)],**font)
        #plt.xlabel('Monthly Risk (%)',**font)
        #plt.ylabel('Monthly Return (%)',**font)
        #if sml:
            #plt.title('Portfolio: ' + str(round(100.*self.x1,self.digits)) + ' in ' + self.ticker1 + '%, ' + str(round(100.*(1.-self.x1),self.digits)) + '% in ' + self.ticker2)
        #if cml:
            #plt.title('Portfolio: ' + str(round(100.*self.x1_star,2)) + ' in ' + self.ticker1 + '%, ' + str(round(100.*self.x2_star,2)) + '% in ' + self.ticker2)
        #plt.grid()
        #plt.show()
#
    #def scatter_plot(self):
        #"""
        #Under construction
        #"""
        #dat1 = self.stock1[self.col1]
        #dat2 = self.stock2[self.col2]
        #pf = np.polyfit(dat1,dat2,1)
        #plt.plot(100.*dat1,100.*dat2,'o')
        #plt.plot(100.*dat1,100.*(self.alpha+self.beta*dat1),linewidth=3)
        #plt.xlabel('Monthly ' + self.col1 + ' (%)')
        #plt.ylabel('Monthly ' + self.col2 + ' (%)')
        #title = '$a = $' + str(np.round(self.alpha,self.digits))
        #title += '; $b = $' + str(np.round(self.beta,self.digits))
        #title += '; $R^{2} = $' + form(self.rsqr)
        #plt.title(title)  
        #plt.grid()
        #plt.show()
