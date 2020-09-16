# ------------------------------------------------------------------------------
# TODO find way of automatically downloading Fama French
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

    def __init__(key,ticker1,ticker2,x1=.5,digits=4):
        """
        Initializes Twostocks object

        Parameters
        ----------

        ticker1, ticker2: str, str
            DataFrames with columns 'ticker', 'date', and 'ret'
        x1 : float
            Weight on stock1 in the portfolio (the weight on stock2 is 1-x1)

        Examples
        --------
        """     

        # parameters
        x1 = x1
        x2 = 1.-x1
        digits = 4
        ticker1 = ticker1
        ticker2 = ticker2

        #---------------------------------------------

        # load data
        ts = TimeSeries(key, output_format = "pandas")

        # look for Fama French (and download if not found)
        ff_file_name = "F-F_Research_Data_Factors.CSV"
        if ff_file_name is not in os.listdir():
            print(ff_file_name + " not found: downloading from internet")
            try: 

            except:
                raise Exception("Couldn't download " + ff_file_name)
        
        # load/clean Fama French
        try: 
            X = pd.read_csv(ff_file_name,header=2)
            X.rename(columns={'Unnamed: 0' : 'date'})
        except:
            raise Exception("Couldn't load/clean " + ff_file_name)

        X = []
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

            # meger to X list
            opts = {'left_index' : True, 'right_index' : True}
            #X = X.merge(tick_dat,**opts)
            X.append(tick_dat)

        stock1 = X[0]
        stock2 = X[1]

        # compute returns
        #for t in tickers:
            #X[t+'_RET'] = (X[t+'_PRC']+X[t+'_DIV'])/X[t+'_PRC'].shift()-1.

        # kill the first row (with the NAs)
        #X = X.loc[X.index[1:],]

        #---------------------------------------------

        # column names
        col1 = ticker1 + ' Return'
        col2 = ticker2 + ' Return'
        
        # format and merge
        #portfolio = pd.merge(stock1,stock2)
        #portfolio['Portfolio Return'] = (x1*portfolio[col1]
                #+(1.-x1)*portfolio[col2])

        ## asic statistics
        if False:
            mean1 = float(stock1.mean())
            mean2 = float(stock2.mean())
            stdv1 = float(stock1.std())
            stdv2 = float(stock2.std())
            cov12 = float(portfolio.cov().loc[col1,col2])
            var1 = stdv1**2.
            var2 = stdv2**2.
            mean_p = x1*mean1+(1.-x1)*mean2
            T1 = var1*x1**2.
            T2 = var2*(1.-x1)**2.
            T3 = 2.*x1*(1.-x1)*cov12
            stdv_p = np.sqrt(T1+T2+T3)
            corr = cov12/(stdv1*stdv2)
            sr1 = mean1/stdv1        
            sr2 = mean2/stdv2

            # portfolio object
            x1_tilde = stdv2*(sr1-sr2*corr)
            x2_tilde = stdv1*(sr2-sr1*corr)
            x1_star = x1_tilde/(x1_tilde+x2_tilde)
            x2_star = x2_tilde/(x1_tilde+x2_tilde)

            # tangency
            mean_tan = mean1*x1_star+mean2*x2_star
            stdv_tan = np.sqrt((x1_star**2.)*var1+(x2_star**2.)*var2+2.*x1_star*x2_star*cov12)

    def to_csv(**kwargs):
        X.to_csv(kwargs)

    def __str__(:
        return str({
            'mean1'     :   np.round(100.*mean1,4),
            'mean2'     :   np.round(100.*mean2,4),
            'stdv1'     :   np.round(100.*stdv1,4),
            'stdv2'     :   np.round(100.*stdv2,4),
            'corr'      :   np.round(corr,4),
            'mean_p'    :   np.round(100.*mean_p,4),
            'stdv_p'    :   np.round(100.*stdv_p,4),
            'sr_p'      :   np.round(mean_p/stdv_p,4),
            'sr_tan'    :   np.round(mean_tan/stdv_tan,4),
            'x1_tan'    :   np.round(100.*x1_star,4),
            'x2_tan'    :   np.round(100.*x2_star,4)
        })
#
    #def risk_return_plot(n_plot=1000,sml=True,cml=True,sys_ido=True,frontier=True,vertical=False):
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
        #mean_p = x1*mean1+x2*mean2
        #def stdv_p_fun(x):
            #return np.sqrt((x**2.)*var1+((1.-x)**2.)*var2+2.*x*(1.-x)*cov12)
        #stdv_p_fun = np.vectorize(stdv_p_fun)
        #stdv_p = stdv_p_fun(x1)
        #minvarport = minimize_scalar(stdv_p_fun,[mean1,mean2])
        #min_mean_p = mean1*minvarport.x+mean2*(1.-minvarport.x)
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
        #plt.plot(np.round(100.*stdv1,2),np.round(100.*mean1,2),'ok')  
        #plt.plot(np.round(100.*stdv2,2),np.round(100.*mean2,2),'ok')  
        #plt.plot(0.,0.,'ok')
        #if vertical:
            #t = np.linspace(-.25,1.25,n_plot)
            #stdv_mid = np.round(100.*.5*(stdv1+stdv2),2)
            #plt.plot(stdv_mid*np.ones(n_plot),100.*np.linspace(0.,1.5*mean1,n_plot),'-',color='tab:orange')
        #if sml:
            #leg.append('Capital Allocation Line (CAL)')
            #t = np.linspace(0.,1.5,n_plot)
            #plt.plot(t*100.*stdv_p,t*100.*mean_p,color='tab:orange',linewidth=3)
            #plt.plot(np.round(100.*stdv_p,2),np.round(100.*mean_p,2),'ok')
        #if cml: 
            #leg.append('CAL with Tangent Portfolio')
            #t = np.linspace(-.25,2.25,n_plot)
            #plt.plot(t*100.*stdv_tan,t*100.*mean_tan,color='tab:green',linewidth=3)
            #plt.plot(100.*stdv_tan,100.*mean_tan,'ok')
            #plt.annotate('Portfolio',(100.*stdv_tan,100.*mean_tan),**font)
        #plt.annotate(ticker1,(np.round(100.*stdv1,2),np.round(100.*mean1)),**font)
        #plt.annotate(ticker2,(np.round(100.*stdv2,2),np.round(100.*mean2)),**font)
        #plt.annotate('T-bill',(0.,0.),**font)
        #plt.legend(leg)
        #plt.xticks([0.,np.round(100.*stdv1,2),np.round(100.*stdv2,2)],**font)
        #plt.yticks([0.,np.round(100.*mean1,2),np.round(100.*mean2,2)],**font)
        #plt.xlabel('Monthly Risk (%)',**font)
        #plt.ylabel('Monthly Return (%)',**font)
        #if sml:
            #plt.title('Portfolio: ' + str(round(100.*x1,digits)) + ' in ' + ticker1 + '%, ' + str(round(100.*(1.-x1),digits)) + '% in ' + ticker2)
        #if cml:
            #plt.title('Portfolio: ' + str(round(100.*x1_star,2)) + ' in ' + ticker1 + '%, ' + str(round(100.*x2_star,2)) + '% in ' + ticker2)
        #plt.grid()
        #plt.show()
#
    #def scatter_plot(:
        #"""
        #Under construction
        #"""
        #dat1 = stock1[col1]
        #dat2 = stock2[col2]
        #pf = np.polyfit(dat1,dat2,1)
        #plt.plot(100.*dat1,100.*dat2,'o')
        #plt.plot(100.*dat1,100.*(alpha+beta*dat1),linewidth=3)
        #plt.xlabel('Monthly ' + col1 + ' (%)')
        #plt.ylabel('Monthly ' + col2 + ' (%)')
        #title = '$a = $' + str(np.round(alpha,digits))
        #title += '; $b = $' + str(np.round(beta,digits))
        #title += '; $R^{2} = $' + form(rsqr)
        #plt.title(title)  
        #plt.grid()
        #plt.show()
