import numpy as np
import matplotlib.pyplot as plt

# parameters------------------------------------------------------------------#
n = 100
K = .5
S = np.linspace(0.,1.,n)
sz = 26

# option functions------------------------------------------------------------#
def call(s,k):
    return np.max(np.array([s-k,0*s]),axis=0)

def put(s,k): 
    return np.max(np.array([k-s,0*s]),axis=0)

# plotting functions----------------------------------------------------------#
def driver(name,payoff):
    plt.plot(S,payoff,'-k',linewidth=6)
    plt.xlabel('stock price ($S$)',size=sz)
    plt.ylabel('portfolio payoff',size=sz)
    plt.yticks(np.array([-2*K,-K,0,K,2*K]),['$-2K$','$-K$','$0$','$K$','$2K$'],size=sz)
    #plt.yticks(np.array([-K,0,K]),['$-K$','$0$','$K$'],size=sz)
    plt.xticks(np.array([0,K]),['$0$','$K$'],size=sz)
    plt.axis([0.,2*K,-2*K,2*K])
    #plt.axis([0,2*K,-K,K])
    plt.grid()
    plt.savefig(name+".pdf",bbox_inches='tight')
    plt.close()

def option_plot(name,payoff):
    """Wrapper for driver"""
    driver("long"+name,payoff)
    driver("short"+name,-payoff)

# plot------------------------------------------------------------------------#
option_plot("stock",S)               
option_plot("riskfree",K+0.*S-.025)            
option_plot("call",call(S,K))          
option_plot("put",put(S,K))           
option_plot("straddle",call(S,K)+put(S,K)) 
option_plot("coveredcall",S-call(S,K))    
option_plot("protectiveput",S+put(S,K))  
option_plot("forward",S-K)
