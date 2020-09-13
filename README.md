# Markowitz
Implements Markowitz portfolio theory (for educational purposes)

portfolio1 = twostock.TwoStocks(D,'AAPL','PG',n=1000)
portfolio2 = twostock.TwoStocks(D,'CVX','XOM',n=1000)
portfolio3 = twostock.TwoStocks(D,'AAPL','MSFT',n=1000)
portfolio4 = twostock.TwoStocks(D,'WBA','WMT',n=1000)
x = portfolio4.summary()
