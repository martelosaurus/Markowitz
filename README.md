# Markowitz
Implements Markowitz portfolio theory (for educational purposes). **Requires an Alpha Vantage API key.** This repository and its contents were written by Jordan Martel, who is no way affiliated with Alpha Vantage. Currenlty only considers two stock portfolios.

```python
portfolio1 = twostock.TwoStocks(D,'AAPL','PG',n=1000)
portfolio2 = twostock.TwoStocks(D,'CVX','XOM',n=1000)
portfolio3 = twostock.TwoStocks(D,'AAPL','MSFT',n=1000)
portfolio4 = twostock.TwoStocks(D,'WBA','WMT',n=1000)
x = portfolio4.summary()
```
