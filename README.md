# Markowitz
Implements Markowitz portfolio theory (for educational purposes). **Requires an Alpha Vantage API key.** This repository and its contents were written by Jordan Martel, who is no way affiliated with Alpha Vantage. Currenlty only considers two stock portfolios.

```python
from portfolio import Portfolio
portfolio1 = Portfolio('A1B2C3D4',['AAPL','PG','MCD','DIS'])
print(portfolio1)
portfolio1.plot(sml=True)
```
