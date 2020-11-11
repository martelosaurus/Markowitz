# ------------------------------------------------------------------------------
import pandas as pd
from alpha_vantage.alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.alpha_vantage.timeseries import TimeSeries

def peval(key,ticker):
	# API objects
	ts = TimeSeries(key, output_format = "pandas")
	fd = FundamentalData(key,output_format="pandas")
	# income statement, balance sheet, and daily adjusted stock prices
	tick_dat_is, _ = fd.get_income_statement_quarterly(symbol=ticker)
	tick_dat_bs, _ = fd.get_balance_sheet_quarterly(symbol=ticker)
	tick_dat_da, _ = ts.get_daily_adjusted(symbol=ticker)
	# merge everything
	return tick_dat_is, tick_dat_bs

