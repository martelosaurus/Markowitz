def beta_plots(df):
	cov = df.cov()
	corr = df.corr()
	std = df.std()
	avg = df.mean()
	stats = {
		'rf_rtrn' : avg['rf'],
		'mkt_rtrn' : avg['mkt'],
		'ret_rtrn' : avg['ret'],
		'mkt_risk' : std['mkt'],
		'ret_risk' : std['ret'],
		'ret_mkt_cov' : cov.loc['ret','mkt'],
		'ret_mkt_corr' : corr.loc['ret','mkt'],
	} 
	beta = {
		'beta' : round(stats['ret_mkt_cov']/stats['mkt_risk']**2.,4)
	}
	perf = {
		'mkt_sharpe' : round((stats['mkt_rtrn']-stats['rf_rtrn'])/stats['mkt_risk'],4),
		'ret_sharpe' : round((stats['ret_rtrn']-stats['rf_rtrn'])/stats['ret_risk'],4),
		'ret_prem' : round(100.*(stats['ret_rtrn']-stats['rf_rtrn']),4),
		'mkt_prem' : round(100.*(stats['mkt_rtrn']-stats['rf_rtrn']),4)
	}
	stats = {	
		'rf_rtrn' : round(100.*avg['rf'],4),
		'mkt_rtrn' : round(100.*avg['mkt'],4),
		'ret_rtrn' : round(100.*avg['ret'],4),
		'mkt_risk' : round(100.*std['mkt'],4),
		'ret_risk' : round(100.*std['ret'],4),
		'ret_mkt_corr' : round(corr.loc['ret','mkt'],4),
	}
	beta['alpha'] = round(perf['ret_prem']-beta['beta']*perf['mkt_prem'],4)
	return stats, beta, perf
	
