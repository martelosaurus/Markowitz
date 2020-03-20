import numpy as np
import pandas as pd
import re

def round_columns(D,digits=4):
	"""relies too much on duck typing; write method for pandas.DataFrame later"""

	# reformat items
	for col in list(D.columns):

		# if float, round to four decimal places
		if D[col].dtype is np.dtype('float64'):
			D[col] = D[col].map(lambda i: round(i,digits))
	return D

def format_dates(D):
	"""only matters for bonds, where you have multiple dates"""
	datere = '[0-9]{2}\/[0-9]{2}\/[0-9]{4}'
	# if date, reformat
	if D[col].dtype is np.dtype('object'): 
		if all(D[col].map(lambda d: re.search(datere,d))):
			D[col] = D[col].map(lambda i: datetime.strptime(i,'%m/%d/%Y'))
	return D
