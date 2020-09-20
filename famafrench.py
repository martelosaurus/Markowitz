import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# file name and url
ff_file_name = "F-F_Research_Data_Factors.CSV"
ff_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

class FamaFrench:

	def __init__(self,start='200909',end=None):

		# look for Fama French (and download if not found)
		if ff_file_name is not in os.listdir():
			print(ff_file_name + " not found: downloading from internet")
			try: 
				r = requests.get(url)
			except:
				raise Exception("Couldn't download " + ff_file_name)

		# load/clean Fama French
		try: 
			X = pd.read_csv(ff_file_name,header=2)
			X.rename(columns={'Unnamed: 0' : 'date'})
		except:
			raise Exception("Couldn't load/clean " + ff_file_name)


	def __str__(self):
		pass

	def get(self):
		"""
		return data.frame
		"""

	def plot(self):
		pass
