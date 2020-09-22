import requests
from os import listdir
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from zipfile import ZipFile

# ------------------------------------------------------------------------------
_french_csv = "F-F_Research_Data_Factors.CSV"
_french_zip = "F-F_Research_Data_Factors_CSV.zip"
_french_url = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_first_ym   = 192601 # January of the first year of the data
_data_cols  = ['Mkt-RF','SMB','HML','RF']

# ------------------------------------------------------------------------------
def download_zip(url, save_path, chunk_size=128):
    """wrapper from 'requests' documentation"""
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def column_maid(row):
        try:
                return np.int64(row)
        except:
                return 99

# ------------------------------------------------------------------------------
class FamaFrench:

        def __init__(self,start='192607',end=None):

                # Fama French file name
                today = date.today().__str__().replace('-','')[:6]
                ff_file_name = "fama-french-" + today

                # look for Fama French (and download if not found or stale)
                if ff_file_name not in listdir():
                        print(ff_file_name + " not found: downloading from internet...")
                        try: 
                                download_zip(_french_url+_french_zip,_french_zip)
                                ZipFile(_french_zip).extract(_french_csv)
                        except:
                                raise Exception("Couldn't download or unzip " + ff_file_name)

                        # load/clean Fama French
                        try: 
                                # TODO this is slow, but I can't replicate with pd.read_csv
                                self.X = pd.read_csv(
                                        _french_csv,
                                        header=2,
                                        index_col=0,
                                        dtype=dict(zip(_data_cols,len(_data_cols)*[np.float])),
                                        converters = {0 : column_maid}
                                )

                                # split into monthly and annual data sets
                                # aliens/cavemen: this won't work for your time periods
                                self.X = self.X.dropna()
                                self.Xm = self.X.loc[self.X.index>_first_ym]
                                self.Xa = self.X.loc[self.X.index<_first_ym]

                        except:
                                raise Exception("Couldn't load/clean " + ff_file_name)
                # if found and not stale, load
                else:
                        try:
                                self.X = pd.read_csv(ff_file_name,header=2,index_col=1)
                        except:
                                raise Exception("Existing file not formatted correctly")

        def __str__(self):
                # TODO: print basic stats
                header = '\tMkt-RF\tSMB\tHML\tRF\n'
                return header

        def get(self,start='',end=None,monthly=True,decimal=True):
                """
                return data.frame
                """
                return self.X

        def plot(self,mktrf=True,smb=True,hml=True,start='',monthly=True,end=None):
                leg = []
                if mktrf:
                    leg.append('Market (Mkt-RF)')
                    plt.plot()
                if smb:
                    leg.append('Growth (SMB)')
                    plt.plot()
                if hml:
                    leg.append('Value (HML)')
                    plt.plot()
                plt.legend(leg)

FF = FamaFrench()
