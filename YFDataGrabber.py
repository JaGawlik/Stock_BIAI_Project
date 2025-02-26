import yfinance as yf
from datetime import date
import os

curretnt_dir = os.getcwd()

folder_path = os.path.join(curretnt_dir,'YFData')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

ticker = 'TSLA'

data = yf.download(ticker, start="2010-01-01", end=date.today())

print(data)

file_path = os.path.join(folder_path, f'{ticker}_historical_data.csv')
data.to_csv(file_path, sep=';', index=True)
