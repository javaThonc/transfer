import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


df =  pd.read_csv('./clean_data.csv')
"""
 compare with paper, there is no BRK-B, RDS-B, RIO
"""
stock_list ['BHP', 'AMZN', 'CVX', 'BAC', 'JNJ', 'BA', 'KO', 'AAPL', 'CHL', 'D',
              'DOW', 'CMCSA', 'PTR' , 'MRK', 'GE', 'MO', 'GOOGL', 'DCM', 'DUK',
               'DIS', 'JPM', 'NVS', 'MA', 'PEP', 'INTC', 'NTT', 'EXC',
              'SYT', 'HD', 'TOT', 'SPY', 'PFE', 'MMM', 'PG', 'MSFT', 'T', 'NGG',
              'VALE', 'TM', 'XOM', 'WFC', 'UNH', 'UPS', 'WMT', 'ORCL', 'VZ', 'SO'
           ]
all_data = np.zeros(len(stock_list), 2515)
for i in range(stock_list):
    stock = stock_list[i]
    open_df = df[df['instrument'] == stock]['$open']
    open_price = np.array(open_df)
    all_data[i] = open_price

print(all_data.shape)
np.save('crsp', all_data)





















