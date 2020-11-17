import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


df =  pd.read_csv('./clean_46.csv')
"""
 compare with paper, there is no BRK-B, RDS-B, RIO
"""
stock_list =  ['BHP', 'AMZN', 'CVX', 'BAC', 'JNJ', 'BA', 'KO', 'AAPL', 'CHL', 'D',
              'DOW', 'CMCSA', 'PTR' , 'MRK', 'GE', 'MO', 'DCM', 'DUK',
               'DIS', 'JPM', 'NVS', 'MA', 'PEP', 'INTC', 'NTT', 'EXC',
              'SYT', 'HD', 'TOT', 'SPY', 'PFE', 'MMM', 'PG', 'MSFT', 'T', 'NGG',
              'TM', 'XOM', 'WFC', 'UNH', 'UPS', 'WMT', 'ORCL', 'VZ', 'SO'
           ]
all_data = np.zeros((len(stock_list), 2515),dtype = np.float32 )
for i in range(len(stock_list)):
    stock = stock_list[i]
    open_df = df[df['instrument'] == stock]['$open']
    open_price = np.array(open_df)
    print(stock)
    print(len(open_price))
    all_data[i] = open_price

print(all_data.shape)
np.save('crsp', all_data)





















