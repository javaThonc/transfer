# backtest in us stock market
import pandas as pd 
import numpy as np
import os

for path in os.listdir('./output/'):
    print(path)
    label_name = 'LABEL0'
    label = pd.read_pickle("./sp500_data_tanh.pkl")[label_name]
    score = pd.read_pickle('./output/%s/pred_score.pkl'%(path))
    score = score.to_frame()
    score['label0'] = label
    score['label1'] = score['label0'].groupby(level=0).apply(lambda x:x-x.mean())
    ret = score.groupby(level=0).apply(lambda x:x.nlargest(10, 'score').label1.mean())
    print("AR = ", ret.mean()*250)
    ret = score.groupby(level=0).apply(lambda x:x.nlargest(10, 'score').label1.mean())
    print("IR = ", ret.mean()/ret.std()*np.sqrt(250))
    ic = score.groupby(level=0).apply(lambda x:x.score.corr(x.label0))
    print("IC = ", ic.mean())
    print("ICIR = ", ic.mean()/ic.std())