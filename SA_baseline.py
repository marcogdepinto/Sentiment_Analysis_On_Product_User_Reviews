#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:44:04 2020

@author: Marco Polignano
"""
import pandas as pd
import ndjson
import numpy as np

from config import TRAINING_SET_PATH

dataframe = pd.DataFrame()

#LOADING TRAINING SET
with open(TRAINING_SET_PATH) as f:
            reader = ndjson.reader(f)

            for post in reader:
                df = pd.DataFrame([post], columns=post.keys())
                dataframe = pd.concat([dataframe, df],
                                           axis=0,
                                           ignore_index=True)

print(dataframe)

X = dataframe['sentence']
y = dataframe['score']



#RMSE function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


#MOST FREQUENT SCORE
r = []
for k in range(0,len(y)):
    r.append(5)


rmse_val = rmse(y, r)
print("RMS error is: " + str(rmse_val))
#ON TRAINING SET 1.16458



#MOST AVERAGE SCORE
r = []
avg = np.average(y)
for k in range(0,len(y)):
    r.append(avg)


rmse_val = rmse(y, r)
print("RMS error is: " + str(rmse_val))
#ON TRAINING SET 1.03338