'''
Heavily rely on
https://www.kaggle.com/fizzbuzz/how-to-perform-rank-averaging


pred=rank_average_ensemble(path_list)
writer(pred,output_file)
'''
import argparse
import numpy as np
import pandas as pd
import os

DEBUG=False

from scipy.stats import rankdata

def rank_average_ensemble(path_list, LABEL="prediction"):
    print("랭크 기반 앙상블 ", len(path_list), "파일")
    predict_list=[]

    for path in path_list:
        predict_list.append(pd.read_csv(path)[LABEL].values)

    predictions = np.zeros_like(predict_list[0])
    for predict in predict_list:
        predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])
    predictions /= len(predict_list)
    return predictions

def writer(predictions,output_file_path):
    with open(output_file_path, 'w', encoding='utf8') as f:
        f.write("id,prediction\n")
        for idx, row in enumerate(predictions):
            f.write('{},{}\n'.format(idx,predictions[idx]))