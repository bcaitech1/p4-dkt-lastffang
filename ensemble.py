'''
Heavily rely on
https://www.kaggle.com/fizzbuzz/how-to-perform-rank-averaging

'''
import argparse
import numpy as np
import pandas as pd
import os

DEBUG=False

from scipy.stats import rankdata
def average_ensemble(path_list,LABEL="prediction"):
    print("평균 앙상블 ", len(path_list), "개 파일")
    predict_list=[]

    for path in path_list:
        predict_list.append(pd.read_csv(path)[LABEL].values)

    predictions = np.zeros_like(predict_list[0])
    for predict in predict_list:
        predictions = np.add(predictions, predict)
    predictions /= len(predict_list)
    return predictions

def rank_average_ensemble(path_list, LABEL="prediction"):
    print("랭크 기반 앙상블 ", len(path_list), "개 파일")
    predict_list=[]

    for path in path_list:
        predict_list.append(pd.read_csv(path)[LABEL].values)

    predictions = np.zeros_like(predict_list[0])
    for predict in predict_list:
        predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])
    predictions /= len(predict_list)
    return predictions

def weight_average_ensemble(path_list, weight, LABEL="prediction"):
    assert len(path_list)==len(weight)
    assert 0.99<sum(weight)<1.01
    print("가중 앙상블 ", len(path_list), "개 파일")
    predict_list=[]

    for path in path_list:
        predict_list.append(pd.read_csv(path)[LABEL].values)

    predictions = np.zeros_like(predict_list[0])
    for idx, predict in enumerate(predict_list):
        predictions = np.add(predictions, predict*weight[idx])
    return predictions

def writer(predictions,output_file_path):
    with open(output_file_path, 'w', encoding='utf8') as f:
        f.write("id,prediction\n")
        for idx, row in enumerate(predictions):
            f.write('{},{}\n'.format(idx,predictions[idx]))

if __name__ == "__main__":
    path_list=["./en/ens2.csv","./en/ens3.csv","./en/ens4.csv","./en/ens5.csv","./en/ens6.csv","./en/ens7.csv"]
    for i in path_list:
        print(i)
    pred=average_ensemble(path_list, LABEL="prediction")
    writer(pred,"./super_avg.csv")