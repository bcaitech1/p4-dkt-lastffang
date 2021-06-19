import os
from datetime import datetime
import time
import pandas as pd
import numpy as np


def main():
    dtype = {
        'userID': 'int16',
        'answerCode': 'int8',
        'KnowledgeTag': 'int16'
    }

    data_dir = '/opt/ml/input/data/train_dataset/'


    # ### 사용하려는 dataset 이름을 아래 cell 에서 설정해주세요!!!!

    # Baseline dataset 사용하는 경우 아래 주석 해제!
    # train_name = 'train_data.csv'
    # test_name = 'test_data.csv'

    # Time delta dataset 사용하는 경우 아래 주석 해제!
    train_name = 'train_all_three.csv'
    test_name = 'test_all_three.csv'


    raw_train_df = pd.read_csv(os.path.join(data_dir, train_name), dtype=dtype, parse_dates=['Timestamp'])
    raw_train_df = raw_train_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    raw_test_df = pd.read_csv(os.path.join(data_dir, test_name), dtype=dtype, parse_dates=['Timestamp'])
    raw_test_df = raw_test_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    # TODO: 전체 데이터에 대해서 통계량 계산 가능
    # full_df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)
    # full_df = full_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    # ### train 데이터에서 시험지/문제/태그별 평균 정답률과 총 정답 수 뽑기

    testId_mean_sum = raw_train_df.groupby(['testId'])['answerCode'].agg(['mean','sum']).to_dict()
    assessmentItemID_mean_sum = raw_train_df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum']).to_dict()
    KnowledgeTag_mean_sum = raw_train_df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum']).to_dict()


    # ### train 데이터에서 대분류별 평균 정답률과 총 정답 수 뽑기
    raw_train_df['assessment_category'] = raw_train_df.apply(lambda row: row.assessmentItemID[2], axis=1)

    assessment_category_mean_sum = raw_train_df.groupby(['assessment_category'])['answerCode'].agg(['mean', 'sum']).to_dict()


    # ### train 데이터에서 문제/태그별 평균 소요 시간 뽑기

    # 먼저 소요 시간을 계산
    def convert_time(s):
        timestamp = time.mktime(s.timetuple())
        return int(timestamp)

    if 'elapsed' in raw_train_df.columns:
        raw_train_df['elapsed_time'] = raw_train_df['elapsed'].apply(lambda x: np.log(x))
    else:
        # TODO: Deal with zeros first before using log of elapsed_time
        raw_train_df['Timestamp_int'] = raw_train_df['Timestamp'].apply(convert_time)
        raw_train_df['elapsed_time'] = raw_train_df.loc[:, ['userID', 'Timestamp_int', 'testId']].groupby(
            ['userID', 'testId']).diff().shift(-1).fillna(int(10))


    et_by_kt_mean = raw_train_df.groupby(['KnowledgeTag'])['elapsed_time'].agg(['mean']).to_dict()
    et_by_as_mean = raw_train_df.groupby(['assessmentItemID'])['elapsed_time'].agg(['mean']).to_dict()


    # ### 구한 train 통계량들로 데이터 만들기
    def set_basic_stats(df):
        df["testId_answer_rate"] = df.testId.map(testId_mean_sum['mean'])
        df['testId_answer_sum'] = df.testId.map(testId_mean_sum['sum'])
        df["assessmentItemID_answer_rate"] = df.assessmentItemID.map(assessmentItemID_mean_sum['mean'])
        df['assessmentItemID_answer_sum'] = df.assessmentItemID.map(assessmentItemID_mean_sum['sum'])
        df["knowledge_tag_mean"] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['mean'])
        df['knowledge_tag_sum'] = df.KnowledgeTag.map(KnowledgeTag_mean_sum['sum'])

        if 'assessment_category' not in df.columns:
            df['assessment_category'] = df.apply(lambda row: row.assessmentItemID[2], axis=1)
        df['assessment_category_mean'] = df.assessment_category.map(assessment_category_mean_sum['mean'])

        if 'elapsed_time' not in df.columns:
            if 'elapsed' in df.columns:
                df['elapsed_time'] = df['elapsed'].apply(lambda x: np.log(x))
            else:
                df['Timestamp_int'] = df['Timestamp'].apply(convert_time)
                df['elapsed_time'] = df.loc[:, ['userID', 'Timestamp_int', 'testId']].groupby(
                    ['userID', 'testId']).diff().shift(-1).fillna(int(10))

        df['et_by_kt'] = df.KnowledgeTag.map(et_by_kt_mean['mean'])
        df['et_by_as'] = df.assessmentItemID.map(et_by_as_mean['mean'])

        return df

    raw_train_df = set_basic_stats(raw_train_df)
    raw_train_df.head()

    raw_test_df = set_basic_stats(raw_test_df)
    raw_test_df.head()

    # ### 새로운 데이터를 csv 로 저장
    # - 원래 데이터 이름 뒤에 "basic_stats" 가 붙은 새로운 파일로 저장됩니다.

    train_name = f"{train_name.split('.')[0]}_basic_stats.csv"
    test_name = f"{test_name.split('.')[0]}_basic_stats.csv"

    raw_train_df.to_csv(os.path.join(data_dir, train_name), sep=',', index=False)
    raw_test_df.to_csv(os.path.join(data_dir, test_name), sep=',', index=False)


if __name__ == "__main__":
    main()

