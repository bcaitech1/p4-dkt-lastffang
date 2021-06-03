import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in self.args.cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + '_classes.npy')
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else 'unknown')

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)

        return df




    # 원하는 continuous feature 추가
    def __add_continuous_features(self, df):


        # 예로 아래와 같이 추가 해볼 수 있다.
        '''

        1개 피처 추가 하는 방법
        df['answer_mean_max'] = df.groupby(['userID'])['answer_mean'].agg('max')
        df['answer_mean_max'] = df['answer_mean_max'].fillna(float(1))


        '''

        return df

    # inference 시 사용하게 되는 확정된 features
    def __add_confirmed_continuous_features(self, df):
        df['answer_mean'] = df.groupby('userID')['answerCode'].transform('mean')  # 사용자별 정답률
        df['assessment_category_mean'] = df.groupby('assessment_category')['answerCode'].transform('mean')  # 대분류의 정답률
        df['knowledge_tag_mean'] = df.groupby('KnowledgeTag')['answerCode'].transform('mean')  # 지식 태그 분류별 정답률
        df['testId_answer_rate'] = df.groupby('testId')['answerCode'].transform('mean')  # 시험지 별로 정답률
        df['assessmentItemID_answer_rate'] = df.groupby('assessmentItemID')['answerCode'].transform('mean')  # 문항별 정답률

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp_int'] = df['Timestamp'].apply(convert_time)
        df['elapsed_time'] = df.loc[:, ['userID', 'Timestamp_int', 'testId']].groupby(
            ['userID', 'testId']).diff().shift(-1).fillna(int(10))
        df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
        df_group = df.groupby(['userID', 'testId'])['answerCode']
        df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
        df['user_total_ans_cnt'] = df_group.cumcount()
        df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']
        df['user_total_acc'] = df['user_total_acc'].fillna(float(0))
        df['et_by_kt'] = df.groupby('KnowledgeTag')['elapsed_time'].transform(
            lambda x: x.quantile(q=0.5))  # KT별 평균 소요 시간
        df['et_by_as'] = df.groupby('assessmentItemID')['elapsed_time'].transform(
            lambda x: x.quantile(q=0.5))  # 문항별 평균 소요 시간

        df['answer_mean_max'] = df.groupby(['userID'])['answer_mean'].agg('max')
        df['answer_mean_max'] = df['answer_mean_max'].fillna(float(1))


        return df

    # 원하는 categorical feature 추가
    def __add_category_features(self, df):


        return df

    # inference 시 사용하게 되는 확정된 features
    def __add_confirmed_category_features(self, df):
        df['assessment_category'] = df.apply(lambda row: row.assessmentItemID[2], axis=1)  # 대분류
        return df

    def load_data_from_file(self, file_name, is_train=True):

        csv_file_path_to_load = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path_to_load)

        print('Before feature engineering : ', self.args.cate_cols, self.args.num_cols)

        if is_train and self.args.do_train_feature_engineering:
            df = self.__add_category_features(df)
            df = self.__add_continuous_features(df)
            csv_file_path_to_write = os.path.join(self.args.data_dir, self.args.train_file_to_write)
            df.to_csv(csv_file_path_to_write,index=False)

        #inference
        if not is_train:
            df = self.__add_confirmed_category_features(df)
            df = self.__add_confirmed_continuous_features(df)


        # 사용하고자 하는 features를 아래에 작성하면 됨 #
        self.args.num_cols.extend(
            ['answer_mean', 'assessment_category_mean', 'knowledge_tag_mean', 'testId_answer_rate',
             'assessmentItemID_answer_rate', 'elapsed_time', 'user_total_acc', 'et_by_kt', 'et_by_as','answer_mean_max'])
        self.args.cate_cols.extend(['assessment_category'])

        df = self.__preprocessing(df, is_train)

        print('After feature engineering : ', self.args.cate_cols, self.args.num_cols)
        print(df[self.args.cate_cols].head())
        print(df[self.args.num_cols].head())

        # print(df.head())
        '''
        cate_len : 추후 category feature를 embedding할 시에 (model.py) embedding_layer의 input 크기를 결정할때 사용
        dictionary에 저장

        원래 코드
        # self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        # self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        # self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        '''

        self.args.cate_len = {}

        for cate_name in self.args.cate_cols:
            self.args.cate_len[cate_name] = len(np.load(os.path.join(self.args.asset_dir, f'{cate_name}_classes.npy')))

        # print(self.args.cate_len)
        # exit()
        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        columns = ['userID'] + self.args.cate_cols + self.args.num_cols

        '''
        df에서 카테고리 데이터를 label encoding한 내용으로 변환
        순서는 category + num

        원래 코드
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values
                )
            )
        '''
        group = df[columns].groupby('userID').apply(
            lambda x: tuple([x[col].values for col in self.args.cate_cols + self.args.num_cols]))

        return group.values

    def load_train_data(self,file_name):
        self.train_data = self.load_data_from_file(file_name,is_train=True)

    def load_test_data(self,file_name):
        self.test_data = self.load_data_from_file(file_name,is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        # category, numeric
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        '''
        원래 코드
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        cate_cols = [test, question, tag, correct]
        '''
        cate_cols = [row[i] for i in range(len(row))]

        '''
        #TODO
        max_seq_len까지만 사용, 나머지는 버리는데 이부분에서 data augmentation 필요
        '''

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        # category, numeric, mask
        return cate_cols

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):
    pin_memory = True
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory,
                                                   collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                                                   batch_size=args.batch_size, pin_memory=pin_memory,
                                                   collate_fn=collate)

    return train_loader, valid_loader
