import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from datetime import datetime
from pytz import timezone


def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    '''
    기본 baseline의 features
    preprocess에서 feature engineering을 거치면 뒤에 더 추가됨
    '''
    args.cate_cols = ['answerCode', 'testId', 'assessmentItemID', 'KnowledgeTag']
    args.num_cols = []

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data_origin = preprocess.get_train_data()

    if not args.run_name:
        args.run_name = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    wandb.init(project='lastFown', entity='gonipark', name='-'.join([args.prefix, args.run_name]), config=vars(args))

    k = args.kfold_num
    interval = len(train_data_origin) // k
    start=0
    cv_count=1
    if args.do_CV:
        for i in range(k):
            train_data, valid_data = preprocess.split_data(train_data_origin, start, interval, shuffle=True)
            trainer.run(args, train_data, valid_data,cv_count)
            start += interval
            cv_count+=1
    else:
        train_data, valid_data = preprocess.split_data(train_data_origin, start, interval, shuffle=True)
        trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)