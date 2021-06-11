import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from datetime import datetime
from pytz import timezone
from sklearn.model_selection import KFold, StratifiedKFold


def main(args):
    wandb.login()

    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    '''
    기본 baseline의 features
    preprocess에서 feature engineering을 거치면 뒤에 더 추가됨
    '''
    args.cate_cols = ['answerCode', 'testId', 'assessmentItemID', 'KnowledgeTag']
    args.cont_cols = []

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_to_load)
    train_data_origin = preprocess.get_train_data()

    if not args.run_name:
        args.run_name = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")

    wandb.init(project='p-stage-4', entity='lastffang', name='-'.join([args.prefix, args.run_name]), config=vars(args))

    k = args.kfold_num

    if args.cv_strategy == "random":
        kf = KFold(n_splits=k, shuffle=True)
        splits = kf.split(X=train_data_origin)
    else:
        # No cross validation option just runs on first skf train-validation split
        train_last_answerCode = [user_sequence[0][-1] for user_sequence in train_data_origin]
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        splits = skf.split(X=train_data_origin, y=train_last_answerCode)

    auc_avg = 0

    for fold_num, (train_index, valid_index) in enumerate(splits):
        train_data = train_data_origin[train_index]
        valid_data = train_data_origin[valid_index]
        best_auc = trainer.run(args, train_data, valid_data, fold_num + 1)

        if not args.cv_strategy:
            break

        auc_avg += best_auc

    if args.cv_strategy:
        auc_avg /= k

        print("*" * 50, 'auc_avg', "*" * 50)
        print(auc_avg)


if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
