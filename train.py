import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from datetime import datetime
from pytz import timezone

from dkt.data_augmentation import DataAugmentation


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
    args.cont_cols = []

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_to_load)
    train_data = preprocess.get_train_data()
    print("train size",len(train_data))
    #data_augmentation = DataAugmentation(args, train_data)
    #augmented_train_data = data_augmentation.do_data_augmentation()
    #print("aug train size", len(augmented_train_data))
    #augmented_train_data = augmented_train_data[:60000]
    train_data, valid_data = preprocess.split_data(train_data, shuffle=True)

    if not args.run_name:
        args.run_name = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")

    wandb.init(project='p-stage-4', entity='lastffang', name='-'.join([args.prefix, args.run_name]), config=vars(args))
    trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
