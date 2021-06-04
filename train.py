import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from datetime import datetime
from pytz import timezone
from os import system
def inferenceForCV(model, cv_count, every_fold_preds):

    setSeeds(42)

    '''
    기본 baseline의 features
    preprocess에서 feature engineering을 거치면 뒤에 더 추가됨
    '''
    args.cate_cols = ['answerCode', 'testId', 'assessmentItemID', 'KnowledgeTag']
    args.num_cols = []

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    preds=trainer.inference(args, test_data, model)
    print(preds)
    every_fold_preds = [x + y for x, y in zip(every_fold_preds, preds)]
    print(every_fold_preds)
    if cv_count== args.kfold_num:
        every_fold_preds = [i / args.kfold_num for i in every_fold_preds]

        write_path = os.path.join(args.output_dir,'output_final.csv')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(every_fold_preds):
                w.write('{},{}\n'.format(id, p))

    return every_fold_preds

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
    preprocess.load_train_data(args.file_name)

    train_data_origin = preprocess.get_train_data()
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()


    if not args.run_name:
        args.run_name = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")

    wandb.init(project='p-stage-4', entity='lastffang', name='-'.join([args.prefix, args.run_name]), config=vars(args))

    args.kfold_num = 5
    args.do_CV = True
    k = args.kfold_num
    
    interval = len(train_data_origin) // k
    start = 0

    if args.do_CV == True:
        every_fold_preds = [0 for _ in range(744)]
        auc_avg=0
        for cv_count in range(1,k+1):
            train_data, valid_data = preprocess.split_data(train_data_origin, start, interval, shuffle=True)
            best_model, best_auc = trainer.run(args, train_data, valid_data, cv_count)
            start += interval
            every_fold_preds = inferenceForCV(best_model, cv_count, every_fold_preds)
            auc_avg+=best_auc
        auc_avg/=5
        print("*"*50,'auc_avg',"*"*50)
        print(auc_avg)

    else:
        train_data, valid_data = preprocess.split_data(train_data_origin, start, interval, shuffle=True, seed=args.seed)
        print(len(train_data), len(valid_data))
        trainer.run(args, train_data, valid_data)

    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
