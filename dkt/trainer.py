import os
import torch
import numpy as np

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM,  LastQuery, RNNATTN, Bert
from dkt.utils import setSeeds

import wandb

def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1

            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.aug_shuffle_n > 0 and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    shuffle_datas.append(data)
    for i in range(args.aug_shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data

def run(args, train_data, valid_data, cv_count=0):
    '''
    #TODO
    max_seq_len까지만 사용, 나머지는 버리는데 이부분에서 data augmentation 필요
    '''

    # augmentation
    if args.augmentation:
        args.window = True
        args.stride = args.max_seq_len
        augmented_train_data = data_augmentation(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")
            train_data = augmented_train_data

    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = int(args.total_steps * args.warmup_ratio)

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        if not args.cv_strategy:
            model_name = args.model_name
        else:
            model_name = f"{args.model_name.split('.pt')[0]}_{cv_count}.pt"

        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)

        ### VALID
        auc, acc, _, _, val_loss = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        if args.scheduler == 'plateau':
            last_lr = optimizer.param_groups[0]['lr']
        else:
            last_lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc, "val_loss":val_loss, "learning_rate": last_lr})

        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
            },
                args.model_dir, model_name,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()

    return best_auc


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        '''
        input 순서는 category + continuous + mask

        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        + 추가 cont
        + 'mask'
        '''

        preds = model(input)
        targets = input[0]  # correct
        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
            loss = loss.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
            loss = loss.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)
        '''
        input 순서는 category + continuous + mask

        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        + 추가 cont
        + 'mask'
        '''

        preds = model(input)
        targets = input[0] # correct
        loss = compute_loss(preds, targets, args)
        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
            loss = loss.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
            loss = loss.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets, loss_avg

def inference(args, test_data):
    ckpt_file_names = []
    all_fold_preds = []

    if not args.cv_strategy:
        ckpt_file_names = [args.model_name]
    else:
        ckpt_file_names = [f"{args.model_name.split('.pt')[0]}_{i + 1}.pt" for i in range(args.kfold_num)]

    for fold_idx, ckpt in enumerate(ckpt_file_names):
        model = load_model(args, ckpt)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data, True)

        total_preds = []

        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)
            preds = model(input)
            # predictions
            preds = preds[:, -1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else:  # cpu
                preds = preds.detach().numpy()

            total_preds += list(preds)

        all_fold_preds.append(total_preds)

        output_file_name = "output.csv" if not args.cv_strategy else f"output_{fold_idx + 1}.csv"
        write_path = os.path.join(args.output_dir, output_file_name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write('{},{}\n'.format(id,p))

    if len(all_fold_preds) > 1:
        # Soft voting ensemble
        votes = np.sum(all_fold_preds, axis=0) / len(all_fold_preds)

        write_path = os.path.join(args.output_dir, "output_softvote.csv")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(votes):
                w.write('{},{}\n'.format(id,p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    setSeeds(args.seed)

    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn' or args.model == 'gruattn': model = RNNATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'lqtrnn': model = LastQuery(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    '''
    batch 순서는 category + continuous + mask
    'answerCode', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
    + 추가 cont
    + 'mask'
    원래코드
    # test, question, tag, correct, mask = batch
    '''
    cate_features = batch[:len(args.cate_cols)]
    cont_features = batch[len(args.cate_cols):len(args.cate_cols)+len(args.cont_cols)]
    mask = batch[-1]
    mask = mask.type(torch.FloatTensor)  # change to float

    features = []

    for name, cate_feature in zip(args.cate_cols, cate_features):
        if name == 'answerCode':
            # correct
            # correct = correct.type(torch.FloatTensor)
            features.append(cate_feature.type(torch.FloatTensor))

            '''
            interaction
            interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
            saint의 경우 decoder에 들어가는 input이다
            오피스아워에서 언급한 코드 수정내용 반영
            '''

            interaction = cate_feature + 1  # 패딩을 위해 correct값에 1을 더해준다.
            interaction = interaction.roll(shifts=1, dims=1)
            interaction_mask = mask.roll(shifts=1, dims=1)
            interaction_mask[:, 0] = 0  # set padding index to the first sequence
            interaction = (interaction * interaction_mask).to(torch.int64)

            features.append(interaction)
        else:
            '''
            일반 category
            원래 코드
            test = ((test + 1) * mask).to(torch.int64)
            question = ((question + 1) * mask).to(torch.int64)
            tag = ((tag + 1) * mask).to(torch.int64)
            '''
            # question, test, tag
            features.append(((cate_feature + 1) * mask).to(torch.int64))

    [features.append((cont_feature * mask).to(torch.double).type(torch.FloatTensor)) for cont_feature in cont_features]

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    # gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    # gather_index = gather_index.view(-1, 1) - 1

    '''
    device memory로 이동
    원래 코드
    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    interaction = interaction.to(args.device)
    mask = mask.to(args.device)
    '''

    features.append(mask)
    features = [feature.to(args.device) for feature in features]

    # return (test, question,
    #         tag, correct, mask,
    #         interaction)

    return tuple(features)


# loss계산하고 parameter update!
def compute_loss(preds, targets, args):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
    """
    loss = get_criterion(preds, targets, args)
    # 마지막 시퀀스에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args, model_name=None):
    if not model_name:
        model_name = args.model_name
    model_path = os.path.join(args.model_dir, model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
