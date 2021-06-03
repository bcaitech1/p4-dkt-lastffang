import os
import torch
import numpy as np

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, RNNATTN, Bert

import wandb

def run(args, train_data, valid_data, cv_count):
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

        model_name='model'+str(cv_count)+'.pt'

        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)

        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})
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


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        '''
        input 순서는 category + numeric + mask
        
        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        + 추가 num
        + 'mask'
        '''
        
        preds = model(input)
        targets = input[0] # correct
        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)


    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)
        '''
        input 순서는 category + numeric + mask
        
        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        + 추가 num
        + 'mask'
        '''

        preds = model(input)
        targets = input[0] # correct

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets


def inference(args, test_data):

    every_fold_preds=[0 for _ in range(744)]
    for cv_num in range(1,args.kfold_num+1):
        model = load_model(args, cv_num)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)

        total_preds = []

        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)
            preds = model(input)
            # predictions
            preds = preds[:,-1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()

            total_preds+=list(preds)

        every_fold_preds=[x+y for x,y in zip(every_fold_preds,total_preds)]

        file_name='output'+str(cv_num)+'.csv'
        write_path = os.path.join(args.output_dir, file_name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write('{},{}\n'.format(id,p))

        if cv_num==args.kfold_num:
            every_fold_preds=[i/cv_num for i in every_fold_preds]

            file_name = 'output_final.csv'
            write_path = os.path.join(args.output_dir, file_name)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            with open(write_path, 'w', encoding='utf8') as w:
                print("writing prediction : {}".format(write_path))
                w.write("id,prediction\n")
                for id, p in enumerate(every_fold_preds):
                    w.write('{},{}\n'.format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn' or args.model == 'gruattn': model = RNNATTN(args)
    if args.model == 'bert': model = Bert(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    '''
    batch 순서는 category + numeric + mask
    
    'answerCode', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
    + 추가 num
    + 'mask'

    원래코드
    # test, question, tag, correct, mask = batch
    '''
    cate_features = batch[:len(args.cate_cols)]
    num_features = batch[len(args.cate_cols):len(args.cate_cols)+len(args.num_cols)]
    mask = batch[-1]
    mask = mask.type(torch.FloatTensor) # change to float

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

            interaction = cate_feature + 1 # 패딩을 위해 correct값에 1을 더해준다.
            interaction = interaction.roll(shifts=1, dims=1)
            interaction_mask = mask.roll(shifts=1, dims=1)
            interaction_mask[:, 0] = 0 # set padding index to the first sequence
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

    [features.append((num_feature * mask).to(torch.double).type(torch.FloatTensor)) for num_feature in num_features]

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
    #마지막 시퀀스에 대한 값만 loss 계산
    loss = loss[:,-1]
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



def load_model(args, cv_num):
    model_name='model'+str(cv_num)+'.pt'
    model_path = os.path.join(args.model_dir, model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
