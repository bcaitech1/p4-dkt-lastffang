import os
import argparse
import json

with open('arg_choices.json') as f:
    choices = json.load(f)

def parse_args(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')

    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')

    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model.pt', type=str, help='model file name')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')

    parser.add_argument('--max_seq_len', default=10, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    # wandb
    if mode == 'train':
        parser.add_argument('--prefix', required=True, type=str, help='prefix of wandb run name (e.g. username or initials).')
        parser.add_argument('--run_name', type=str, help='wandb run name. Defaults to current time')

    # 모델
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=8, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.8, type=float, help='drop out rate')

    # Optimizer
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay of optimizer')

    # Scheduler
    parser.add_argument('--plateau_patience', default=10, type=int, help='patience of plateau scheduler')
    parser.add_argument('--plateau_factor', default=0.5, type=float, help='factor of plateau scheduler')

    # 훈련
    parser.add_argument('--n_epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=15, type=int, help='for early stopping')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup step ratio')

    #cv
    parser.add_argument('--kfold_num', default=5, type=int, help='number of fold')
    parser.add_argument('--do_CV', action='store_true', help='do cross validation or not')

    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')


    ### 중요 ###
    parser.add_argument('--model', default='lstm', type=str, choices=choices["model_options"], help='model type')
    parser.add_argument('--optimizer', default='adam', type=str, choices=choices["optimizer_options"], help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, choices=choices["scheduler_options"], help='scheduler type')
    parser.add_argument('--criterion', default='BCE', type=str, choices=choices["criterion_options"], help='criterion type')



    args = parser.parse_args()

    return args