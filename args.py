import os
import argparse
import json

with open('arg_choices.json') as f:
    choices = json.load(f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--device', default='gpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')

    parser.add_argument('--train_file_to_load', default='train_data_basic_stats.csv', type=str, help='train file name to load')
    parser.add_argument('--do_train_feature_engineering', default='True', type=str2bool, help='whether do feature engineering or not')
    parser.add_argument('--train_file_to_write', default='train_data_new1.csv', type=str, help='new train file name to write')

    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model.pt', type=str, help='model file name')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data_basic_stats.csv', type=str, help='test file name')

    parser.add_argument('--max_seq_len', default=20, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    # wandb
    if mode == 'train':
        parser.add_argument('--prefix', default='jh', required=True, type=str, help='prefix of wandb run name (e.g. username or initials).')
        parser.add_argument('--run_name', default='test', type=str, help='wandb run name. Defaults to current time')

    # 모델
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=8, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.1, type=float, help='drop out rate')

    # Optimizer
    parser.add_argument('--optimizer', default='adam', type=str, choices=choices["optimizer_options"],
                        help='optimizer type')

    # Optimizer-parameters
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay of optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum for SGD")

    # Scheduler
    parser.add_argument('--scheduler', default='plateau', type=str, choices=choices["scheduler_options"], help='scheduler type')

    # Scheduler-parameters
    #plateau
    parser.add_argument('--plateau_patience', default=10, type=int, help='patience of plateau scheduler')
    parser.add_argument('--plateau_factor', default=0.5, type=float, help='factor of plateau scheduler')

    #cosine anealing
    parser.add_argument('--t_max', default=10, type=int, help='cosine annealing scheduler: t max')
    parser.add_argument('--T_0', default=10, type=int, help='cosine annealing warm start scheduler: T_0')
    parser.add_argument('--T_mult', default=2, type=int, help='cosine annealing warm start scheduler: T_mult')
    parser.add_argument('--eta_min', default=0.01, type=float, help='cosine annealing warm start scheduler: eta_min')

    #linear_warmup
    parser.add_argument('--warmup_ratio', default=0.3, type=float, help='warmup step ratio')

    #Step LR
    parser.add_argument('--step_size', default=50, type=int, help='step LR scheduler: step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='step LR scheduler: gamma')

    # 훈련
    parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=10, type=int, help='for early stopping')

    # cv
    parser.add_argument('--kfold_num', default=5, type=int, help='number of fold')
    parser.add_argument('--cv_strategy', default=None, type=str, choices=choices["cv_options"], help='cross validation method')


    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')

    #sampling
    parser.add_argument('--reservoir_sampling', default='False', type=str2bool, help='whether do reservoir sampling or not')

    ### 중요 ###
    parser.add_argument('--model', default='lstmattn', type=str, choices=choices["model_options"], help='model type')
    parser.add_argument('--criterion', default='BCE', type=str, choices=choices["criterion_options"], help='criterion type')

    # aumentation option
    parser.add_argument('--augmentation', default='False', type=str2bool, help='whether do augment data or not')
    parser.add_argument('--aug_shuffle_n', default=0, type=int, help='')

    args = parser.parse_args()

    return args