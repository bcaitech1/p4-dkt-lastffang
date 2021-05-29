import json
import argparse
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="cfg", type=str, default='configs/default.json')

args = parser.parse_args()

with open(args.cfg) as f:
    cfg = json.load(f)


train_arguments = []
inf_arguments = []

for k in cfg:
    train_arguments.append("--" + k)
    train_arguments.append(cfg[k])
    if k == 'prefix' or k == 'run_name':
        continue
    inf_arguments.append("--" + k)
    inf_arguments.append(cfg[k])

# Train Model
call(["python", "train.py"] + train_arguments)

# Inference file
call(["python", "inference.py"] + inf_arguments)
