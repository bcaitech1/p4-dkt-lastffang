import json
import argparse
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="cfg", type=str, default='configs/default.json')

args = parser.parse_args()

with open(args.cfg) as f:
    cfg = json.load(f)


arguments = []

for k in cfg:
    arguments.append("--" + k)
    arguments.append(str(cfg[k]))

# Train Model
call(["python", "train.py"] + arguments)

# Inference file
call(["python", "inference.py"] + arguments)
