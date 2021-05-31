# P Stage 4 DKT

## 코드 구조
```bash
code
├── README.md
├── args.py
├── baseline.ipynb
├── dkt
│  ├── criterion.py
│  ├── dataloader.py
│  ├── metric.py
│  ├── model.py
│  ├── optimizer.py
│  ├── scheduler.py
│  ├── trainer.py
│  └── utils.py
├── inference.py
├── requirements.txt
└── train.py
```

## Model, Optimizer, Scheduler, Criterion 추가 방법
### Model
1. `dkt/model.py`에 구현한 모델 추가
2. `dkt/trainer.py`의 `get_model` 함수에 모델을 불러오는 코드 작성
3. `arg_choices.json`의 `model_options`에 해당 모델을 불러오는 argument value 추가


### Optimizer
1. `dkt/optimizer.py`에 새로운 optimizer 추가
2. `arg_choices.json`의 `optimizer_options`에 해당 optimizer를 불러오는 argument value 추가

### Scheduler
1. `dkt/scheduler.py`에 새로운 scheduler 추가
2. `arg_choices.json`의 `scheduler_options`에 해당 scheduler를 불러오는 argument value 추가

### Criterion
1. `dkt/criterion.py`에 새로운 criterion 추가
2. `arg_choices.json`의 `criterion_options`에 해당 criterion를 불러오는 argument value 추가


## 코드 실행 방법
* 모델 학습
```bash
python train.py --prefix prefix --run_name run_name --model_dir models/model_dir
```

* Inference
```bash
python inference.py --model_dir models/model_dir
```

* Train & Inference를 한 번에 실행하기
```bash
python run_with_json.py --config json_file_path
```
