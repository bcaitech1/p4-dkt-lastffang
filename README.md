# P Stage 4 DKT

## 프로젝트 개요

**DKT 프로젝트 소개**

KT는 **Knowledge Tracing**으로, 학생들의 지난 교육 기록을 활용하여, 아직 풀지 않는 문제에 대해, 학생이 그 문제를 맞출지 예측하는 Task입니다.

DKT는 **Deep Knowledge Tracing**으로 KT에 Deep Learning 기술을 적용한 것으로, 지식 상태를 추적하는 딥러닝 방법입니다.

DKT를 활용하면, 학생 개개인의 학습상태 예측이 가능해지고, 이를 기반으로 학생의 부족한 영역에 대한 문제 추천을 함으로써, 개개인별 맞춤화된 교육을 제공해줄 수 있습니다.

DKT는 점점 디지털화 되가는 교육 환경에 중요한 기술로 떠오르고 있습니다.

저희 프로젝트는 7,442명에 대한 문제 풀이와 2,266,586개의 문제 데이터를 이용해, 시험지의 마지막 대한 문항에 대해, **이 학생이 문제를 맞출지, 아닐지 예측하는 모델을 구축**하는 것입니다.

<p align="center"><img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/629369db-fb87-4cf7-9d9e-ce990c2d537a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T115912Z&X-Amz-Expires=86400&X-Amz-Signature=b675367905c2667713d4517c540c17f2d93bee2b1f4cf36ea4bb1857c8c3e4fd&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22"></p>

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
* 데이터 생성

1. `jupyter/pre-fe.ipynb` 를 수행해서 미리 계산해야 하는 feature 들을 포함한 데이터를 생성
2. 생성된 파일 이름으로 `args` 의 `train_file_to_load`, `test_file_name` 를  설정

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
