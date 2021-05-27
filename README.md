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
