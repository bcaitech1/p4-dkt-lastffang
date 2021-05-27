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
├── evaluation.py
├── inference.py
├── requirements.txt
└── train.py
```

## 코드 실행 방법
* Train & Inference를 한 번에 실행하기
```bash
python run_with_json.py --config json_file_path
```
