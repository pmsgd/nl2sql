# Installation

Install pip requirements:

```bash
pip3 install -r requirements.txt
```

Install spacy en model:

```bash
python3 -m spacy download en
```

# Data preprocessing

Run preprocessing python script for all datasets:

```bash
mkdir inputsets outputsets models

python3 preprocess.py --queries ../WikiSQL/data/dev.jsonl --tables ../WikiSQL/data/dev.tables.jsonl --output inputsets/dev
python3 preprocess.py --queries ../WikiSQL/data/train.jsonl --tables ../WikiSQL/data/train.tables.jsonl --output inputsets/train
python3 preprocess.py --queries ../WikiSQL/data/test.jsonl --tables ../WikiSQL/data/test.tables.jsonl --output inputsets/test
```

# Training

Run training on local machine (with --no_cuda if CUDA enabled GPU is missing):

```bash
python3 train.py --train_data inputsets/train --validation_data inputsets/test --save_model models/model
```

# Translation

Run translation on local machine (with --no_cuda if CUDA enabled GPU is missing):

```bash
python3 translate.py --src inputsets/dev.en --model models/model
```
