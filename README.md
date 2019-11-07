# Installation

Install pip requirements:

```bash
pip3 install -r requirements.txt
```

Install spacy en model:

```bash
python3 -m spacy download en
```

# Training

Run training on local machine:

```bash
python3 train.py --train_data dataset.dev --validation_data dataset.dev --no_cuda
```

# TODO

1. extract en sentence to csv
2. extract sql metadata and transform them into csv (custom separator for better tokenization ?)
8. train/validate
9. create cli? tool for translation
