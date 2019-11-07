import argparse
from transformer.Constants import SQL_SEPARATOR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--tables', required=True)
    # path without extension !
    parser.add_argument('--output', required=True)

    opt = parser.parse_args()

    # TODO extract en query (question in jsonl file)
    # TODO extract sql query (sql in jsonl file converted like in query.py with column names
    # from header in tables jsonl)
    # TODO add separators between sql tokens
    # find appropriate sql token not included in dataset
    # TODO save them to csvs as file.en and file.sql


if __name__ == '__main__':
    main()
