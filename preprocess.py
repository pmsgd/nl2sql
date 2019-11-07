import argparse
import json
from transformer.Constants import SQL_SEPARATOR

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

queryset = []
tableset = []
table_dict = {}

def process(sentence_file, sql_file):
    for rec in queryset:
        rec_dict = json.loads(rec)
        sentence_file.write(rec_dict['question'].strip() + "\n")

        sql = []
        sql.append("select")

        agg = agg_ops[rec_dict['sql']['agg']]
        if agg != '':
            sql.append(agg)

        table_id = rec_dict['table_id']
        column = table_dict[table_id][rec_dict['sql']['sel']]
        sql.append(column)

        sql.append('from table')
        sql.append('where')

        conds = rec_dict['sql']['conds']
        transf_conds = []
        for c in conds:
            transf_cond = []
            transf_cond.append(table_dict[table_id][c[0]])
            transf_cond.append(cond_ops[c[1]])
            transf_cond.append(str(c[2]))
            transf_conds.append(SQL_SEPARATOR.join(transf_cond))
        sql.append("|AND|".join(transf_conds))

        sql_file.write(SQL_SEPARATOR.join(sql) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--tables', required=True)
    # path without extension !
    parser.add_argument('--output', required=True)

    opt = parser.parse_args()

    with open(opt.queries) as qfile:
        for line in qfile:
            queryset.append(line.rstrip())
    qfile.close()

    with open(opt.tables) as tbfile:
        for line in tbfile:
            tableset.append(line.rstrip())
    tbfile.close()

    for rec in tableset:
        rec_dict = json.loads(rec)
        table_dict[rec_dict['id']] = rec_dict['header']

    sentence_file = open(opt.output + ".en", "w+")
    sql_file = open(opt.output + ".sql", "w+")

    process(sentence_file, sql_file)

    sentence_file.close()
    sql_file.close()

if __name__ == '__main__':
    main()
