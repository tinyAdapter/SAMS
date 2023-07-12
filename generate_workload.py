import argparse
import os
import glob
import random
import numpy as np

from tqdm import tqdm
from typing import Callable, List
from src.data_loader import SQLAttacedLibsvmDataset, SQLAwareDataset

random.seed(1998)

'''
Generate Workload
'''

pwd = os.getcwd()


parser = argparse.ArgumentParser(description='wordload_generation')
parser.add_argument('--output_dir', type=str, default='./workload/frappe_test')
parser.add_argument('--data_dir', type=str,
                    default='./third_party/data/frappe', help="")
parser.add_argument('--nfield', type=int, default=10, help="")
parser.add_argument('--n', type=int, default=30,
                    help="number of sql in workload")
parser.add_argument('--max_select_col', type=int, default=4,
                    help="max selected column number for filter")


def generate_workload(n: int, output_dir: str, data_dir: str, nfield: int, max_select_col: int, sample_func: Callable):

    # read train dataset get dict configuration
    train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]
    train_dataset = SQLAttacedLibsvmDataset(train_file, nfield, max_select_col)

    # test dataset
    test_file = glob.glob("%s/te*libsvm" % data_dir)[0]
    test_dataset = SQLAwareDataset(test_file, nfield, max_select_col)

    value_map = train_dataset.col_cardinalities
    col_padding = train_dataset.padding_feature_id

    feat_id = test_dataset.feat_id.numpy()

    sql_file_path = os.path.join(output_dir, 'sql.txt')

    sqlf = open(sql_file_path, 'w', encoding='utf-8')

    data_dir_path = os.path.join(output_dir, 'data_idx')
    os.mkdir(data_dir_path)
    
    i = 0
    while i < n:
        sql_query, padding_tuple = sample_func(value_map, max_select_col)
        sub_data_idx = filter_data(feat_id, sql_query, col_padding)

        if len(sub_data_idx) == 0:
            continue

        data_idx_file_path = os.path.join(data_dir_path, 'data_idx_%d.txt' % i)
        size = len(sub_data_idx)
        with open(data_idx_file_path, 'w', encoding='utf-8') as dataf:
            dataf.write(str(sub_data_idx))

        sqlf.write(str(size) +":"+str(padding_tuple) + ":" + str(sql_query) + '\n')
        i += 1

    dataf.close()
    sqlf.close()


def filter_data(data: np.array, sql_query: List, col_padding: List):
    filter_condition = None
    for i, col_value in enumerate(sql_query):
        if col_value == col_padding[i]:
            continue
        if filter_condition is None:
            filter_condition = (data[:, i] == col_value)
        else:
            filter_condition &= (data[:, i] == col_value)

    sub_data_idx = np.where(filter_condition)[0]
    return sub_data_idx.tolist()


def random_sample(col_cardinalities: List[List], max_select_col: int):
    col_num = len(col_cardinalities)
    max_select_col = max(col_num, max_select_col)

    select_col_n = random.randint(0, max_select_col)

    # all padding
    sql_tuple = [col[-1] for col in col_cardinalities]
    padding_tuple = [0] * len(col_cardinalities)

    # random choice select_col_n colomn
    cols = random.sample(range(0, col_num), select_col_n)

    for col in cols:
        col_unique_list = col_cardinalities[col]
        v = random.choice(col_unique_list[:-1])  # don't include padding
        sql_tuple[col] = v
        padding_tuple[col] = 1
    return sql_tuple, padding_tuple


if __name__ == "__main__":
    args = parser.parse_args()
    sample_func = random_sample

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generate_workload(args.n, args.output_dir, args.data_dir,
                      args.nfield, args.max_select_col, sample_func)
    print(f"Finished Generate Workload, saved in {args.output_dir}")
