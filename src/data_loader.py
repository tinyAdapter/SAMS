import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import random


class SQLAwareDataset(Dataset):
    """ Dataset loader for Libsvm data format """
    def __init__(self, fname, nfields, max_filter_col):

        def decode_libsvm(line):
            columns = line.split(' ')
            map_func = lambda pair: (int(pair[0]), float(pair[1]))
            id, value = zip(*map(lambda col: map_func(col.split(':')), columns[1:]))
            sample = {'id': torch.LongTensor(id),
                      'value': torch.FloatTensor(value),
                      'y': float(columns[0])}
            return sample

        with open(fname) as f:
            sample_lines = sum(1 for line in f)

        self.feat_id = torch.LongTensor(sample_lines, nfields)
        self.feat_value = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)

        self.nsamples = 0
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_id[self.nsamples] = sample['id']
                        self.feat_value[self.nsamples] = sample['value']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')

        # generate the columsn statics
        self.max_columns = min(self.feat_id.shape[1], max_filter_col)

        # Convert the tensor to a list of lists, where each inner list contains the unique values from each column
        self.col_cardinalities = [self.feat_id[:, i].unique().tolist() for i in range(self.feat_id.shape[1])]

        # this is used in infernece, as infernece must testd on the trained sql.
        self.sql_history = set()

    def sample_all_data_by_sql(self):
        """
        This is for infernece,
        :param sql_his: sql history at training stage.
        :return:
        """
        all_sql = []
        all_data = []

        for sql in self.sql_history:
            data_dic = self.select_row_with_sql(sql)
            # if the sql don;t have matched data, continue to next.
            if data_dic["id"] is None:
                continue
            # add to all_data.
            all_sql.append(sql)
            all_data.append(data_dic)

        return all_sql, all_data

    def sample_batch_sql_and_data(self, batch_size: int):
        """
        Sample a batch of sql and corresponding data.
        :param batch_size:
        :return:
        """
        sql_batch = []
        data_batch = []
        while True:
            # sample
            sql = self.sample_sql()
            data_dic = self.select_row_with_sql(sql)

            # if the sql don;t have matched data, continue to next.
            if data_dic["id"] is None:
                continue

            # add to batch data
            sql_batch.append(sql)
            data_batch.append(data_dic)

            # add to his for inference testing
            self.sql_history.add(sql)
            if len(sql_batch) >= batch_size:
                break

        return sql_batch, data_batch

    def sample_sql(self) -> tuple:
        # 1. firstly randomly pick number of the columns
        ncol = random.randint(1, self.max_columns)
        # initialize all columns to cardinalities + 1 (index)
        random_sql = [len(col) for col in self.col_cardinalities]
        # 2. second, randomly find one value per column to form a sql.
        random_columns_idx = random.sample(range(self.max_columns), ncol)
        for idx in random_columns_idx:
            random_sql[idx] = random.choice(self.col_cardinalities[idx])
        return tuple(random_sql)

    def select_row_with_sql(self, random_sql: tuple) -> dict:
        cols_to_check = torch.tensor(random_sql)
        matching_rows = (self.feat_id[:, :len(cols_to_check)] == cols_to_check).all(dim=1)

        # Check if there is at least one match
        if matching_rows.any():
            # Get the index of the first matching row
            first_match_index = matching_rows.nonzero(as_tuple=True)[0][0]

            selected_row_feat_id = self.feat_id[first_match_index, :].squeeze()
            selected_row_feat_value = self.feat_value[first_match_index, :].squeeze()
            selected_row_y = self.y[first_match_index, :].squeeze()
        else:
            selected_row_feat_id = None
            selected_row_feat_value = None
            selected_row_y = None
        return {'id': selected_row_feat_id, 'value': selected_row_feat_value, 'y': selected_row_y}

    def reset_sql_his(self):
        self.sql_history.clear()


def sql_dataloader(args):
    data_dir = args.data_dir + args.dataset
    train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]
    val_file = glob.glob("%s/va*libsvm" % data_dir)[0]
    test_file = glob.glob("%s/te*libsvm" % data_dir)[0]

    train_loader = SQLAwareDataset(train_file, args.nfield, args.max_filter_col)
    val_loader = SQLAwareDataset(val_file, args.nfield, args.max_filter_col)
    test_loader = SQLAwareDataset(test_file, args.nfield, args.max_filter_col)

    return train_loader, val_loader, test_loader, -1
