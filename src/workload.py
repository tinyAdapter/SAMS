import os

from torch.utils.data import Dataset,Subset

class Workload(object):

    def __init__(self, workload_dir:str, data:Dataset):
        self.workload_dir = workload_dir
        self.data = data

        sql_file_path = os.path.join(workload_dir, 'sql.txt')
        self.sqls = Workload.parse_sql_file(sql_file_path)
        self.n = len(self.sqls)


    def parse_sql_file(sql_file_path:str):
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sqls = [eval(line.split(':')[1]) for line in f.readlines()]
        return sqls


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        '''return a sql embedding and filtered subset dataset according to this embedding
        '''

        data_idx_file_path = os.path.join(self.workload_dir, "data_idx" ,"data_idx_%d.txt" % idx)
        sub_indices = []
        with open(data_idx_file_path, 'r', encoding='utf') as f:
            sub_indices = eval(f.read())

        return self.sqls[idx], Subset(self.data, sub_indices)



if __name__ == '__main__':
    print("Test Workload")
    test_file_path = "/home/huangdan/SAMS/third_party/data/frappe/test.libsvm"
    nfield = 10
    m = 4
    
    from data_loader import SQLAwareDataset
    test_dataset = SQLAwareDataset(test_file_path,nfield, m)
    
    workload_dir = "/home/huangdan/SAMS/workload/frappe_test"
    workload = Workload(workload_dir, test_dataset)
    
    for (sql, subset) in workload:
        print(f"sql :{sql}")
        print(f"subset len {len(subset)}")
        for (fid, fvalue, y) in subset:
            print(f"fid : {fid}")
            print(f"fvalue : {fvalue}")
            print(f"y : {y}")
            exit(0)
    