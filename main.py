import calendar
import os
import time
import argparse
import traceback
import warnings


def arch_args(parser):
    # MOE-NET
    parser.add_argument('--K', default=4, type=int,
                        help='# duplication layer of each MOElayer')
    parser.add_argument('--moe_num_layers', default=5,
                        type=int, help='# hidden MOElayers of MOENet')
    parser.add_argument('--moe_hid_layer_len', default=10,
                        type=int, help='hidden layer length in MoeLayer')

    # hyperNet
    parser.add_argument('--num_layers', default=4, type=int,
                        help='# hidden layers of hyperNet')
    parser.add_argument('--hid_layer_len', default=10,
                        type=int, help='hidden layer length in hyerNet')

    # embedding layer
    parser.add_argument('--data_nemb', type=int,
                        default=10, help='embedding size 10')
    parser.add_argument('--sql_nemb', type=int, default=10,
                        help='embedding size 10')

    # other setting
    parser.add_argument('--dropout', default=0.0,
                        type=float, help='dropout rate')


def trainner_args(parser):

    parser.add_argument('--alpha', default=0.1, type=float,
                        help='entmax alpha to control sparsity')

    parser.add_argument('--max_filter_col', type=int, default=4,
                        help='the number of columns to choose in select...where...')

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=369,
                        help='the number of features, '
                             'frappe: 5500, '
                             'uci_diabetes: 369,'
                             'criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=43,
                        help='the number of fields, '
                             'frappe: 10, '
                             'uci_diabetes: 43,'
                             'criteo: 39')

    parser.add_argument('--epoch', type=int, default=3,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 40, criteo: 10'
                        )

    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help="learning reate")

    parser.add_argument('--iter_per_epoch', type=int, default=10,
                        help="200 for frappe, uci_diabetes, 2000 for criteo")

    # MLP train config
    parser.add_argument('--report_freq', type=int,
                        default=30, help='report frequency')


def data_set_config(parser):
    parser.add_argument('--data_dir', type=str, default="./third_party/data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='uci_diabetes',
                        help='cifar10, cifar100, ImageNet16-120, '
                             'frappe, '
                             'criteo, '
                             'uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1, help='[2, 2, 2]')


def tensorboard_config(parser:argparse.ArgumentParser):
    parser.add_argument('--exp', type=str, default = 'tensor_log', help="the directory to store training tensorboard log")
    parser.add_argument('--train_dir', type=str, required=True, help="the name of this train process")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='system')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_folder', default="sams_logs", type=str)
    parser.add_argument('--log_name', type=str,
                        default="run_log", help="file name to store the log")

    arch_args(parser)
    data_set_config(parser)
    trainner_args(parser)
    tensorboard_config(parser)
    
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault(
        "log_file_name", f"{args.log_name}_{args.dataset}_{ts}.log")
    
    from src.singleton import logger
    import src.data_loader as data
    from src.tensorlog import setup_tensorboard_writer
    from src.run_time import CombinedModel
    
    # init data loader
    train_loader, val_loader, test_loader = data.sql_attached_dataloader(args=args)
    # val_loader, test_loader = data.sql_dataloader(args=args)

    # col_cardinality_sum + 1 for total feature ids.
    # TODO(Lingze) why not record it when loading data
    col_cardinality_sum = max(
        ele for sublist in train_loader.dataset.col_cardinalities for ele in sublist) + 1
    
    writer = setup_tensorboard_writer(args)
    # init model
    model = CombinedModel(
        args=args,
        writer=writer,
        col_cardinality_sum=col_cardinality_sum)

    model.trian(train_loader, val_loader, test_loader)

    print("Done")
    # except Exception as e:
    #     print(traceback.format_exc())
    #     logger.error("An error occurred: %s", e)
    #     logger.error("Traceback: %s", traceback.format_exc())
