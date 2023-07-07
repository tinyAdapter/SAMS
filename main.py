
import argparse
import src.data_loader as data
import src.run_time as runtime


def arch_args(parser):

    # MOE-NET
    parser.add_argument('--K', default=4, type=int, help='# duplication layer of each MOElayer')
    parser.add_argument('--moe_num_layers', default=4, type=int, help='# hidden MOElayers of MOENet')
    parser.add_argument('--moe_hid_layer_len', default=10, type=int, help='hidden layer length in MoeLayer')

    # hyperNet
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers of hyperNet')
    parser.add_argument('--hid_layer_len', default=10, type=int, help='hidden layer length in hyerNet')

    # embedding layer
    parser.add_argument('--data_nemb', type=int, default=10, help='embedding size 10')
    parser.add_argument('--sql_nemb', type=int, default=10, help='embedding size 10')

    # other setting
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')


def trainner_args(parser):

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

    parser.add_argument('--epoch', type=int, default=9,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 40, criteo: 10'
                             )

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')

    parser.add_argument('--iter_per_epoch', type=int, default=2000,
                        help="200 for frappe, uci_diabetes, 2000 for criteo")

    # MLP train config
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')


def data_set_config(parser):
    parser.add_argument('--data_dir', type=str, default="../exp_data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='uci_diabetes',
                        help='cifar10, cifar100, ImageNet16-120, '
                             'frappe, '
                             'criteo, '
                             'uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')

    parser.add_argument('--workers', default=1, type=int, help='worker in data loader')


def parse_arguments():
    parser = argparse.ArgumentParser(description='system')

    # job config
    parser.add_argument('--log_name', type=str, default="main_T_100s", help="file name to store the log")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_folder', default="LogCriteo", type=str, help='num GPus')

    arch_args(parser)
    data_set_config(parser)
    trainner_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # init data loader
    train_loader, val_loader, test_loader, total_columns, col_cardinality_sum = data.libsvm_dataloader(args=args)
    # init model
    model = runtime.CombinedModel(
        args=args,
        total_columns=total_columns,
        col_cardinality_sum=col_cardinality_sum)

    model.trian(train_loader, val_loader, test_loader)

    print("Done")
