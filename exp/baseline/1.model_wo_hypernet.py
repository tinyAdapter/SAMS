
from torch import optim
from torch.utils.data import DataLoader
import argparse
import calendar
import os
import time
import traceback


class MOEOnlyRuntime:

    def __init__(self, args):
        """
        :param args: all parameters
        """
        self.args = args

        # declarition model
        self.hyper_net = None
        self.moe_net = None
        self.sparsemax = None

        self.construct_model()

    def construct_model(self):
        self.moe_net = MOENet(
            nfield=self.args.nfield,
            nfeat=self.args.nfeat,
            nemb=self.args.data_nemb,
            moe_num_layers=self.args.moe_num_layers,
            moe_hid_layer_len=self.args.moe_hid_layer_len,
            dropout=self.args.dropout,
            K=self.args.K,
        )

    def trian(self,
              train_loader: DataLoader, val_loader: SQLAwareDataset, test_loader: SQLAwareDataset,
              use_test_acc=True):
        """
        :param train_loader: data loaer
        :param val_loader: data loaer
        :param test_loader: data loaer
        :param use_test_acc: if use test dataset to valid during trianing.
        :return:
        """

        start_time, best_valid_auc = time.time(), 0.

        # define the loss function, for two class classification
        opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(self.args.device)

        # define the parameter, which includes both networks
        params = list(self.moe_net.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr)

        # scheduler to update the learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epoch,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1 to prevent exploding gradients
        for p in params:
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        # records running info
        info_dic = {}
        valid_auc = -1
        test_auc = -1
        # train for epoches.
        for epoch in range(self.args.epoch):
            logger.info(f'Epoch [{epoch:3d}/{self.args.epoch:3d}]')

            # 1. train
            train_auc, train_loss = self.run(epoch, train_loader, opt_metric, optimizer=optimizer, namespace='train')
            scheduler.step()

            # 2. valid with valid
            # update val_loader's history
            val_loader.sql_history = train_loader.dataset.sql_history
            valid_auc, valid_loss = self.run(epoch, val_loader, opt_metric, namespace='val')

            # 3. valid with test
            if use_test_acc:
                test_loader.sql_history = train_loader.dataset.sql_history
                test_auc, test_loss = self.run(epoch, test_loader, opt_metric, namespace='test')
            else:
                test_auc = -1

            # set to empty for the next epoch
            train_loader.dataset.reset_sql_his()

            info_dic[epoch] = {
                "train_auc": train_auc,
                "valid_auc": valid_auc,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_val_total_time": time.time() - start_time}

        if valid_auc >= best_valid_auc:
            best_valid_auc, best_test_auc = valid_auc, test_auc
            logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
        else:
            logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

    #  train one epoch of train/val/test
    def run(self, epoch, data_loader, opt_metric, optimizer=None, namespace='train'):
        """
        :param epoch:
        :param data_loader:
        :param opt_metric: the loss function
        :param optimizer:
        :param namespace: train | valid | test
        :return:
        """

        if optimizer:
            self.hyper_net.train()
            self.moe_net.train()
        else:
            self.hyper_net.eval()
            self.moe_net.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        if namespace == 'train':

            for batch_idx, data_batch in enumerate(data_loader):
                if namespace == 'train' \
                        and self.args.iter_per_epoch is not None \
                        and batch_idx >= self.args.iter_per_epoch:
                    break
                sql_batch_tensor = data_batch["sql"].to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                data_batch['id'] = data_batch['id'].to(self.args.device)
                data_batch['value'] = data_batch['value'].to(self.args.device)

                # Create tensor of size (L, K) where each element is 1/K
                base_tensor = torch.full((self.args.moe_num_layers, self.args.K), 1.0 / self.args.K)
                # Replicate this tensor B times to create final tensor
                arch_advisor = base_tensor.unsqueeze(0).repeat(self.args.batch_size, 1, 1)

                # calculate y and loss
                y = self.moe_net.forward(data_batch, arch_advisor)
                loss = opt_metric(y, target)
                optimizer.zero_grad()

                # one step to update both hypernet and moenet
                loss.backward()
                optimizer.step()

                # logging
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))

                time_avg.update(time.time() - timestamp)
                if batch_idx % self.args.report_freq == 0:
                    logger.info(f'Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(data_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')
        else:

            mini_batches = data_loader.sample_all_data_by_sql(self.args.batch_size)

            for batch_idx, (sql_batch, data_batch) in enumerate(mini_batches):
                sql_batch_tensor = torch.tensor(sql_batch).to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                data_batch['id'] = data_batch['id'].to(self.args.device)
                data_batch['value'] = data_batch['value'].to(self.args.device)

                with torch.no_grad():
                    arch_advisor = self.hyper_net(sql_batch_tensor)
                    assert arch_advisor.size(1) == self.args.moe_num_layers * self.args.K, \
                        f"{arch_advisor.size()} is not {self.args.batch_size, self.args.moe_num_layers * self.args.K}"

                    # Create tensor of size (L, K) where each element is 1/K
                    base_tensor = torch.full((self.args.moe_num_layers, self.args.K), 1.0 / self.args.K)
                    # Replicate this tensor B times to create final tensor
                    arch_advisor = base_tensor.unsqueeze(0).repeat(arch_advisor.size(0), 1, 1)

                    # calculate y and loss
                    y = self.moe_net.forward(data_batch, arch_advisor)
                    loss = opt_metric(y, target)

                # logging
                auc = utils.roc_auc_compute_fn(y, target)
                loss_avg.update(loss.item(), target.size(0))
                auc_avg.update(auc, target.size(0))

                time_avg.update(time.time() - timestamp)
                if batch_idx % self.args.report_freq == 0:
                    logger.info(f'Epoch [{epoch:3d}/{self.args.epoch}][{batch_idx:3d}/{len(data_loader)}]\t'
                                f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                                f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')
        return auc_avg.avg, loss_avg.avg


def arch_args(parser):
    # MOE-NET
    parser.add_argument('--K', default=4, type=int,
                        help='# duplication layer of each MOElayer')
    parser.add_argument('--moe_num_layers', default=5,
                        type=int, help='# hidden MOElayers of MOENet')
    parser.add_argument('--moe_hid_layer_len', default=10,
                        type=int, help='hidden layer length in MoeLayer')

    # embedding layer
    parser.add_argument('--data_nemb', type=int,
                        default=10, help='embedding size 10')
    # other setting
    parser.add_argument('--dropout', default=0.0,
                        type=float, help='dropout rate')


def trainner_args(parser):

    parser.add_argument('--alpha', default=0.1, type=float,
                        help='entmax alpha to control sparsity')

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='system')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_folder', default="baseline1", type=str)
    parser.add_argument('--log_name', type=str,
                        default="run_log", help="file name to store the log")

    arch_args(parser)
    data_set_config(parser)
    trainner_args(parser)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault(
        "log_file_name", f"{args.log_name}_{args.dataset}_{ts}.log")

    from src.singleton import logger
    from src.data_loader import *
    from src.model import *
    import third_party.utils.func_utils as utils

    try:
        # 1. data loader
        train_loader, val_loader, test_loader = libsvm_dataloader(args=args)
        model = MOEOnlyRuntime(args=args)
        model.trian(train_loader, val_loader, test_loader)

    except:
        logger.info(traceback.format_exc())

