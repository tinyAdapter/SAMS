import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import calendar
import os
import time
import traceback


# simpel DNN
class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """

    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout, noutput):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.mlp_ninput = nfield * nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid, dropout, noutput)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding.forward_moe(x)  # B*F*E
        y = self.mlp(x_emb.view(-1, self.mlp_ninput))  # B*1
        return y.squeeze(1)


class ModelTrainer:

    @classmethod
    def fully_train_arch(cls,
                         model: nn.Module,
                         use_test_acc: bool,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader: DataLoader,
                         args,
                         logger=None
                         ) -> (float, float, dict):

        start_time, best_valid_auc = time.time(), 0.

        # training params
        device = args.device
        num_labels = args.num_labels
        lr = args.lr
        iter_per_epoch = args.iter_per_epoch

        opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        info_dic = {}
        valid_auc = -1
        for epoch in range(args.epoch):
            logger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]')
            # train and eval
            # print("begin to train...")
            train_auc, train_loss = ModelTrainer.run(logger,
                                                     epoch, iter_per_epoch, model, train_loader, opt_metric, args,
                                                     optimizer=optimizer, namespace='train')
            scheduler.step()

            # print("begin to evaluate...")
            valid_auc, valid_loss = ModelTrainer.run(logger,
                                                     epoch, iter_per_epoch, model, val_loader,
                                                     opt_metric, args, namespace='val')

            if use_test_acc:
                test_auc, test_loss = ModelTrainer.run(logger,
                                                       epoch, iter_per_epoch, model, test_loader,
                                                       opt_metric, args, namespace='test')
            else:
                test_auc = -1

            info_dic[epoch] = {
                "train_auc": train_auc,
                "valid_auc": valid_auc,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_val_total_time": time.time() - start_time}

            # record best auc and save checkpoint
            if valid_auc >= best_valid_auc:
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
            else:
                logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

        return valid_auc, time.time() - start_time, info_dic

    #  train one epoch of train/val/test
    @classmethod
    def run(cls, logger, epoch, iter_per_epoch, model, data_loader, opt_metric, args, optimizer=None,
            namespace='train'):
        if optimizer:
            model.train()
        else:
            model.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        for batch_idx, batch in enumerate(data_loader):
            if namespace == 'train' and iter_per_epoch is not None and batch_idx >= iter_per_epoch:
                break

            target = batch['y'].to(args.device)
            batch['id'] = batch['id'].to(args.device)
            batch['value'] = batch['value'].to(args.device)

            if namespace == 'train':
                y = model(batch)
                loss = opt_metric(y, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y = model(batch)
                    loss = opt_metric(y, target)

            auc = utils.roc_auc_compute_fn(y, target)
            loss_avg.update(loss.item(), target.size(0))
            auc_avg.update(auc, target.size(0))

            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            if batch_idx % args.report_freq == 0:
                logger.info(f'Epoch [{epoch:3d}/{args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                            f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                            f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')
        return auc_avg.avg, loss_avg.avg


def arch_args(parser):
    parser.add_argument('--num_layers', default=4, type=int,
                        help='# hidden layers of hyperNet')
    parser.add_argument('--hid_layer_len', default=10,
                        type=int, help='hidden layer length in hyerNet')

    # embedding layer
    parser.add_argument('--data_nemb', type=int,
                        default=10, help='embedding size 10')
    # other setting
    parser.add_argument('--dropout', default=0.0,
                        type=float, help='dropout rate')


def trainner_args(parser):
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
    parser.add_argument('--log_folder', default="baseline2", type=str)
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
    from src.data_loader import libsvm_dataloader
    from src.model import *
    import third_party.utils.func_utils as utils

    try:
        # read the checkpoint

        # 1. data loader
        train_loader, val_loader, test_loader = libsvm_dataloader(args=args)

        model = DNNModel(
            nfield=args.nfield,
            nfeat=args.nfeat,
            nemb=args.data_nemb,
            mlp_layers=args.num_layers,
            mlp_hid=args.hid_layer_len,
            dropout=args.dropout,
            noutput=1
        )

        valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
            model=model,
            use_test_acc=False,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args)
    except:
        logger.info(traceback.format_exc())
