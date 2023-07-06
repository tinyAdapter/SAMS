
import model
import time
import utils
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from singleton import logger


class CombinedModel:

    def __init__(self, args, sql_combinations: int):
        """
        :param args: all parameters
        :param sql_combinations:  # of combinations
        """
        self.args = args

        # declarition
        self.hyper_net = None
        self.moe_net = None

        self.total_combinations = sql_combinations

        self.construct_model()

    def construct_model(self):
        self.hyper_net = model.HyperNet(
            nfield=1,
            nfeat=self.total_combinations,
            nemb=self.args.sql_nemb,
            nlayers=self.args.num_layers,
            hid_layer_len=self.args.hid_layer_len,
            dropout=self.args.dropout,
            L=self.args.moe_num_layers,
            K=self.args.K,
        )

        self.moe_net = model.MOENet(
            nfield=self.args.nfield,
            nfeat=self.args.nfeat,
            nemb=self.args.data_nemb,
            moe_num_layers=self.args.moe_num_layers,
            moe_hid_layer_len=self.args.moe_hid_layer_len,
            dropout=self.args.dropout,
        )

    def trian(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, use_test_acc=True):

        start_time, best_valid_auc = time.time(), 0.

        # define the loss function, for two class classification
        opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(self.args.device)

        # define the parameter
        params = list(self.hyper_net.parameters()) + list(self.moe_net.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.lr)

        # scheduler to update the learning learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epoch_num,  # Maximum number of iterations.
            eta_min=1e-4)  # Minimum learning rate.

        # gradient clipping, set the gradient value to be -1 - 1 to prevent exploding gradients
        for p in params:
            p.register_hook(lambda grad: torch.clamp(grad, -1., 1.))

        info_dic = {}
        valid_auc = -1
        test_auc = -1
        # train for epoches.
        for epoch in range(self.args.epoch_num):
            logger.info(f'Epoch [{epoch:3d}/{self.args.epoch_num:3d}]')

            train_auc, train_loss = self.run( epoch, train_loader, opt_metric, optimizer=optimizer, namespace='train')
            scheduler.step()

            valid_auc, valid_loss = self.run(epoch, val_loader, opt_metric, namespace='val')

            if use_test_acc:
                test_auc, test_loss = self.run(epoch, test_loader, opt_metric, namespace='test')
            else:
                test_auc = -1

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
        if optimizer:
            self.hyper_net.train()
            self.moe_net.train()
        else:
            self.hyper_net.eval()
            self.moe_net.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        for batch_idx, batch in enumerate(data_loader):
            # if suer set this, then only train fix number of iteras
            # stop training current epoch for evaluation
            if namespace == 'train' and self.args.iter_per_epoch is not None and batch_idx >= self.args.iter_per_epoch:
                break

            sql_id = batch['sql'].to(self.args.device)
            target = batch['y'].to(self.args.device)
            batch['id'] = batch['id'].to(self.args.device)
            batch['value'] = batch['value'].to(self.args.device)

            if namespace == 'train':

                # 1. get the arch_advisor B*L*K
                arch_advisor = self.hyper_net(sql_id)
                assert arch_advisor.size() == (self.args.batch_size, self.args.moe_num_layers * self.args.K)

                # reshape it to (B, L, K)
                arch_advisor = arch_advisor.reshape(self.args.batch_size, self.args.moe_num_layers, self.args.K)

                # todo: conduct sparse softmax
                arch_advisor = arch_advisor

                # calculate y and loss
                y = self.moe_net.forward(batch, arch_advisor)
                loss = opt_metric(y, target)
                optimizer.zero_grad()

                # one step to update both hypernet and moenet
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    arch_advisor = self.hyper_net(sql_id)
                    assert arch_advisor.size() == (self.args.batch_size, self.args.moe_num_layers * self.args.K)
                    # reshape it to (B, L, K)
                    arch_advisor = arch_advisor.reshape(self.args.batch_size, self.args.moe_num_layers, self.args.K)
                    # calculate y and loss
                    y = self.moe_net.forward(batch, arch_advisor)
                    loss = opt_metric(y, target)

            # for multiple classification
            # auc = utils.roc_auc_compute_fn(torch.nn.functional.softmax(y, dim=1)[:, 1], target)
            auc = utils.roc_auc_compute_fn(y, target)
            loss_avg.update(loss.item(), target.size(0))
            auc_avg.update(auc, target.size(0))

            time_avg.update(time.time() - timestamp)
            timestamp = time.time()
            if batch_idx % self.args.report_freq == 0:
                logger.info(f'Epoch [{epoch:3d}/{self.args.epoch_num}][{batch_idx:3d}/{len(data_loader)}]\t'
                            f'{time_avg.val:.3f} ({time_avg.avg:.3f}) AUC {auc_avg.val:4f} ({auc_avg.avg:4f}) '
                            f'Loss {loss_avg.val:8.4f} ({loss_avg.avg:8.4f})')

        logger.info(f'{namespace}\tTime {utils.timeSince(s=time_avg.sum):>12s} '
                    f'AUC {auc_avg.avg:8.4f} Loss {loss_avg.avg:8.4f}')
        return auc_avg.avg, loss_avg.avg





