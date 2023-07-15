
import time
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import third_party.utils.func_utils as utils
from src.singleton import logger
from src.data_loader import SQLAwareDataset

from src.model import initialize_model

class Wrapper(object):

    def __init__(self, args, writer:SummaryWriter):
        """
        :param args: all parameters
        """
        self.args = args
        self.writer = writer
        self.net = initialize_model(args)
        self.save_best_model = args.save_best_model
        
        
    def train(self,
              train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
              use_test_acc=True):
        """
        :param train_loader: data loaer
        :param val_loader: data loaer
        :param test_loader: data loaer
        :param use_test_acc: if use test dataset to valid during trianing.
        :return:
        """

        start_time, best_valid_auc, best_test_auc = time.time(), 0., 0.

        # define the loss function, for two class classification
        opt_metric = nn.BCEWithLogitsLoss(reduction='mean').to(self.args.device)

        # define the parameter, which includes both networks
        params = self.net.parameters()
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

        best_net = self.net
        best_epoch = 0
        # train for epoches.
        for epoch in range(self.args.epoch):
            logger.info(f'Epoch [{epoch:3d}/{self.args.epoch:3d}]')

            # 1. train
            train_auc, train_loss = self.run(epoch, train_loader, opt_metric, optimizer=optimizer, namespace='train')
            scheduler.step()

            # 2. valid with valid
            # update val_loader's history
            # logger.info(f'Training sql histry = {len(train_loader.dataset.sql_history)}')
            # val_loader.sql_history = train_loader.dataset.sql_history
            valid_auc, valid_loss = self.run(epoch, val_loader, opt_metric, namespace='val')

            # 3. valid with test
            if use_test_acc:
                # test_loader.sql_history = train_loader.dataset.sql_history
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
            
            self.writer.add_scalar('Loss/Training_Epoch_Ave_Loss', train_loss, epoch)
            self.writer.add_scalar('Loss/Valid_Epoch_Ave_Loss', valid_loss, epoch)
            self.writer.add_scalar('Loss/Test_Epoch_Ave_Loss', test_loss, epoch)
            
            
            self.writer.add_scalar('AUC/Train_Ave_AUC', train_auc, epoch)
            self.writer.add_scalar('AUC/Valid_Ave_AUC', valid_auc, epoch)
            self.writer.add_scalar('AUC/Test_Ave_AUC', test_auc, epoch)

            self.writer.flush()
            
            if valid_auc >= best_valid_auc:
                best_epoch = epoch
                best_valid_auc, best_test_auc = valid_auc, test_auc
                logger.info(f'best valid auc: valid {valid_auc:.4f}, test {test_auc:.4f}')
                # update best model
                if self.save_best_model:

                    best_net = deepcopy(self.net)
                    
            else:
                logger.info(f'valid {valid_auc:.4f}, test {test_auc:.4f}')

        if self.save_best_model:
            dir_path = os.path.dirname(self.writer.log_dir)
            torch.save(best_net.state_dict(), os.path.join(dir_path, "best_model.pth"))
            print(f"Successfully save best model in {best_epoch} into {dir_path}\n")
        
        result_info = f"Get best model in epoch {best_epoch}, \
                    its performanc in validation dataset {best_valid_auc:.4f}, \
                    in test dataset {best_test_auc:.4f}"
        self.writer.add_text("Performance", result_info)
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

        self.net.to(self.args.device)
        if optimizer:
            self.net.train()
        else:
            self.net.eval()

        time_avg, timestamp = utils.AvgrageMeter(), time.time()
        loss_avg, auc_avg = utils.AvgrageMeter(), utils.AvgrageMeter()

        if namespace == 'train':

            # for batch_idx in range(self.args.iter_per_epoch):
            # todo: how to ensure the epoch traverse all combinations? re-use dataloader with sql aware?
            # randomly sample one batch
            # sql_batch, data_batch = data_loader.sample_batch_sql_and_data(self.args.batch_size)
            # sql_batch_tensor = torch.tensor(sql_batch).to(self.args.device)

            for batch_idx, data_batch in enumerate(data_loader):
                if namespace == 'train' \
                        and self.args.iter_per_epoch is not None \
                        and batch_idx >= self.args.iter_per_epoch:
                    break
                sql = data_batch["sql"].to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                x_id = data_batch['id'].to(self.args.device)
                x_value = data_batch['value'].to(self.args.device)

                y = self.net((x_id, x_value), sql)

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
                
                step = epoch * len(data_loader) + batch_idx
                # self.writer.add_scalar('Loss/Training_Step_RealTime_Loss', loss.item(), step)
                # self.writer.add_scalar('Loss/Training_Step_Ave_Loss', loss_avg.avg, step)
                
        else:
            for batch_idx, data_batch in enumerate(data_loader):
                sql = data_batch["sql"].to(self.args.device)
                target = data_batch['y'].to(self.args.device)
                x_id = data_batch['id'].to(self.args.device)
                x_value = data_batch['value'].to(self.args.device)

                with torch.no_grad():
                    y = self.net((x_id, x_value), sql)
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

    
    # def inference(self, sql_batch, data_batch):
        
    #     self.net.eval()
        
    #     sql_batch_tensor = torch.tensor(sql_batch).to(self.args.device)
    #     data_batch['id'] = data_batch['id'].to(self.args.device)
    #     data_batch['value'] = data_batch['value'].to(self.args.device)

    #     with torch.no_grad():
    #         arch_advisor = self.hyper_net(sql_batch_tensor)
    #         assert arch_advisor.size(1) == self.args.moe_num_layers * self.args.K, \
    #             f"{arch_advisor.size()} is not {self.args.batch_size, self.args.moe_num_layers * self.args.K}"

    #         # reshape it to (B, L, K), the last batch may less than self.args.batch_size -> arch_advisor.size(0)
    #         arch_advisor = arch_advisor.reshape(arch_advisor.size(0), self.args.moe_num_layers, self.args.K)
    #         arch_advisor = self.sparsemax(arch_advisor)
    #         # calculate y and loss
    #         y = self.moe_net.forward(data_batch, arch_advisor)

    #         return y


    def close(self):
        self.writer.close()

    

