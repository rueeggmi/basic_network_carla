import numpy as np
import torch
from torchvision.utils import make_grid
from torch.nn.utils import clip_grad_norm_
from base import BaseTrainer
from utils import inf_loop, MetricTracker, plot_grad_flow, plot_classes_preds
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchsummary import summary
# from PIL import Image, ImageDraw, ImageFont


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.loss_weights = config['loss_weights']
        self.clip = config['grad_clipping']
        self.calculate_mean_std = False  # change to True if mean & std of dataset needs to be known for normalization

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # print summary of network (adapt input size to image size)
        summary(self.model, input_size=([(3, 160, 346), (1,)]))


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # calculate dataset mean and std if requested. Values can be used for image normalization.
        if self.calculate_mean_std:
            mean = torch.tensor([0.0, 0.0, 0.0])
            meansq = torch.tensor([0.0, 0.0, 0.0])
            count = 0

            for batch_idx, (data, speed, steer, throttle, brake) in enumerate(self.data_loader):
                for i in range(0, data.shape[0]):
                    mean = mean + data[i].sum(axis=[1,2])
                    #mean = data.sum()
                    #data_sqd = data[i] ** 2
                    meansq = meansq + (data[i] ** 2).sum(axis=[1,2])
                    count += np.prod(data[i].shape[1:3])

            total_mean = mean / count
            total_var = (meansq / count) - (total_mean ** 2)
            total_std = torch.sqrt(total_var)
            print("mean: " + str(total_mean))
            print("std: " + str(total_std))

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, speed, steer, throttle, brake) in enumerate(self.data_loader):
            data, speed = data.to(self.device), speed.to(self.device)
            # steer, throttle, brake = steer.to(self.device), throttle.to(self.device), brake.to(self.device)
            # target = [steer.to(self.device), throttle.to(self.device), brake.to(self.device), speed.to(self.device)]
            target = torch.cat((steer.to(self.device), throttle.to(self.device), brake.to(self.device), speed.to(self.device)), dim=1)
            self.optimizer.zero_grad()
            loss_weights = [self.loss_weights['steer'], self.loss_weights['throttle'], 
                            self.loss_weights['brake'], self.loss_weights['speed']]
            output = self.model(data, speed)
            output.to(self.device)

            loss = self.criterion(output.float(), target.float())  # , loss_weights)
            loss.backward()

            clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if epoch == 1 and batch_idx == 0:
                self.writer.add_graph(self.model, [data, speed])

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_figure('images', plot_classes_preds(data.detach().cpu().numpy()[0:4, :, :, :],
                                                                    target.detach().cpu().numpy()[0:4, :],
                                                                    output.detach().cpu().numpy()[0:4, :]))
                #self.writer.add_image('input', make_grid(imgs_save.cpu(), nrow=8, normalize=True))
                self.writer.add_figure('grad_flow', plot_grad_flow(self.model.named_parameters()))


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("starting with validation epoch")
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, speed, steer, throttle, brake) in enumerate(self.valid_data_loader):
                data, speed = data.to(self.device), speed.to(self.device)
                target = torch.cat(
                    (steer.to(self.device), throttle.to(self.device), brake.to(self.device), speed.to(self.device)),
                    dim=1)
                output = self.model(data, speed)
                loss = self.criterion(output.float(), target.float())
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_figure('images', plot_classes_preds(data.detach().cpu().numpy()[0:4, :, :, :],
                                                                    target.detach().cpu().numpy()[0:4, :],
                                                                    output.detach().cpu().numpy()[0:4, :]))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                #self.writer.add_scalar('validation_loss', loss.item())

        # add histogram of model parameters to the tensorboard
        # print("add histogram to tensorboard")
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
            self.writer.add_histogram("grad{}".format(name), p.grad.data.cpu().numpy(), bins='auto')
        # print("Added all histograms")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
