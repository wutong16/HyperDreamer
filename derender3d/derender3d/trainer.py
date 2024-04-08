import glob
import os
from datetime import datetime

import torch

from . import meters
from . import utils
from .dataloaders import get_data_loaders


class Trainer():
    def __init__(self, cfgs, model):
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.resume = cfgs.get('resume', True)
        self.pretrained = cfgs.get('pretrained', False)
        self.pretrained_checkpoint  = cfgs.get('pretrained_checkpoint', None)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.archive_code = cfgs.get('archive_code', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.visualize_validation = cfgs.get('visualize_validation', False)
        self.only_validation = cfgs.get('only_validation', False)
        self.val_log_freq = cfgs.get('val_log_freq', self.log_freq)
        self.cfgs = cfgs

        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self
        self.train_loader, self.vis_loader, self.val_loader, self.test_loader = get_data_loaders(cfgs)

    def load_checkpoint(self, optim=True, metrics=True, overwrite_checkpoint_path=None):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if overwrite_checkpoint_path is None:
            if self.checkpoint_name is not None:
                checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
            else:
                checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
                if len(checkpoints) == 0:
                    return 0
                checkpoint_path = checkpoints[-1]
                self.checkpoint_name = os.path.basename(checkpoint_path)
        else:
            checkpoint_path = overwrite_checkpoint_path
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if optim:
            self.model.load_optimizer_state(cp)
        if metrics:
            self.metrics_trace = cp['metrics_trace']
        epoch = cp['epoch']
        return epoch

    def save_checkpoint(self, epoch, optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        state_dict = self.model.get_model_state()
        if optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        self.model.to_device(self.device)
        self.current_epoch = self.load_checkpoint(optim=False)
        if self.test_result_dir is None:
            self.test_result_dir = os.path.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth',''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            m = self.run_epoch(self.test_loader, epoch=self.current_epoch, is_test=True)

    def train(self):
        """Perform training."""
        ## archive code and configs
        if self.archive_code:
            utils.archive_code(os.path.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py', '.yml'])
        utils.dump_yaml(os.path.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        ## initialize
        start_epoch = 0
        self.metrics_trace.reset()
        self.train_iter_per_epoch = len(self.train_loader)
        self.val_iter_per_epoch = len(self.val_loader)
        self.model.to_device(self.device)
        self.model.init_optimizers()

        ## resume from checkpoint
        if self.resume:
            start_epoch = self.load_checkpoint(optim=True, metrics=True)
        elif self.pretrained:
            self.load_checkpoint(optim=False, metrics=False, overwrite_checkpoint_path=self.pretrained_checkpoint)

        ## initialize tensorboardX logger
        if self.use_logger:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

            ## cache one batch for visualization
            self.vis_input = self.vis_loader.__iter__().__next__()

        ## run epochs
        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            if not self.only_validation:
                metrics = self.run_epoch(self.train_loader, epoch)
                self.metrics_trace.append("train", metrics)

            with torch.no_grad():
                metrics = self.run_epoch(self.val_loader, epoch, is_validation=True)
                self.metrics_trace.append("val", metrics)

            if (epoch+1) % self.save_checkpoint_freq == 0 and not self.only_validation:
                self.save_checkpoint(epoch+1, optim=True)
            self.metrics_trace.plot(pdf_path=os.path.join(self.checkpoint_dir, 'metrics.pdf'))
            self.metrics_trace.save(os.path.join(self.checkpoint_dir, 'metrics.json'))

        print(f"Training completed after {epoch+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False, is_test=False):
        """Run one epoch."""
        is_train = not is_validation and not is_test
        metrics = self.make_metrics()

        if is_train:
            print(f"Starting training epoch {epoch}")
            self.model.set_train()
        else:
            print(f"Starting validation epoch {epoch}")
            self.model.set_eval()

        for iter, input in enumerate(loader):
            if is_train:
                self.model.set_progress(self.current_epoch, self.num_epochs, iter, len(loader))

            m = self.model.forward(input)
            if is_train:
                self.model.backward()
            elif is_test:
                self.model.save_results(self.test_result_dir)

            metrics.update(m, self.batch_size)
            print(f"{'T' if is_train else 'V'}{epoch:02}/{iter:05}/{metrics}")

            if self.use_logger and is_train:
                total_iter = iter + epoch*self.train_iter_per_epoch
                if total_iter % self.log_freq == 0:
                    self.model.set_eval()
                    self.model.forward(self.vis_input)
                    self.model.visualize(self.logger, total_iter=total_iter, max_bs=25)
                    self.model.set_train()
            # if self.use_logger and not is_train:
            #     total_iter = iter + epoch * self.val_iter_per_epoch
            #     if iter % self.val_log_freq == 0:
            #         self.model.visualize(self.logger, total_iter=total_iter, max_bs=25, prefix='Val_')

        if self.use_logger and not is_train:
            for k, v in metrics.get_data_dict().items():
                self.logger.add_scalar(f'Val/Metrics/{k}', v, epoch)
        if not is_train:
            print("Metrics:")
            print(metrics.get_data_dict())

        return metrics
