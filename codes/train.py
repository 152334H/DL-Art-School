import maybe_bnb
import os
import math
import argparse
import random
import logging
import shutil

from tqdm import tqdm

import torch
from data.data_sampler import DistIterSampler
from trainer.eval.evaluator import create_evaluator

from utils import util, options as option
from data import create_dataloader, create_dataset, get_dataset_debugger
from trainer.ExtensibleTrainer import ExtensibleTrainer
from time import time
from datetime import datetime

from utils.util import opt_get, map_cuda_to_correct_device
import multiprocessing as mp

def init_dist(backend, **kwargs):
    # These packages have globals that screw with Windows, so only import them if needed.
    import torch.distributed as dist

    rank = int(os.environ['LOCAL_RANK'])
    assert rank < torch.cuda.device_count()
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, **kwargs)

class Trainer:

    def init(self, opt_path, opt, launcher):
        self.use_8bit = opt_get(opt, ['use_8bit'], True)
        if self.use_8bit:
            maybe_bnb.populate()
        else:
            maybe_bnb.populate(False, False, False, embedding=None)
        self._profile = False
        self.val_compute_psnr = opt_get(opt, ['eval', 'compute_psnr'], False)
        self.val_compute_fea = opt_get(opt, ['eval', 'compute_fea'], False)
        self.current_step = 0
        self.total_training_data_encountered = 0

        #### loading resume state if exists
        if opt['path'].get('resume_state', None):
            # distributed resuming: all load into default GPU
            resume_state = torch.load(opt['path']['resume_state'], map_location=map_cuda_to_correct_device)
        else:
            resume_state = None

        #### mkdir and loggers
        if self.rank <= 0:  # normal training (self.rank -1) OR distributed training (self.rank 0)
            if resume_state is None:
                util.mkdir_and_rename(
                    opt['path']['experiments_root'])  # rename experiment folder if exists
                util.mkdirs(
                    (path for key, path in opt['path'].items() if not key == 'experiments_root' and path is not None
                     and 'pretrain_model' not in key and 'resume' not in key))
            shutil.copy(opt_path, os.path.join(opt['path']['experiments_root'], f'{datetime.now().strftime("%d%m%Y_%H%M%S")}_{os.path.basename(opt_path)}'))

            # config loggers. Before it, the log will not work
            util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                              screen=True, tofile=True)
            self.logger = logging.getLogger('base')
            self.logger.info(option.dict2str(opt))
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                self.tb_logger_path = os.path.join(opt['path']['experiments_root'], 'tb_logger')
                from torch.utils.tensorboard import SummaryWriter
                self.tb_logger = SummaryWriter(log_dir=self.tb_logger_path)
        else:
            util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
            self.logger = logging.getLogger('base')

        if resume_state is not None:
            option.check_resume(opt, resume_state['iter'])  # check resume options

        # convert to NoneDict, which returns None for missing keys
        opt = option.dict_to_nonedict(opt)
        self.opt = opt

        #### wandb init
        if opt['wandb'] and self.rank <= 0:
            import wandb
            os.makedirs(os.path.join(opt['path']['log'], 'wandb'), exist_ok=True)
            project_name = opt_get(opt, ['wandb_project_name'], opt['name'])
            run_name = opt_get(opt, ['wandb_run_name'], None)
            wandb.init(project=project_name, dir=opt['path']['log'], config=opt, name=run_name)

        #### random seed
        seed = opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        if self.rank <= 0:
            self.logger.info('Random seed: {}'.format(seed))
        seed += self.rank  # Different multiprocessing instances should behave differently.
        util.set_random_seed(seed)

        torch.backends.cudnn.benchmark = opt_get(opt, ['cuda_benchmarking_enabled'], True)
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.deterministic = True
        if opt_get(opt, ['anomaly_detection'], False):
            torch.autograd.set_detect_anomaly(True)

        # Save the compiled opt dict to the global loaded_options variable.
        util.loaded_options = opt

        #### create train and val dataloader
        dataset_ratio = 1  # enlarge the size of each epoch
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train':
                self.train_set, collate_fn = create_dataset(dataset_opt, return_collate=True)
                self.dataset_debugger = get_dataset_debugger(dataset_opt)
                if self.dataset_debugger is not None and resume_state is not None:
                    self.dataset_debugger.load_state(opt_get(resume_state, ['dataset_debugger_state'], {}))
                train_size = int(math.ceil(len(self.train_set) / dataset_opt['batch_size']))
                total_iters = int(opt['train']['niter'])
                self.total_epochs = int(math.ceil(total_iters / train_size))
                if opt['dist']:
                    self.train_sampler = DistIterSampler(self.train_set, self.world_size, self.rank, dataset_ratio)
                    self.total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                    shuffle = False
                else:
                    self.train_sampler = None
                    shuffle = True
                self.train_loader = create_dataloader(self.train_set, dataset_opt, opt, self.train_sampler, collate_fn=collate_fn, shuffle=shuffle)
                if self.rank <= 0:
                    self.logger.info('Number of training data elements: {:,d}, iters: {:,d}'.format(
                        len(self.train_set), train_size))
                    self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                        self.total_epochs, total_iters))
            elif phase == 'val':
                self.val_set, collate_fn = create_dataset(dataset_opt, return_collate=True)
                self.val_loader = create_dataloader(self.val_set, dataset_opt, opt, None, collate_fn=collate_fn)
                if self.rank <= 0:
                    self.logger.info('Number of val images in [{:s}]: {:d}'.format(
                        dataset_opt['name'], len(self.val_set)))
            else:
                raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
        assert self.train_loader is not None

        #### create model
        self.model = ExtensibleTrainer(opt)

        ### Evaluators
        self.evaluators = []
        if 'eval' in opt.keys() and 'evaluators' in opt['eval'].keys():
            # In "pure" mode, we propagate through the normal training steps, but use validation data instead and average
            # the total loss. A validation dataloader is required.
            if opt_get(opt, ['eval', 'pure'], False):
                assert hasattr(self, 'val_loader')

            for ev_key, ev_opt in opt['eval']['evaluators'].items():
                self.evaluators.append(create_evaluator(self.model.networks[ev_opt['for']],
                                                        ev_opt, self.model.env))

        #### resume training
        if resume_state:
            self.logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                resume_state['epoch'], resume_state['iter']))

            self.start_epoch = resume_state['epoch']
            self.current_step = resume_state['iter']
            self.total_training_data_encountered = opt_get(resume_state, ['total_data_processed'], 0)
            if opt_get(opt, ['path', 'optimizer_reset'], False):
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('!! RESETTING OPTIMIZER STATES')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            else:
                self.model.resume_training(resume_state, 'amp_opt_level' in opt.keys())  # handle optimizers and schedulers
        else:
            self.current_step = -1 if 'start_step' not in opt.keys() else opt['start_step']
            self.total_training_data_encountered = 0 if 'training_data_encountered' not in opt.keys() else opt['training_data_encountered']
            self.start_epoch = 0
        if 'force_start_step' in opt.keys():
            self.current_step = opt['force_start_step']
            self.total_training_data_encountered = self.current_step * opt['datasets']['train']['batch_size']
        opt['current_step'] = self.current_step

        #### validation
        if 'val_freq' in opt['train'].keys():
            self.val_freq = opt['train']['val_freq'] * opt['datasets']['train']['batch_size']
        else:
            self.val_freq = int(opt['train']['val_freq_megasamples'] * 1000000)

        self.next_eval_step = self.total_training_data_encountered + self.val_freq
        del resume_state  # For whatever reason, this relieves a memory burden on the first GPU for some training sessions.

    def do_step(self, train_data):
        if self._profile:
            print("Data fetch: %f" % (time() - _t))
            _t = time()

        opt = self.opt
        batch_size = self.opt['datasets']['train']['batch_size']  # It may seem weird to derive this from opt, rather than train_data. The reason this is done is
                                                                  # because train_data is process-local while the opt variant represents all of the data fed across all GPUs.
        self.current_step += 1
        self.total_training_data_encountered += batch_size
        will_log = self.current_step % opt['logger']['print_freq'] == 0

        #### update learning rate
        self.model.update_learning_rate(self.current_step, warmup_iter=opt['train']['warmup_iter'])

        #### training
        if self._profile:
            print("Update LR: %f" % (time() - _t))
        _t = time()
        self.model.feed_data(train_data, self.current_step)
        gradient_norms_dict = self.model.optimize_parameters(self.current_step, return_grad_norms=will_log)
        iteration_rate = (time() - _t) / batch_size
        if self._profile:
            print("Model feed + step: %f" % (time() - _t))
            _t = time()

        #### log
        if self.dataset_debugger is not None:
            self.dataset_debugger.update(train_data)
        if will_log:
            # Must be run by all instances to gather consensus.
            current_model_logs = self.model.get_current_log(self.current_step)
        if will_log and self.rank <= 0:
            logs = {'step': self.current_step,
                    'samples': self.total_training_data_encountered,
                    'megasamples': self.total_training_data_encountered / 1000000,
                    'iteration_rate': iteration_rate}
            logs.update(current_model_logs)
            if self.dataset_debugger is not None:
                logs.update(self.dataset_debugger.get_debugging_map())
            logs.update(gradient_norms_dict)
            message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(self.epoch, self.current_step)
            for v in self.model.get_current_learning_rate():
                message += '{:.3e},'.format(v)
            message += ')] '
            for k, v in logs.items():
                if 'histogram' in k:
                    self.tb_logger.add_histogram(k, v, self.current_step)
                elif isinstance(v, dict):
                    self.tb_logger.add_scalars(k, v, self.current_step)
                else:
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        self.tb_logger.add_scalar(k, v, self.current_step)
            if opt['wandb'] and self.rank <= 0:
                import wandb
                wandb_logs = {}
                for k, v in logs.items():
                    if 'histogram' in k:
                        wandb_logs[k] = wandb.Histogram(v)
                    else:
                        wandb_logs[k] = v
                if opt_get(opt, ['wandb_progress_use_raw_steps'], False):
                    wandb.log(wandb_logs, step=self.current_step)
                else:
                    wandb.log(wandb_logs, step=self.total_training_data_encountered)
            self.logger.info(message)

        #### save models and training states
        if self.current_step % opt['logger']['save_checkpoint_freq'] == 0:
            self.model.consolidate_state()
            if self.rank <= 0:
                if opt['logger']['disable_state_saving'] is False:
                    self.logger.info('Saving models and training states.')
                else:
                    self.logger.info('Saving model.')

                if opt['upgrades']['number_of_checkpoints_to_save'] > 0 or \
                        opt['upgrades']['number_of_states_to_save'] > 0:

                    number_of_states_to_save = opt['upgrades']['number_of_states_to_save'] \
                        if not opt['logger']['disable_state_saving'] else 0

                    self.logger.info(
                        f"Leaving only {opt['upgrades']['number_of_checkpoints_to_save']} checkpoints and "
                        f"{number_of_states_to_save} states"
                    )
                    self.model.limit_number_of_checkpoints_and_states(
                        next(iter(opt['networks'].keys())),
                        models_number=opt['upgrades']['number_of_checkpoints_to_save'],
                        state_number=number_of_states_to_save,
                    )

                self.model.save(self.current_step)
                state = {'epoch': self.epoch, 'iter': self.current_step, 'total_data_processed': self.total_training_data_encountered}
                if self.dataset_debugger is not None:
                    state['dataset_debugger_state'] = self.dataset_debugger.get_state()
                if opt['logger']['disable_state_saving'] is False:
                    self.model.save_training_state(state)
                else:
                    self.logger.info("State saving is disabled. Skipping state save, you won't be able to resume training from this session.")
            if 'alt_path' in opt['path'].keys():
                import shutil
                print("Synchronizing tb_logger to alt_path..")
                alt_tblogger = os.path.join(opt['path']['alt_path'], "tb_logger")
                shutil.rmtree(alt_tblogger, ignore_errors=True)
                shutil.copytree(self.tb_logger_path, alt_tblogger)

        do_eval = self.total_training_data_encountered > self.next_eval_step
        if do_eval:
            self.next_eval_step = self.total_training_data_encountered + self.val_freq
        if opt_get(opt, ['eval', 'pure'], False) and do_eval:
            metrics = []
            for val_data in tqdm(self.val_loader):
                self.model.feed_data(val_data, self.current_step, perform_micro_batching=False)
                metrics.append(self.model.test())
            reduced_metrics = {}
            for metric in metrics:
                for k, v in metric.as_dict().items():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 0:
                        if k in reduced_metrics.keys():
                            reduced_metrics[k].append(v)
                        else:
                            reduced_metrics[k] = [v]
            if self.rank <= 0:
                for k, v in reduced_metrics.items():
                    val = torch.stack(v).mean().item()
                    self.tb_logger.add_scalar(f'val_{k}', val, self.current_step)
                    print(f">>Eval {k}: {val}")
                if opt['wandb']:
                    import wandb
                    wandb.log({f'eval_{k}': torch.stack(v).mean().item() for k,v in reduced_metrics.items()})

        if len(self.evaluators) != 0 and do_eval:
            eval_dict = {}
            for eval in self.evaluators:
                if eval.uses_all_ddp or self.rank <= 0:
                    eval_dict.update(eval.perform_eval())
            if self.rank <= 0:
                print("Evaluator results: ", eval_dict)
                for ek, ev in eval_dict.items():
                    self.tb_logger.add_scalar(ek, ev, self.current_step)
                if opt['wandb']:
                    import wandb
                    wandb.log(eval_dict)

        # Should not be necessary, but make absolutely sure that there is no grad leakage from validation runs.
        for net in self.model.networks.values():
            net.zero_grad()

    def do_training(self):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if self.opt['dist']:
                self.train_sampler.set_epoch(epoch)

            tq_ldr = tqdm(self.train_loader) if self.rank <= 0 else self.train_loader

            _t = time()
            for train_data in tq_ldr:
                self.do_step(train_data)

    def create_training_generator(self, index):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self.epoch = epoch
            if self.opt['dist']:
                self.train_sampler.set_epoch(epoch)
            tq_ldr = tqdm(self.train_loader, position=index)

            _t = time()
            for train_data in tq_ldr:
                yield self.model
                self.do_step(train_data)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_vit_latent.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    if args.launcher != 'none':
        # export CUDA_VISIBLE_DEVICES for running in distributed mode.
        if 'gpu_ids' in opt.keys():
            gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
            print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    trainer = Trainer()

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        trainer.rank = -1
        if len(opt['gpu_ids']) == 1:
            torch.cuda.set_device(opt['gpu_ids'][0])
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist('nccl')
        trainer.world_size = torch.distributed.get_world_size()
        trainer.rank = torch.distributed.get_rank()
        torch.cuda.set_device(torch.distributed.get_rank())

    trainer.init(args.opt, opt, args.launcher)
    trainer.do_training()
