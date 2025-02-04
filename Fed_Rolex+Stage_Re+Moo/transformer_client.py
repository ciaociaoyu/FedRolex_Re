import copy
import datetime
import time

import ray
import torch

import models
from models import transformer
from data import BatchDataset, SplitDataset, make_data_loader
from logger import Logger
from metrics import Metric
from utils import make_optimizer, to_device


@ray.remote(num_gpus=0.15)
class TransformerClient:
    def __init__(self, log_path, cfg):
        self.dataset = None
        self.local_parameters = None
        self.m = None
        self.start_time = None
        self.num_active_users = None
        self.optimizer = None
        self.model = None
        self.lr = None
        self.label_split = None
        self.data_loader = None
        self.model_rate = None
        self.client_id = None
        cfg = ray.get(cfg[0])
        self.metric = Metric()
        self.logger = Logger(log_path)
        self.cfg = cfg

    def update(self, client_id, dataset_ref, model_ref):
        dataset = ray.get(dataset_ref['dataset'])
        data_split = ray.get(dataset_ref['split'])
        label_split = ray.get(dataset_ref['label_split'])
        local_parameters = ray.get(model_ref['local_params'])
        # dataset_ref = torch.load('data_store')
        # dataset = (dataset_ref['dataset'])
        # data_split = (dataset_ref['split'])
        # label_split = (dataset_ref['label_split'])
        # local_parameters = {k: v.clone().cuda() for k, v in local_parameters.items()}
        self.local_parameters = local_parameters
        self.client_id = client_id
        self.model_rate = model_ref['model_rate']
        cfg = self.cfg
        self.dataset = BatchDataset(dataset, cfg['bptt'])
        self.label_split = label_split
        self.lr = model_ref['lr']
        self.metric = Metric()

    def step(self, m, num_active_users, start_time):
        cfg = self.cfg
        # print(cfg['bptt'])
        self.model = transformer.transformer(model_rate=self.model_rate, cfg=self.cfg).cpu()

        self.model.load_state_dict(self.local_parameters)
        self.model = self.model.to('cuda')
        self.model.train(True)
        self.optimizer = make_optimizer(self.model, self.lr)
        self.m = m
        self.num_active_users = num_active_users
        self.start_time = start_time
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, step_input in enumerate(self.dataset):
                input_size = step_input['label'].size(0)
                # step_input['label_split'] = None
                step_input['label_split'] = torch.tensor(self.label_split[self.client_id])
                step_input = to_device(step_input, 'cuda')
                self.optimizer.zero_grad()
                output = self.model(step_input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                evaluation = self.metric.evaluate(cfg['metric_name']['train']['Local'], step_input, output)
                self.logger.append(evaluation, 'train', n=input_size)
            self.log(local_epoch, cfg)
        return self.pull()

    def pull(self):
        model_state = {k: v.detach().clone() for k, v in self.model.cpu().state_dict().items()}
        return model_state

    def log(self, epoch, cfg):
        if self.m % int((self.num_active_users * cfg['log_interval']) + 1) == 0 or True:
            local_time = (time.time() - self.start_time) / (self.m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (self.num_active_users - self.m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * self.num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * self.m / self.num_active_users),
                             'ID: {}({}/{})'.format(self.client_id, self.m + 1, self.num_active_users),
                             'Learning rate: {}'.format(self.lr),
                             'Rate: {}'.format(self.model_rate),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            self.logger.append(info, 'train', mean=False)
            self.logger.write('train', cfg['metric_name']['train']['Local'])

    def test_model_for_user(self, m, batch_dataset, epoch):
        with torch.no_grad():
            cfg = self.cfg
            metric = Metric()
            model = transformer.transformer(model_rate=self.model_rate, cfg=cfg).cpu()
            model.load_state_dict(self.local_parameters)
            model = model.to('cuda')
            model.train(False)
            count = 0
            mean_loss = 0
            mean_Perplexity = 0
            for i, input in enumerate(batch_dataset):
                input_size = input['label'].size(0)
                input = to_device(input, 'cuda')
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
                mean_loss += float(evaluation['Global-Loss'])
                mean_Perplexity += float(evaluation['Global-Perplexity'])
                count += 1
            print("--------------")
            print("这是客户端{}的精度，使用全体测试集".format(m))
            print("模型rate：{}".format(self.model_rate))
            print("epoch：{}".format(epoch))
            print("--------------")
            if count == 0:
                print("error count = 0")
            else:
                print("Loss：{}".format(mean_loss / count))
                print("Perplexity：{}".format(mean_Perplexity / count))
        return

