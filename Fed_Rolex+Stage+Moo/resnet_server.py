import copy
from collections import OrderedDict
from min_norm_solvers import MinNormSolver
import numpy as np
import ray
import torch


class ResnetServerRoll:
    def __init__(self, global_model, rate, dataset_ref, cfg_id):
        self.tau = 1e-2
        self.v_t = None
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.eta = 1e-2
        self.m_t = None
        self.user_idx = None
        self.param_idx = None
        self.dataset_ref = dataset_ref
        self.cfg_id = cfg_id
        self.cfg = ray.get(cfg_id)
        self.global_model = global_model.cpu()
        self.global_parameters = global_model.state_dict()
        self.rate = rate
        self.label_split = ray.get(dataset_ref['label_split'])
        self.make_model_rate()
        self.num_model_partitions = 50
        self.model_idxs = {}
        self.roll_idx = {}
        self.rounds = 0
        self.tmp_counts = {}
        for k, v in self.global_parameters.items():
            self.tmp_counts[k] = torch.ones_like(v)
        self.reshuffle_params()
        self.reshuffle_rounds = 512
        self.stage = 0
        # 初始化要冻结的范围
        self.freeze_model_idx = self.freeze_model([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def reshuffle_params(self):
        for k, v in self.global_parameters.items():
            if 'conv1' in k or 'conv2' in k:
                output_size = v.size(0)
                self.model_idxs[k] = torch.randperm(output_size, device=v.device)
                self.roll_idx[k] = 0
        return self.model_idxs

    def step(self, local_parameters, param_idx, user_idx):
        self.combine(local_parameters, param_idx, user_idx)
        self.rounds += 1
        if self.rounds % self.reshuffle_rounds:
            self.reshuffle_params()

    def broadcast(self, local, lr):
        self.stage = self.rounds // 600
        # self.stage = 5
        # 现在是stage by stage
        cfg = self.cfg
        self.global_model.train(True)
        num_active_users = cfg['active_user']
        # user_idx随机选取
        self.user_idx = copy.deepcopy(torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist())
        # self.user_idx = copy.deepcopy(torch.arange(num_active_users).tolist())

        
        local_parameters, self.param_idx = self.distribute(self.user_idx)

        param_ids = [ray.put(local_parameter) for local_parameter in local_parameters]

        ray.get([client.update.remote(self.user_idx[m],
                                      self.dataset_ref,
                                      {'lr': lr,
                                       'model_rate': self.model_rate[self.user_idx[m]],
                                       'local_params': param_ids[m]})
                 for m, client in enumerate(local)])
        return local, self.param_idx, self.user_idx

    def make_model_rate(self):
        cfg = self.cfg
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        roll = self.rounds
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                # idx_i是下一层的输入通道，上一层的输出通道=下一层的输入通道
                                # input_idx_i_m是这一层的输入通道
                                input_idx_i_m = idx_i[m]
                                if self.stage == 0:
                                    scaler_rate = 0.0625
                                    model_idx = torch.arange(output_size, device=v.device)
                                    output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]

                                elif self.stage == 1:
                                    scaler_rate = 0.125 if m not in [8, 9] else 0.0625
                                    model_idx = torch.arange(output_size, device=v.device)
                                    if m in [8, 9]:
                                        start_idx = int(np.ceil(output_size * 0.0625))
                                    else:
                                        start_idx = 0
                                    output_idx_i_m = model_idx[
                                                     start_idx:start_idx + int(np.ceil(output_size * scaler_rate))]

                                elif self.stage == 2:
                                    scaler_rate = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.125,
                                                   7: 0.125, 8: 0.0625, 9: 0.0625}.get(m, 0.25)
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx_roll = model_idx[int(np.ceil(output_size * 0.125)):int(
                                        np.ceil(output_size * 0.125)) * 2]
                                    model_idx_roll = torch.roll(model_idx_roll, roll, -1)

                                    if m in [8, 9]:
                                        output_idx_i_m = model_idx_roll[:int(np.ceil(output_size * scaler_rate))]
                                    elif m in [6, 7]:
                                        start_idx = int(np.ceil(output_size * 0.125))
                                        output_idx_i_m = model_idx[
                                                         start_idx:start_idx + int(np.ceil(output_size * scaler_rate))]
                                    else:
                                        output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]

                                elif self.stage == 3:
                                    scaler_rate = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.25, 5: 0.25, 6: 0.125, 7: 0.125,
                                                   8: 0.0625, 9: 0.0625}.get(m, 0.5)
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx_roll = model_idx[int(np.ceil(output_size * 0.25)):int(
                                        np.ceil(output_size * 0.25)) * 2]
                                    model_idx_roll = torch.roll(model_idx_roll, roll, -1)

                                    if m in [8, 9, 6, 7]:
                                        output_idx_i_m = model_idx_roll[:int(np.ceil(output_size * scaler_rate))]
                                    elif m in [4, 5]:
                                        start_idx = int(np.ceil(output_size * 0.25))
                                        output_idx_i_m = model_idx[
                                                         start_idx:start_idx + int(np.ceil(output_size * scaler_rate))]
                                    else:
                                        output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]

                                elif self.stage == 4:
                                    scaler_rate = {0: 1, 1: 1, 2: 0.5, 3: 0.5, 4: 0.25, 5: 0.25, 6: 0.125, 7: 0.125,
                                                   8: 0.0625, 9: 0.0625}.get(m, 1)
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx_roll = model_idx[int(np.ceil(output_size * 0.5)):int(
                                        np.ceil(output_size * 0.5)) * 2]
                                    model_idx_roll = torch.roll(model_idx_roll, roll, -1)

                                    if m in [8, 9, 6, 7, 4, 5]:
                                        output_idx_i_m = model_idx_roll[:int(np.ceil(output_size * scaler_rate))]
                                    elif m in [2, 3]:
                                        start_idx = int(np.ceil(output_size * 0.5))
                                        output_idx_i_m = model_idx[
                                                         start_idx:start_idx + int(np.ceil(output_size * scaler_rate))]
                                    else:
                                        output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]
                                elif self.stage >= 5:
                                    # evaluation
                                    scaler_rate = {0: 1, 1: 1, 2: 0.5, 3: 0.5, 4: 0.25, 5: 0.25, 6: 0.125, 7: 0.125,
                                                   8: 0.0625, 9: 0.0625}.get(m, 1)
                                    model_idx = torch.arange(output_size, device=v.device)
                                    output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]
                                # 这个可以让模型正确初始化
                                self.model_rate[self.user_idx[m]] = scaler_rate
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            # 偏置层是一维的
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx

    def distribute(self, user_idx):
        self.make_model_rate()
        # 这个函数确定每个Client分到的模型比率
        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    def convert12(self, model_state_dict_items):
        flattened_tensor = torch.cat([param.view(-1) for _, param in model_state_dict_items])
        return flattened_tensor

    def combine(self, local_parameters, param_idx, user_idx):
        # 这个函数增加了MOO(多目标优化)相关的设置
        cfg = self.cfg
        self.global_parameters = self.global_model.cpu().state_dict()
        global_vector_tensor = self.convert12(self.global_parameters.items()).float()
        local_parameters_vector_g = torch.zeros(cfg['num_users'], len(global_vector_tensor)).float()

        for m in range(len(local_parameters)):
            local_parameters_gradient = copy.deepcopy(self.global_parameters)

            for k, v in local_parameters_gradient.items():
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
                tmp_v_frozen = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
                tmp_v[torch.meshgrid(param_idx[m][k])] = local_parameters[m][k] - v[torch.meshgrid(param_idx[m][k])]
                # 冻结梯度
                if self.stage == 1:
                    if m < 8:
                        tmp_v[torch.meshgrid(self.freeze_model_idx[self.stage][k])] = 0
                elif self.stage == 2:
                    if m < 6:
                        tmp_v[torch.meshgrid(self.freeze_model_idx[self.stage][k])] = 0
                elif self.stage == 3:
                    if m < 4:
                        tmp_v[torch.meshgrid(self.freeze_model_idx[self.stage][k])] = 0
                elif self.stage >= 4:
                    # stage>=4之后，冻结一半参数
                    if m < 2:
                        tmp_v[torch.meshgrid(self.freeze_model_idx[self.stage][k])] = 0
                tmp_v[torch.meshgrid(self.freeze_model_idx[self.stage][k])] = tmp_v_frozen[
                    torch.meshgrid(self.freeze_model_idx[self.stage][k])]
                local_parameters_gradient[k] = tmp_v

            local_parameters_vector_g[m] = self.convert12(local_parameters_gradient.items()).float()

        sol, min_norm = MinNormSolver.find_min_norm_element(
            [local_parameters_vector_g[m] for m in range(len(local_parameters))])
        # 得到每个客户端的系数
        print(sol)
        print(self.stage)

        for m in range(len(local_parameters)):
            global_vector_tensor = global_vector_tensor + torch.tensor(sol[m]) * local_parameters_vector_g[m]
            global_vector_tensor = global_vector_tensor.float()
            # 更新后的全局参数=系数*客户端梯度+全局参数

        prev_ind = 0
        for k, v in self.global_parameters.items():
            # 更新到全局参数中（数据格式为model.state_dict()）
            flat_size = int(np.prod(list(v.size())))
            v.data.copy_(
                global_vector_tensor[prev_ind:prev_ind + flat_size].view(v.size()))
            prev_ind += flat_size

        self.global_model.load_state_dict(self.global_parameters)
        return

    def freeze_model(self, user_idx):
        idx = [OrderedDict() for _ in range(len(user_idx))]
        idx_i = [None for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'conv1' in k or 'conv2' in k:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)

                            if idx_i[m] is None:
                                idx_i[m] = torch.arange(input_size, device=v.device)
                            input_idx_i_m = idx_i[m]

                            # Set scaler_rate based on the value of m
                            if m == 0:
                                output_idx_i_m = torch.tensor([], dtype=torch.long, device=v.device)
                                input_idx_i_m = output_idx_i_m
                            else:
                                scaler_rate = 0.5 if m >= 4 else 0.25 if m == 3 else 0.125 if m == 2 else 0.0625
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]

                            idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    elif parameter_type == 'bias':
                        # 偏置层是一维的
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                else:
                    idx[m][k] = torch.tensor([], dtype=torch.long, device=v.device)
        return idx


class ResnetServerRandom(ResnetServerRoll):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = {0: 1, 1: 1, 2: 0.5, 3: 0.5, 4: 0.25, 5: 0.25, 6: 0.125, 7: 0.125,
                                               8: 0.0625, 9: 0.0625}.get(user_idx[m], 1)
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                # 这个可以让模型正确初始化
                                self.model_rate[self.user_idx[m]] = scaler_rate
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx


class ResnetServerStatic(ResnetServerRoll):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx
