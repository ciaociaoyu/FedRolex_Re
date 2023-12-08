import copy
from collections import OrderedDict

import numpy as np
import ray
import torch


class TransformerServerRoll:
    def __init__(self, global_model, rate, dataset_ref, cfg_id):
        self.stage = 0
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
        self.global_model = global_model
        self.global_parameters = global_model.cpu().state_dict()
        self.rate = rate
        self.label_split = ray.get(dataset_ref['label_split'])
        self.make_model_rate()
        self.num_model_partitions = 50
        self.model_idxs = {}
        self.rounds = 0
        self.tmp_counts = {}

        # 广播分发的rate
        self.b_rate = 0
        print("server initialized")

    def generate_model_for_test(self, scaler_rate):
        # 产生一个特定比率的模型
        cfg = self.cfg
        idx_i = None
        idx = {}
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        input_size = v.size(1)
                        output_size = v.size(0)
                        if 'embedding' in k.split('.')[-2]:
                            output_idx_i_m = torch.arange(output_size, device=v.device)
                            local_input_size = int(np.ceil(input_size * scaler_rate))
                            ridx = torch.arange(input_size, device=v.device)
                            input_idx_i_m = ridx[:local_input_size]
                            idx_i = input_idx_i_m
                        elif 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = idx_i
                            output_idx_i_m = torch.arange(output_size, device=v.device)
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i
                            local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                            * scaler_rate))
                            ridx = torch.arange(output_size, device=v.device)
                            output_idx_i_m = (ridx.reshape(
                                cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                            idx_i = output_idx_i_m
                        else:
                            input_idx_i_m = idx_i
                            local_output_size = int(np.ceil(output_size * scaler_rate))
                            ridx = torch.arange(output_size, device=v.device)
                            output_idx_i_m = ridx[:local_output_size]
                            idx_i = output_idx_i_m
                        idx[k] = (output_idx_i_m, input_idx_i_m)
                    else:
                        input_idx_i_m = idx_i
                        idx[k] = input_idx_i_m
                else:
                    input_size = v.size(0)
                    if 'decoder' in k and 'linear2' in k:
                        input_idx_i_m = torch.arange(input_size, device=v.device)
                        idx[k] = input_idx_i_m
                    elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                        input_idx_i_m = idx_i
                        idx[k] = input_idx_i_m
                        if 'linear_v' not in k:
                            idx_i = idx[k.replace('bias', 'weight')][1]
                    else:
                        input_idx_i_m = idx_i
                        idx[k] = input_idx_i_m
            else:
                pass
        param_idx = idx
        local_parameters = {}
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        local_parameters[k] = copy.deepcopy(v[torch.meshgrid(param_idx[k])])
                    else:
                        local_parameters[k] = copy.deepcopy(v[param_idx[k]])
                else:
                    local_parameters[k] = copy.deepcopy(v[param_idx[k]])
            else:
                local_parameters[k] = copy.deepcopy(v)
        return local_parameters

    def step(self, local_parameters, param_idx, user_idx):
        self.combine(local_parameters, param_idx, user_idx)
        self.rounds += 1

    def broadcast(self, local, lr):
        cfg = self.cfg
        self.stage = self.rounds // 40
        if 10 < self.rounds < 200:
            if self.rounds % 40 == 0:
                # 扩展模型
                if cfg['interpolate'] == 'bi':
                    self.model_scaler_rate(2 ** (self.stage - 1) * 0.0625, 2 ** (self.stage) * 0.0625)

        self.global_model.train(True)
        num_active_users = cfg['active_user']
        self.user_idx = copy.deepcopy(torch.arange(cfg['num_users'])
                                      [torch.randperm(cfg['num_users'])
            [:num_active_users]].tolist())
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
            if self.stage == 0:
                self.b_rate = 0.0625
            elif self.stage == 1:
                self.b_rate = 0.125
            elif self.stage == 2:
                self.b_rate = 0.25
            elif self.stage == 3:
                self.b_rate = 0.5
            else:
                self.b_rate = 1
        else:
            raise ValueError('Not valid model split mode')

    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if self.model_rate[self.user_idx[m]] > self.b_rate:
                    # 啥都不做，正常分配
                    scaler_rate = self.b_rate
                else:
                    # 分配力所能及的
                    scaler_rate = self.model_rate[self.user_idx[m]]
                self.model_rate[self.user_idx[m]] = scaler_rate
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.arange(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx

    def model_scaler_rate(self, scaler_rate1, scaler_rate2):
        cfg = self.cfg
        idx_i = [None for _ in range(2)]
        idx = [OrderedDict() for _ in range(2)]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(2):
                if m == 0:
                    scaler_rate = scaler_rate1
                else:
                    scaler_rate = scaler_rate2
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.arange(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        for k, v in self.global_parameters.items():
            tmp_v = self.interpolate_tensor(v[torch.meshgrid(idx[0][k])], v[torch.meshgrid(idx[1][k])].size())
            v[torch.meshgrid(idx[1][k])] = tmp_v
            self.global_parameters[k] = v
        return

    def interpolate_tensor(self, input_tensor, output_shape):
        input_dim = len(input_tensor.shape)
        input_shape = input_tensor.shape

        if len(output_shape) != input_dim:
            raise ValueError("Length of output_shape must be equal to the dimension of input_tensor.")

        output_tensor = input_tensor

        for dim in range(input_dim):
            if input_shape[dim] == output_shape[dim]:
                continue

            new_shape = list(output_tensor.shape)
            new_shape[dim] = output_shape[dim]

            indices = torch.linspace(0, input_shape[dim] - 1, steps=output_shape[dim])

            temp_tensor = torch.zeros(new_shape)

            for i in range(output_shape[dim]):
                idx = int(indices[i])
                frac = indices[i] - idx
                slc = [slice(None)] * input_dim  # 创建一个切片对象，用于索引张量
                slc[dim] = i
                if idx < input_shape[dim] - 1:
                    slc1 = slc.copy()
                    slc1[dim] = idx
                    slc2 = slc.copy()
                    slc2[dim] = idx + 1
                    temp_tensor[tuple(slc)] = (1 - frac) * output_tensor[tuple(slc1)] + frac * output_tensor[
                        tuple(slc2)]
                else:
                    slc1 = slc.copy()
                    slc1[dim] = idx
                    temp_tensor[tuple(slc)] = output_tensor[tuple(slc1)]

            output_tensor = temp_tensor
            input_shape = output_tensor.shape  # 更新 input_shape 以匹配 output_tensor 的当前形状

        return output_tensor

    def distribute(self, user_idx):
        self.make_model_rate()
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

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        self.global_parameters = self.global_model.state_dict()
        updated_parameters = copy.deepcopy(self.global_parameters)
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)
        for k, v in updated_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
            for m in range(len(local_parameters)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            if k.split('.')[-2] == 'embedding':
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                            elif 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                    else:
                        if 'decoder' in k and 'linear2' in k:
                            label_split = self.label_split[user_idx[m]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                            count[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                else:
                    tmp_v += local_parameters[m][k]
                    count[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            device = v.device
            tmp_v = tmp_v.to(device)
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = tmp_counts_cpy
        # delta_t = {k: v - self.global_parameters[k] for k, v in updated_parameters.items()}
        # if self.rounds in self.cfg['milestones']:
        #     self.eta *= 0.5
        # if not self.m_t or self.rounds in self.cfg['milestones']:
        #     self.m_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.m_t = {
        #     k: self.beta_1 * self.m_t[k] + (1 - self.beta_1) * delta_t[k] for k in delta_t.keys()
        # }
        # if not self.v_t or self.rounds in self.cfg['milestones']:
        #     self.v_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.v_t = {
        #     k: self.beta_2 * self.v_t[k] + (1 - self.beta_2) * torch.multiply(delta_t[k], delta_t[k])
        #     for k in delta_t.keys()
        # }
        # self.global_parameters = {
        #     k: self.global_parameters[k] + self.eta * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
        #     for k in self.global_parameters.keys()
        # }
        #
        self.global_parameters = updated_parameters
        self.global_model.load_state_dict(self.global_parameters)
        return


class TransformerServerRollSO(TransformerServerRoll):
    def broadcast(self, local, lr):
        cfg = self.cfg
        self.global_model.train(True)
        num_active_users = cfg['active_user']
        self.user_idx = copy.deepcopy(torch.arange(cfg['num_users'])
                                      [torch.randperm(cfg['num_users'])
            [:num_active_users]].tolist())
        local_parameters, self.param_idx = self.distribute(self.user_idx)

        # param_ids = [ray.put(local_parameter) for local_parameter in local_parameters]
        param_ids = local_parameters
        configs = [[self.user_idx[m],
                    self.dataset_ref[self.user_idx[m]],
                    {'lr': lr,
                     'model_rate': self.model_rate[self.user_idx[m]],
                     'local_params': param_ids[m]}] for m, _ in enumerate(param_ids)]
        return local, configs


class TransformerServerRandomSO(TransformerServerRoll):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.randperm(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                roll = self.rounds % output_size
                                ridx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx


class TransformerServerStaticSO(TransformerServerRoll):
    def split_model(self, user_idx):
        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                local_input_size = int(np.ceil(input_size * scaler_rate))
                                ridx = torch.arange(input_size, device=v.device)
                                input_idx_i_m = ridx[:local_input_size]
                                idx_i[m] = input_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = (ridx.reshape(
                                    cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                ridx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass
        return idx
