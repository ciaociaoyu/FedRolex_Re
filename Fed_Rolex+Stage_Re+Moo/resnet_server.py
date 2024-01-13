import copy
from collections import OrderedDict
import numpy as np
import ray
import torch
from models import resnet
import pdb


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
        self.stage = 0
        self.make_model_rate()
        self.num_model_partitions = 50
        self.model_idxs = {}
        self.roll_idx = {}
        self.rounds = 0
        self.tmp_counts = {}
        self.reshuffle_params()
        self.reshuffle_rounds = 512

        # 广播分发的rate
        self.b_rate = 0

    def step(self, local_parameters, param_idx, user_idx):
        self.combine(local_parameters, param_idx, user_idx)
        self.rounds += 1

    def broadcast(self, local, lr):
        cfg = self.cfg
        self.stage = self.rounds // 4
        if 1 < self.rounds < 5:
            if self.rounds % 4 == 0:
                # 扩展模型
                if cfg['interpolate'] == 'bi':
                    # self.model_scaler_rate_bi(2 ** (self.stage) * 0.0625)
                    self.cfg['global_model_rate'] = 1
                    cfg = self.cfg
                    self.model_scaler_rate_bi(self.cfg['global_model_rate'])
                if cfg['interpolate'] == 'pz':
                    # self.model_scaler_rate_pz(2 ** (self.stage) * 0.0625)
                    self.cfg['global_model_rate'] = 1
                    cfg = self.cfg
                    self.model_scaler_rate_pz(self.cfg['global_model_rate'])

        self.global_model.train(True)
        num_active_users = cfg['active_user']
        # user_idx随机选取
        self.user_idx = copy.deepcopy(
            torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist())
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
            # 每次都重新赋值一遍
            if self.stage == 0:
                self.b_rate = 0.5
            elif self.stage == 1:
                self.b_rate = 1
            else:
                self.b_rate = 1
        else:
            raise ValueError('Not valid model split mode')
        return

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
                                # idx_i是下一层的输入通道，上一层的输出通道=下一层的输入通道
                                # input_idx_i_m是这一层的输入通道
                                input_idx_i_m = idx_i[m]
                                if self.model_rate[self.user_idx[m]] > self.b_rate:
                                    # 啥都不做，正常分配
                                    self.model_rate[self.user_idx[m]] = self.b_rate
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                # global_model_rate动态变化，所以scaler_rate也需要动态变化
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:int(np.ceil(output_size * scaler_rate))]
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

    def model_scaler_rate_bi(self, scaler_rate_temp):
        # 这个函数把全局模型扩张到scaler_rate_temp
        # scaler_rate_temp是新的全局模型比率
        cfg = self.cfg
        # 创建一个新的全局模型
        global_model_temp = resnet.resnet18(model_rate=scaler_rate_temp, cfg=cfg)
        global_model_temp_para = global_model_temp.state_dict()
        for k, v in global_model_temp_para.items():
            # 这里把（原始大模型的的一个tensor），插值放到新的模型中（扩张后的）
            tmp_v = self.interpolate_tensor(self.global_parameters[k], v.size())
            v = tmp_v
            global_model_temp_para[k] = v
        # 更换全局模型
        # 参数替换
        self.global_parameters = global_model_temp_para
        # 模型替换
        self.global_model = global_model_temp
        # 加载参数
        self.global_model.load_state_dict(self.global_parameters)
        return

    def model_scaler_rate_pz(self, scaler_rate_temp):
        cfg = self.cfg
        global_model_temp = resnet.resnet18(model_rate=scaler_rate_temp, cfg=cfg).to(cfg['device'])
        global_model_temp_para = global_model_temp.state_dict()
        for k, v in global_model_temp_para.items():
            global_model_temp_para[k] = torch.zeros_like(v)
            # 变成了全0的模型
        for k, v in self.global_parameters.items():
            temp_para_tensor = global_model_temp_para[k]
            # 检查所有维度的尺寸
            if all(v.size(dim) <= temp_para_tensor.size(dim) for dim in range(v.dim())):
                # 使用 advanced indexing 赋值
                slices = tuple(slice(0, v.size(dim)) for dim in range(v.dim()))
                global_model_temp_para[k][slices] = v
            else:
                # 尺寸不匹配，抛出错误
                raise ValueError(
                    f"尺寸错误：self.global_parameters 中的 '{k}' 尺寸大于 global_model_temp_para 中的对应张量。")
        # 更换全局模型
        # 参数替换
        self.global_parameters = global_model_temp_para
        # 模型替换
        self.global_model = global_model_temp
        # 加载参数
        self.global_model.load_state_dict(self.global_parameters)
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

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        self.global_parameters = self.global_model.cpu().state_dict()
        updated_parameters = copy.deepcopy(self.global_parameters)
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)
        for k, v in updated_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device='cpu')
            for m in range(len(local_parameters)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            if 'linear' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(
                                    param_idx[m][k])] * local_parameters[m][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(
                                    param_idx[m][k])]
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                output_size = v.size(0)
                                scaler_rate = self.model_rate[user_idx[m]] / self.cfg['global_model_rate']
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if self.cfg['weighting'] == 'avg':
                                    K = 1
                                elif self.cfg['weighting'] == 'width':
                                    K = local_output_size
                                elif self.cfg['weighting'] == 'updates':
                                    K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                elif self.cfg['weighting'] == 'updates_width':
                                    K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                # K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                # K = local_output_size
                                # K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += K * local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += K
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_parameters[m][k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                    else:
                        if 'linear' in k:
                            label_split = self.label_split[user_idx[m]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_parameters[m][k][
                                label_split]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_parameters[m][k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                else:
                    tmp_v += self.tmp_counts[k] * local_parameters[m][k]
                    count[k] += self.tmp_counts[k]
                    tmp_counts_cpy[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = tmp_counts_cpy
        self.global_parameters = updated_parameters
        self.global_model.load_state_dict(self.global_parameters)
        return



