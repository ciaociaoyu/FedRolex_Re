import numpy as np
import torch


class MinNormSolver:
    MAX_ITER = 400
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """

        dmin = 1e8

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    dps[(i, j)] += torch.mul(vecs[i], vecs[j]).sum().data.cpu().float()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    dps[(i, i)] += torch.mul(vecs[i], vecs[i]).sum().data.cpu().float()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    dps[(j, j)] += torch.mul(vecs[j], vecs[j]).sum().data.cpu().float()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y, _ = torch.sort(y, descending=True)
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.maximum(y - tmax_f, torch.zeros_like(y))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)

        # 创建空的张量来存储满足条件的值
        tm1 = torch.empty(0)
        tm2 = torch.empty(0)

        # 计算满足条件的值
        tm1_cond = proj_grad < 0
        if tm1_cond.sum() > 0:
            tm1 = -1.0 * cur_val[tm1_cond] / proj_grad[tm1_cond]

        tm2_cond = proj_grad > 0
        if tm2_cond.sum() > 0:
            tm2 = (1.0 - cur_val[tm2_cond]) / proj_grad[tm2_cond]

        # 计算skippers
        skippers = torch.sum(tm1 < 1e-7) + torch.sum(tm2 < 1e-7)

        t = torch.tensor(1.0)

        if len(tm1[tm1 > 1e-7]) > 0:
            t = torch.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, torch.min(tm2[tm2 > 1e-7]))

        # 计算下一个点
        next_point = proj_grad * t + cur_val

        # 假设 _projection2simplex 已经支持PyTorch张量
        next_point = MinNormSolver._projection2simplex(next_point)

        return next_point

    def find_min_norm_element(cfg, vecs):
        e = cfg['moo_restrain']
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        # 长度获取
        n = len(vecs)

        # 张量初始化
        sol_vec = torch.zeros(n)

        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        iter_count = 0

        # 创建梯度矩阵
        grad_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            # 计算梯度方向
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)

            # 假设 _next_point 已经支持PyTorch张量
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)

            # 重新计算内积
            v1v1 = torch.tensor(0.0)
            v1v2 = torch.tensor(0.0)
            v2v2 = torch.tensor(0.0)

            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]

            # 假设 _min_norm_element_from2 已经支持PyTorch张量
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)

            # 更新解向量
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            
            # 以下为新加入的约束条件
            # 对 new_sol_vec 的元素进行约束
            new_sol_vec = torch.clamp(new_sol_vec, min=0.1-e, max=0.1+e)
            
            # 由于约束可能导致元素之和不再是 1，因此需要重新标准化
            new_sol_vec /= torch.sum(new_sol_vec)

            # 检查变化量
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return new_sol_vec, nd  # 注意这里返回的是 new_sol_vec 而不是 sol_vec

            sol_vec = new_sol_vec
            iter_count = iter_count + 1

        return sol_vec, nd  # 注意这里返回的也应该考虑是不是 new_sol_vec
