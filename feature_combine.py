import torch
import torch.nn as nn

from typing import  Any, Callable, List, Optional, Type, Union
from torch import Tensor

# 针对的是两个Tensor的堆叠的所以Tensor之间的欧氏距离
def eu_distance(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))

# 写一个针对于LDLF论文中提到的
def eu_distance_2(x1,x2):
    loss_matrix = eu_distance(x1,x2)
    loss_ = .0
    for j in range(N):
        loss_ += loss_matrix[j][j].item()
    return loss_

if __name__ == "__main__":
    a = torch.rand(size = (4,5))
    b = torch.rand(size= (4,5))
    dis_matrix = eu_distance(a,b)
    print(dis_matrix)
    # n_features = 2
    # N = n_features
    # features = []
    # for i in range(n_features):
    #     if i < 5:
    #         _ = torch.ones(size=(3,))
    #         features.append(_)
    #     else:
    #         _ = torch.zeros(size=(3,))
    #         features.append(_)
    #
    # losses = 0.
    # print(type(features), len(features), features[1])
    # for i in range(1, N):
    #     features1 = [torch.zeros(size=(3,)) for _ in range(N)]
    #     for k in range(N):
    #         features1[k] = features[(k + i) % N]
    #     # print(features1)
    #     # 开始算欧氏距离，需要考虑的是计算的是两个Tensor还是两个list的欧式距离
    #     features_ = torch.stack(features)  # stack相当于是对tensor的list或者是tuple的堆叠，cat是拼接
    #     features1_ = torch.stack(features1)
    #     loss_matrix = eu_distance(features_, features1_)  # 10 *10的
    #     print(loss_matrix)
    #     loss_ = .0
    #     for j in range(N):
    #         loss_ += loss_matrix[j][j].item()
    #     losses += loss_
    # print(losses, losses / (2 * N * (N-1)))

    # 以上是特征组合模块的复现，之后的复现就是resnet返回分类结果以及特征，得到batch_size
