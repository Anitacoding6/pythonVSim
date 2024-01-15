import random
from collections import namedtuple
import math
import json
import os
from operator import itemgetter
import torch
import math
import json
import torch.nn.functional as F
import numpy as np

# 更改经验回放元素个数
'''创建元组类以包含经验元素'''
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward', 'terminal_flag')
)


'''
    ReplayMemory 文件定义了一个包含经验的记忆库。
     文件包含功能：
         push：将经验推入内存库
         sample：返回将数据包发送到的下一个节点
         take_recent：返回最近的经历
         take_priority：根据优先概率返回经验
         __len__：返回记忆库中的经验长度
         can_provide_sample：如果能够提供经验，则返回一个布尔值
         update_priorities：更新体验的优先级概率
'''


class ReplayMemory(object):

    # capacity = 16
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.prob_weight = np.empty(0)
        self.temp_priorities_max = 1
        self.start_provide = 0
        self.epsil = 1 ** -2

    '''将 *args 转换为 Experience 并放入 self.memory。
     如果self.memory已满，则替换已被删除的元素。
     在self.memory中最长的。 '''

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))

            # 更新优先级问题
            if self.__len__() >= self.capacity / 4:
                self.prob_weight = np.append(
                    self.prob_weight, max(self.prob_weight))
            else:
                self.prob_weight = np.append(
                    self.prob_weight, self.temp_priorities_max)
        else:
            self.position = (self.position + 1) % self.capacity
            self.memory[self.position] = Experience(*args)
            self.prob_weight[self.position] = max(self.prob_weight)

    '''从 self.memory 中随机抽取一个大小为 batch_size 的样本'''

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    '''在给定大小batch_size的情况下，请使用batch_size最新的经验。 '''

    def take_recent(self, batch_size):
        return self.memory[-batch_size:]

    '''取一个大小为 batch_size 的样本，与我们的模型更不同的样本更有可能被选中。'''

    def take_priority(self, batch_size):

        ind = random.choices(range(len(self.prob_weight)),
                             k=batch_size, weights=self.prob_weight)

        return list(itemgetter(*ind)(self.memory)), ind

    '''返回我们记忆库的长度 '''

    def __len__(self):
        return len(self.memory)

    '''如果能够提供学习经验，则返回布尔值'''

    def can_provide_sample(self, batch_size):
        # print("size: ", len(self.memory), " and ", batch_size)
        if len(self.prob_weight):
            self.temp_priorities_max = max(self.prob_weight)
        return self.__len__() >= self.capacity/10

    '''辅助函数更新优先记忆的概率'''

    def update_priorities(self, indices, cur_q, target_q):
        delta = abs(target_q-cur_q)
        self.prob_weight[indices] = delta.detach().numpy().reshape(-1)
