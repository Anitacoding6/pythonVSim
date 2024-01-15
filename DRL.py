import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import itertools
import random
#from k_shortest_paths import k_shortest_paths as ksp


class DQN(nn.Module):
    def __init__(self, num_inno, road_number_density, road_number_que, cp_number_dis):
        super(DQN, self).__init__()
        hidden_size = 128
        self.num_action_epsilon = 0
        # self.layer4 = nn.LSTM(input_size=num_states + len_send_queue + max_node_node, hidden_size=hidden_size, )
        self.layer1 = nn.Linear(num_inno+road_number_density+road_number_que+cp_number_dis, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        """ HL1 输出线性变换 """
        self.layer5 = nn.Linear(hidden_size, cp_number_dis)
        """ 我们的激活功能  """
        self.relu = nn.ReLU()

    def forward(self, x):
        """  HL1 with relu activation """
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.relu(self.layer3(out))

        out = self.layer5(out)
        return out

