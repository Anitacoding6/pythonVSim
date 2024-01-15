from DRL import DQN
from replay_memory import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

class NeuralNetwork(object):

    def __init__(self, num_inno, road_number_density, road_number_que, cp_number_dis, capacity=2000):  #capacity=16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = 0.005
        # self.ID = node_number
        '''-------wjx add-----'''
        # self.num_action_epsilon=0

        self.policy_net = DQN(num_inno, road_number_density, road_number_que, cp_number_dis).to(self.device)
        self.target_net = DQN(num_inno, road_number_density, road_number_que, cp_number_dis).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=learning_rate)