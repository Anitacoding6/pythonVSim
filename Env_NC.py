from Node import RNode
from Node import Vnode
from Node import Map
import numpy as np
import math
import random
import time
from Node import Interest
import copy
import gol
import golData
from wjxlog import Logpath
from neural_network import NeuralNetwork
import torch.nn.functional as F
import torch
import networkx as nx
from pylab import show
import copy
import matplotlib.pyplot as plt
from RScode import RScode
import pickle
import json
import re


def train_NC(starttime=0, endtime=20, roadvehicledensitylist=[7, 3], num_of_packets=20, num_of_packets_BG=0, num_chunks=10, trainepisodes=1000,
           learn=True):
    starttime = starttime
    endtime = endtime
    num_of_packets = num_of_packets
    num_of_packets_BG = num_of_packets_BG
    trainepisodes = trainepisodes
    roadvehicledensitylist = roadvehicledensitylist
    learn = learn
    num_chunks=num_chunks
    print(f"-----trainepisodes:{trainepisodes},starttime:{starttime},endtime:{endtime},roadvehicledensitylist:{roadvehicledensitylist},num_of_packets:{num_of_packets},num_of_packets_BG:{num_of_packets_BG},num_chunks:{num_chunks}-----")
    # train
    # 只在主程序中初始化一次
    golData._init()
    env = Env1_NC(starttime, endtime, roadvehicledensitylist, num_of_packets, num_of_packets_BG, num_chunks, learn)

    totalrewardlist = []
    for episode in range(trainepisodes):  # 5000
        # 密度改变，x1+x2=12
        # x1 = random.randint(1, 11)
        # x2 = 12 - x1
        # roadvehicledensitylist = [x1, x2]
        print("---------- Episode:", episode + 1, "roadvehicledensitylist:", roadvehicledensitylist, " ----------")
        env.resetEnv_NC(starttime, endtime, roadvehicledensitylist, num_of_packets, num_of_packets_BG, num_chunks,
                        learn)
        print("-----end---resetEnv_NC-----------------")

        # 每一个episode里有一个奖励函数,统计RSU的奖励
        # totalreward=0
        # for i in range(len(env._RNodeidlist)):
        #     totalreward=totalreward+sum(env._RNodelist[i].reward)
        #     print(f'i:{i},{env._RNodelist[i].reward}')
        print(f'env.Rnode.reward:{env.Rnode.reward}')
        totalreward = sum(env.Rnode.reward)
        totalrewardlist.append(totalreward)

        print(f'totalrewardlist:{totalrewardlist}')
        print(
            f'FIP:{env.FIPlist}\n,FDP:{env.FDPlist}\n,HCN:{env.HCNlist}\n,ISD:{env.ISDlist}\n,SIR:{env.SIRlist}\n,SIN:{env.SINlist}\n')
        print(
            f'mean_FIP:{np.mean(env.FIPlist)}\n,mean_FDP:{np.mean(env.FDPlist)}\n,mean_HCN:{np.mean(env.HCNlist)}\n,mean_ISD:{np.mean(env.ISDlist)}\n,mean_SIR:{np.mean(env.SIRlist)}\n,mean_SIN:{np.mean(env.SINlist)}\n')

    picname = 'trainepisodes_' + 'Example2' + str(env.decision_interval) + str(trainepisodes) + '_num_of_packets_' + str(
        num_of_packets) + '_' + str(starttime) + '_' + str(endtime) + '.jpg'
    print(f'totalrewardlist:{totalrewardlist}')

    # 作图1
    plt.subplot(4, 1, 1)  # 等效于plt.subplot(221)
    ISD = meanlist_pernum(env.ISDlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'ISD')
    plt.plot(np.arange(len(ISD)), ISD)
    # picnameISD = 'ISD' + picname
    # plt.savefig(picnameISD, bbox_inches='tight', dpi=450)

    # 作图1
    plt.subplot(4, 1, 2)  # 等效于plt.subplot(221)
    SIR = meanlist_pernum(env.SIRlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'SIR')
    plt.plot(np.arange(len(SIR)), SIR)
    # picnameSIR = 'SIR' + picname
    # plt.savefig(picnameSIR, bbox_inches='tight', dpi=450)

    # 作图1
    plt.subplot(4, 1, 3)  # 等效于plt.subplot(221)
    HCN = meanlist_pernum(env.HCNlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'HCN')
    plt.plot(np.arange(len(HCN)), HCN)
    # picnameHCN = 'HCN' + picname
    # plt.savefig(picnameHCN, bbox_inches='tight', dpi=450)

    plt.subplot(4, 1, 4)  # 等效于plt.subplot(221)
    rr = meanlist_pernum(totalrewardlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'reward')
    plt.plot(np.arange(len(rr)), rr)
    picnamereward = 'reward' + picname
    plt.savefig(picnamereward, bbox_inches='tight', dpi=600)
    np.save("train_reward_ft_NC.npy", rr)  # 加载数据
    np.save("totalrewardlist.npy", totalrewardlist)  # 加载数据

    plt.show()
    torch.save(env.dqn, 'DQNnetqueden_NC.pkl')


def test_NC(starttime=0, endtime=20, roadvehicledensitylist=[7, 3], num_of_packets=20, num_of_packets_BG=0, num_chunks=10, trainepisodes=1000,
           learn=False):
    starttime = starttime
    endtime = endtime
    num_of_packets = num_of_packets
    num_of_packets_BG = num_of_packets_BG
    trainepisodes = trainepisodes
    roadvehicledensitylist = roadvehicledensitylist
    learn = learn
    num_chunks=num_chunks
    print(f"-----trainepisodes:{trainepisodes},starttime:{starttime},endtime:{endtime},roadvehicledensitylist:{roadvehicledensitylist},num_of_packets:{num_of_packets},num_of_packets_BG:{num_of_packets_BG},num_chunks:{num_chunks}-----")
    # train
    # 只在主程序中初始化一次
    env = Env1_NC(starttime, endtime, roadvehicledensitylist, num_of_packets, num_of_packets_BG, num_chunks, learn)

    totalrewardlist = []
    for episode in range(trainepisodes):  # 5000
        # 密度改变，x1+x2=12
        # x1 = random.randint(1, 11)
        # x2 = 12 - x1
        # roadvehicledensitylist = [x1, x2]
        print("---------- Episode:", episode + 1, "roadvehicledensitylist:", roadvehicledensitylist, " ----------")
        env.resetEnv_NC(starttime, endtime, roadvehicledensitylist, num_of_packets, num_of_packets_BG, num_chunks,
                        learn)
        print("-----end---resetEnv_NC-----------------")

        # 每一个episode里有一个奖励函数,统计RSU的奖励
        # totalreward=0
        # for i in range(len(env._RNodeidlist)):
        #     totalreward=totalreward+sum(env._RNodelist[i].reward)
        #     print(f'i:{i},{env._RNodelist[i].reward}')
        print(f'env.Rnode.reward:{env.Rnode.reward}')
        totalreward = sum(env.Rnode.reward)
        totalrewardlist.append(totalreward)

        print(f'totalrewardlist:{totalrewardlist}')
        print(
            f'FIP:{env.FIPlist}\n,FDP:{env.FDPlist}\n,HCN:{env.HCNlist}\n,ISD:{env.ISDlist}\n,SIR:{env.SIRlist}\n,SIN:{env.SINlist}\n')
        print(
            f'mean_FIP:{np.mean(env.FIPlist)}\n,mean_FDP:{np.mean(env.FDPlist)}\n,mean_HCN:{np.mean(env.HCNlist)}\n,mean_ISD:{np.mean(env.ISDlist)}\n,mean_SIR:{np.mean(env.SIRlist)}\n,mean_SIN:{np.mean(env.SINlist)}\n')

    picname = 'trainepisodes_' + 'Example2' + str(env.decision_interval) + str(trainepisodes) + '_num_of_packets_' + str(
        num_of_packets) + '_' + str(starttime) + '_' + str(endtime) + '.jpg'
    print(f'totalrewardlist:{totalrewardlist}')

    # 作图1
    plt.subplot(4, 1, 1)  # 等效于plt.subplot(221)
    ISD = meanlist_pernum(env.ISDlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'ISD')
    plt.plot(np.arange(len(ISD)), ISD)
    picnameISD = 'ISD' + picname
    plt.savefig(picnameISD, bbox_inches='tight', dpi=450)

    # 作图1
    plt.subplot(4, 1, 2)  # 等效于plt.subplot(221)
    SIR = meanlist_pernum(env.SIRlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'SIR')
    plt.plot(np.arange(len(SIR)), SIR)
    picnameSIR = 'SIR' + picname
    plt.savefig(picnameSIR, bbox_inches='tight', dpi=450)

    # 作图1
    plt.subplot(4, 1, 3)  # 等效于plt.subplot(221)
    HCN = meanlist_pernum(env.HCNlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'HCN')
    plt.plot(np.arange(len(HCN)), HCN)
    picnameHCN = 'HCN' + picname
    plt.savefig(picnameHCN, bbox_inches='tight', dpi=450)

    plt.subplot(4, 1, 4)  # 等效于plt.subplot(221)
    rr = meanlist_pernum(totalrewardlist, 1)
    plt.xlabel(u'episode')
    plt.ylabel(u'reward')
    plt.plot(np.arange(len(rr)), rr)
    picnamereward = 'reward' + picname
    plt.savefig(picnamereward, bbox_inches='tight', dpi=600)
    np.save("train_reward_ft_NC.npy", rr)  # 加载数据
    plt.show()
    torch.save(env.dqn, 'DQNnetqueden_NC.pkl')


def meanlist_pernum(list,num):
    list_=[]
    for i in range(int(len(list)/num)):
        # print('i*num',i*num)
        q=0
        for j in range(num):
            # print('j',j)
            q=q+list[i*num+j]
        list_.append(q/num)
    return list_

class Env1_NC():
    def __init__(self,starttime=0,endtime=20,roadvehicledensitylist=[2,4,5,3],num_of_packets=0,num_of_packets_BG=0,num_chunks=5,learn=True):
        self.num_of_packets=num_of_packets
        self.num_of_packets_BG=num_of_packets_BG
        self.comm_range=500
        self.roadvehicledensitylist=roadvehicledensitylist
        self.map=Map(len(roadvehicledensitylist),self.roadvehicledensitylist)
        # print(f'-in-Env1_NC---self.map:{self.map.printmap()},roadsvehiclesidpos:{self.map.roadsvehiclesidpos}')
        # 设置车辆节点，及其位置
        self.Vnodelist = self.setVNodelist()
        self.num_of_vehicles=self.map.roadsvehiclesnum
        self.linkmatrix=np.zeros((self.num_of_vehicles,self.num_of_vehicles))
        self.setneighs()
        self.node_queue_lengths=[]
        self.curr_queue = []
        self.channellist=[]
        self.cpdict={}       #格式：name,cp id 节点集合,记录内容分布状态
        self.cp_numdict={}   #格式：name,cp中CS内缓存的内容个数 num
        self.cp_num_inno={}  #格式：name，cp innovation num；缓存中新颖数据包的个数
        # 记录第2个内容源的id，秩，新颖包的个数
        self.cpdict1={}       #格式：name,cp id 节点集合,记录内容分布状态
        self.cp_numdict1={}   #格式：name,cp中CS内缓存的内容个数 num
        self.cp_num_inno1={}  #格式：name，cp innovation num；缓存中新颖数据包的个数
        self.logpath=Logpath()
        self.resultfile = 'map4Roadv2vresult_DRL' + '.txt'
        self.starttime=starttime
        self.endtime=endtime
        self.decision_interval=0.05  #决策时间是2s
        self.functions_dict = {
            "router_mindis": self.router_mindis,
            "router_randomchoicecp": self.router_randomchoicecp,
            "router_multicp":self.router_multicp}
        self.envtime=self.starttime
        self.envtime_ = self.envtime  # 用于更新环境时间，envtime_=envtime+包的时延; 排队时延=envtime-eventtime。
        self.logpath=Logpath()
        self.file = open('totalrewardlist.txt', 'w')
        # performance evaluation
        self.FIPlist=[]
        self.FDPlist=[]
        self.ISDlist=[]
        self.SIRlist=[]
        self.SINlist=[]
        self.HCNlist=[]
        self.event_npacket={}
        #DRL
        self.RSUnodeid=self.map.intersectionRSUid
        self.Rnode=self.setRNode()
        self.meanquelist=[0,0]
        self.maxquelist=[0,0]
        # ---------NC----------
        self.num_chunks=num_chunks
        self.NCprefix = "NC/movie"
        self.event_npacket_NC={}
        self.event_npacket_NC = self.randomgeneratepacket_n_NC(self.num_of_packets,self.num_chunks,self.NCprefix)

        # 只在这里初始化神经网络，在reset()不再初始化神经网络
        if learn==True:
            print("learn==True,初始化，并训练DQN网络")
            self.dqn = self.init_dqn()  #只有DQN在第2轮+ 不初始化
            # self.dqn = torch.load('DQNnetqueden_NC.pkl')
        else:
            print("learn==False,加载，并测试DQN_NC网络")
            self.dqn = torch.load('DQNnetqueden_NC_packets20.pkl')


    def update_map_Node_neighbors(self,map):
        self.map=map
        # print(f'-update_map_Node_neighbors---self.map:{self.map.printmap()},roadsvehiclesidpos:{self.map.roadsvehiclesidpos}')
        self.Vnodelist = self.setVNodelist()
        self.num_of_vehicles=self.map.roadsvehiclesnum
        self.linkmatrix=np.zeros((self.num_of_vehicles,self.num_of_vehicles))
        self.setneighs()

    def setRNode(self):
        # 设置RSU consumer，位置在中心
        # print(f'entering setRNode.RSU id:{self.map.intersectionRSUid},pos:{self.map.intersectionpos}')
        return RNode(self.map.intersectionRSUid,self.map.intersectionpos)

    def init_dqn(self):
        # dst_onehot+cur_quelen+maxque_neigh_node_ids
        # 仅考虑密度和距离
        # self.roadvehicledensitylist+self.distancelist
        # 只在一个RSU上，部署了一个DRL
        temp_dqn = NeuralNetwork(len(self.roadvehicledensitylist), len(self.roadvehicledensitylist),len(self.roadvehicledensitylist),len(self.roadvehicledensitylist))
        return temp_dqn

    def resetEnv_NC(self,starttime,endtime,roadvehicledensitylist=[2,4,5,3],num_of_packets=0,num_of_packets_BG=0,num_chunks=5,learn=False):
        self.roadvehicledensitylist=roadvehicledensitylist
        self.num_of_packets=num_of_packets
        self.num_of_packets_BG=num_of_packets_BG
        self.map=Map(len(roadvehicledensitylist),self.roadvehicledensitylist)
        # self.map=map
        self.Vnodelist = self.setVNodelist()
        self.num_of_vehicles=self.map.roadsvehiclesnum
        self.linkmatrix=np.zeros((self.num_of_vehicles,self.num_of_vehicles))
        self.setneighs()

        self.node_queue_lengths=[]
        self.curr_queue = []
        self.channellist=[]
        self.cpdict={}      # 格式：name,cp节点集合,记录内容分布状态
        self.cpdict1={}      # 格式：name,cp节点集合,记录内容分布状态
        self.starttime=starttime
        self.endtime=endtime
        self.envtime=self.starttime
        self.envtime_ = self.envtime  # 用于更新环境时间，envtime_=envtime+包的时延; 排队时延=envtime-eventtime。
        self.event_npacket={}
        self.decision_interval=0.05  #决策时间是2s
        self.RSUnodeid=self.map.intersectionRSUid
        self.Rnode=self.setRNode()
        self.meanquelist=[0,0]
        self.maxquelist=[0,0]
        self.prefix = "movie"
        self.BGprefix = "BG"
        self.NCprefix="NC/movie"
        self.num_chunks=num_chunks

        # 每一个epsoide初始化一次，初始化gol模块，只在主程序模块使用一次。
        # 在其他地方使用时，不可重复初始化使用，会清空已配置的全局变量
        gol._init()
        # 新建或者重置跨文件全局变量：gol.set_value(变量名, 变量值）
        gol.set_value('FIP', 0)
        gol.set_value('FDP', 0)
        gol.set_value('HCN', 0)
        gol.set_value('ISD', 0)
        gol.set_value('SIR', 0)
        gol.set_value('SIN', 0)
        gol.set_value('node_queue_lengths', [0] * self.map.roadsvehiclesnum)
        gol.set_value('RSUID', self.map.roadsvehiclesnum-1)
        gol.set_value('num_chunks', self.num_chunks)

        self.event_npacket_NC={}
        self.event_npacket_NC = self.randomgeneratepacket_n_NC(self.num_of_packets,self.num_chunks,self.NCprefix)
        self.setInterest_CS_NC(self.event_npacket_NC)

        # 设置背景流量
        event_npacket_BG = self.randomgeneratepacket_n_BG(self.num_of_packets_BG,self.BGprefix)
        self.setInterest_CS(event_npacket_BG)

        with open(self.resultfile, 'a') as f:
            f.write(f'num_of_packets:{self.num_of_packets},roadvehicledensitylist:{self.roadvehicledensitylist}\n')

        self.Compute_Nodes_queue_length()
        lensum = sum(self.node_queue_lengths)
        print(f'lensum:{lensum}')
        while lensum>0:
            self.update_time()
            self.router_DRL_RSU_queden(starttime, endtime,learn)
            """
            print(f'------start------router_randomchoicecp_NC----------------------------')
            self.router_randomchoicecp_NC()
            print(f'------end------router_randomchoicecp_NC----------------------------')
            """
            lensum=sum(self.node_queue_lengths)

        FIP = gol.get_value('FIP')
        FDP= gol.get_value('FDP')
        HCN = gol.get_value('HCN')
        ISD= gol.get_value('ISD')
        SIR = gol.get_value('SIR')
        SIN = gol.get_value('SIN')
        if num_of_packets!=0:
            SIR=SIN/num_of_packets/num_chunks
        else:
            SIR=0

        self.FIPlist.append(FIP)
        self.FDPlist.append(FDP)
        self.HCNlist.append(HCN)
        self.ISDlist.append(ISD)
        self.SIRlist.append(SIR)
        self.SINlist.append(SIN)

        print(f'-----------FIP:{FIP},FDP:{FDP},HCN:{HCN},ISD:{ISD},SIR:{SIR},SIN:{SIN}')
        file = open('totalrewardlist.txt', 'a')
        file.write(f'\nFIP:{FIP},FDP:{FDP},HCN:{HCN},ISD:{ISD},SIR:{SIR},SIN:{SIN}\n')

    def get_roadsdensity(self):
        densitylist=[]
        for i in range(self.map.roadnum):
            vehiclesnum=self.map.Roadlist[i].vehiclesnum
            roadlength=self.map.Roadlist[i].roadlength
            density=vehiclesnum/(roadlength/self.comm_range)
            densitylist.append(density)
        self.roadvehicledensitylist=densitylist
        print(f'roadsdensitylist:{self.roadsdensitylist}')

    def get_roadque(self):
        quelist=[]
        for i in range(self.map.roadnum):
            quelist_roadi=[0]
            maxque=0
            minque=1000
            # print(f'self.map.Roadlist[i].vehicles:{self.map.Roadlist[i].vehicles}')
            # print(f'self.node_queue_lengths:{self.node_queue_lengths}')
            for j in self.map.Roadlist[i].vehicles:
                que=self.node_queue_lengths[j]
                if que!=0:
                    quelist_roadi.append(que)
            maxque=max(quelist_roadi)
            minque=min(quelist_roadi)
            meanque=sum(quelist_roadi)/len(quelist_roadi)
            quelist.append((maxque,minque,meanque))

            # quelist_roadi.sort()
            # print(f'road {i}:{(maxque,minque,meanque)}, maxque: {maxque}, minque:{minque}, meanque:{meanque}')
            # print(f'quelist_roadi:{quelist_roadi}')
        # print(f'quelist:{quelist}')

    # 获得不同方向最大队列长度
    def get_roadque_cps(self):
        maxquelist=[]
        meanquelist=[]
        for i in range(self.map.roadnum):
            quelist_roadi=[0]
            maxque=0
            minque=1000
            # print(f'self.map.Roadlist[i].vehicles:{self.map.Roadlist[i].vehicles}')
            # print(f'self.node_queue_lengths:{self.node_queue_lengths}')
            for j in self.map.Roadlist[i].vehicles:
                que=self.node_queue_lengths[j]
                if que!=0:
                    quelist_roadi.append(que)
            maxque=max(quelist_roadi)
            # minque=min(quelist_roadi)
            meanque=math.ceil(sum(quelist_roadi)/len(quelist_roadi))
            maxquelist.append(maxque)
            meanquelist.append(meanque)
            self.maxquelist=maxquelist
            self.meanquelist=meanquelist

            # quelist_roadi.sort()
            # print(f'road {i}:{(maxque,minque,meanque)}, maxque: {maxque}, minque:{minque}, meanque:{meanque}')
            # print(f'quelist_roadi:{quelist_roadi}')
        # print(f'maxquelist:{maxquelist},meanquelist:{meanquelist}')

    def get_hoplist(self,forwarderlists):
        hoplist=[]
        for i in forwarderlists:
            hoplist.append(len(i)-1)
        return hoplist

    # 获得通往不同cps的节点队列长度之和
    def get_quelist_cps(self,cps,forwarderlists):

        #获得到节点cps的中继节点集合列表
        # [[66, 2], [66, 43, 26, 28]]
        # print(f'node_queue_lengths:{self.node_queue_lengths}')
        # forwarderlists=self.map.get_forwarderlist(cps)
        quelist=[]
        quelist_roads=[]
        for i in range(self.map.roadnum):
            que_tocpi=0
            quelist_roadi=[]
            forwarderlists[i].remove(forwarderlists[i][0])
            for relayid in forwarderlists[i]:
                # print(f'relayid:{relayid},que:{self.node_queue_lengths[relayid]},pos:{self.map.roadsvehiclesidpos[relayid]}')
                que_tocpi=que_tocpi+self.node_queue_lengths[relayid]
                quelist_roadi.append(self.node_queue_lengths[relayid])
            quelist.append(que_tocpi)
            quelist_roads.append(quelist_roadi)
        # print(f'quelist:{quelist},quelist_roads:{quelist_roads}')
        return quelist



    def router_DRL_RSU_queden(self,starttime, endtime,will_learn=True):
        # print(f'-------router_DRL_RSU_queden----------------')
        # 在RSU侧，DRL训练,暂时先考虑距离和队列长度，不考虑密度
        # 判断当前节点是否是消费者，如果是，为每一个兴趣包选择一个内容源；
        # 如果不是，按照离cp最近的原则，选择中继节点。
        # print("entering router_DRL_RSU_que strategy")
        # 一个节点一个时隙只能处理一次兴趣包或数据包
        #一个包在一个时隙只能处理一次
        packetlist_done=[]
        '''------------------统计各个节点的队列长度------------------'''
        self.Compute_Nodes_queue_length()
        lensum=sum(self.node_queue_lengths)
        # print(f'self.envtime:{self.envtime},quelength:{self.node_queue_lengths},sum:{sum(self.node_queue_lengths)}')
        if lensum>0:
            nodeid=0
            for lenque in self.node_queue_lengths:
                if lenque>0:
                    # print(f'nodeid:{nodeid},lenque:{lenque}')
                    #处理第1个Interest。
                    # print(f'nodeid:{nodeid},self.Vnodelist[nodeid].sendqueuedict.items():{self.Vnodelist[nodeid].sendqueuedict.items()},self.Vnodelist[nodeid].recievequeuedict.items():{self.Vnodelist[nodeid].recievequeuedict.items()}')
                    dict_item_curr_queue=min(self.Vnodelist[nodeid].sendqueuedict.items(),key=lambda x:x[0])
                    eventtime=dict_item_curr_queue[0]
                    self.packet=dict_item_curr_queue[1]

                    # 判断当前的包是不是消费者，如果是消费者，需要更新内容的名字，选择数据源。
                    packettype = self.packet.get_packettype()
                    neigh = self.getneigh(nodeid)
                    # print('neigh',neigh,type(neigh))
                    neigh_num=len(neigh)
                    neighqueue_len=0
                    for nn in neigh:
                        neighqueue_len=neighqueue_len+self.node_queue_lengths[nn]
                    # print(f'nodeid:{nodeid},neigh:{neigh},neigh_len:{neigh_len},len:{len(neigh)}')
                    # print(f'nodeid:{nodeid},neighqueue_len:{neighqueue_len},neigh_num:{neigh_num}')
                    alpha=0.0011
                    reliablepacketrate=1-alpha*neighqueue_len
                    # reliablepacketrate = 1
                    # print(f'neighqueue_len:{neighqueue_len},reliablepacketrate,{reliablepacketrate}')

                    # 判断是兴趣包还是数据包
                    if packettype == "Interest":
                        flag_NCInter = re.match(r"NC", self.packet.get_name())
                        flag_NDNInter = re.match(r"NDN", self.packet.get_name())
                        consumerid=self.packet.get_consumer()
                        if flag_NCInter:
                            nameprefix=self.get_nameprefix(self.packet.get_name())
                            cps = list(self.cpdict[nameprefix])
                            num_inno_cps= self.cp_num_inno[nameprefix]
                            cps_near=list(self.cpdict1[nameprefix])
                            num_inno_cps1= self.cp_num_inno1[nameprefix]
                            # print(f'^^^^^cps:{cps},num_inno_cps:{num_inno_cps},cps_near:{cps_near},num_inno_cps1:{num_inno_cps1}')

                            #选择每个路段较近的内容源作为内容源状态
                            cps_state=[]
                            num_inno_cps_state=[]
                            for i in range(self.map.roadnum):
                                if num_inno_cps1[i]>0:
                                    cps_state.append(cps_near[i])
                                    num_inno_cps_state.append(num_inno_cps1[i])
                                else:
                                    cps_state.append(cps[i])
                                    num_inno_cps_state.append(num_inno_cps[i])
                            # print(f'roadnum:{self.map.roadnum},cps_state:{cps_state},num_inno_cps_state:{num_inno_cps_state}')

                        elif flag_NDNInter:
                            cps = list(self.cpdict[self.packet.get_name()])
                            cps_near = list(self.cpdict1[self.packet.get_name()])
                        else:
                            cps_state = list(self.cpdict[self.packet.get_name()])
                            # print(cps_state)
                            # print(f'self.cpdict:{self.cpdict},cpdict1:{self.cpdict1}')
                        # print(f'consumer:{consumerid},pos:{self.map.intersectionpos},{type(consumerid)},nodeid:{nodeid},{type(nodeid)},packet:{self.packet},cp:{cps}')
                        # print(f'Interest,cps:{cps}')

                        if int(consumerid)==int(nodeid) and len(cps_state)==3:
                            # print(f'entering len(cps_state)==3,cps_state:{cps_state}')
                            # 如果消费者侧需要处理的兴趣包，需要选择cp，并选择中继节点

                            """
                            # self.packet.cp_selected = self.getcp_randomchoice(self.packet.get_name())
                            # 选择最小距离的内容源
                            # self.packet.cp_selected=self.getcp_mindis(consumerid,self.packet.get_name())
                            # 选择道路密度最大的内容源
                            self.packet.cp_selected=list(self.cpdict[self.packet.get_name()])[0]
                            (flag, nextnodes) = self.choosenextnode_mindistance_tocp(nodeid, self.packet.cp_selected)
                          """

                            # 获取转发节点列表
                            forwarderlists = self.map.get_forwarderlist(cps_state)
                            # 获取到cps的hop
                            hop_cps = self.get_hoplist(forwarderlists)
                            cps_hop = torch.tensor(hop_cps).unsqueeze(0)
                            # 获取到cps的队列长度
                            que_cps = self.get_quelist_cps(cps_state, forwarderlists)
                            cps_que = torch.tensor(que_cps).unsqueeze(0)
                            # print(f'cps:{cps},forwarderlists:{forwarderlists},hop_cps:{hop_cps},que_cps:{que_cps}')
                            # cps = list(self.cpdict[self.packet.get_name()])

                            # if flag_NCInter:
                            #     nameprefix = self.get_nameprefix(self.packet.get_name())
                            #     cps = list(self.cpdict[nameprefix])
                            # else:
                            #     cps = list(self.cpdict[self.packet.get_name()])

                            # print(f'--start---num_inno-----')
                            # 从CS_NC中获取新颖数据包的数量，但如果连续两个Interest需要做决策，
                            # self.Vnodelist[cpi].CS_NC_inno可能没有及时更新
                            # 利用Env中更新self.cp_num_inno[self.packet.get_name()]
                            name_p = self.get_nameprefix(self.packet.get_name())
                            # num_inno_cps = self.cp_num_inno[name_p]
                            flag_inno_cps = []
                            for inno_rank in num_inno_cps_state:
                                if inno_rank>0:
                                    flag_inno_cps.append(1)
                                else:
                                    flag_inno_cps.append(0)
                            cps_num_inno=torch.tensor(num_inno_cps_state).unsqueeze(0)
                            cps_flag_inno=torch.tensor(flag_inno_cps).unsqueeze(0)

                            # print(f'--end---num_inno--num_inno_cps_state:{num_inno_cps_state}--type:{type(num_inno_cps_state)}--')
                            # print(f'--end---num_inno--cps_num_inno:{cps_num_inno}--type:{type(cps_num_inno)}--')

                            # 链路通断
                            # 车辆稀疏场景
                            dis_cps = []
                            for cpi in cps_state:
                                #     # print(f'node {cpi} positon is {self.Vnodelist[cpi].position}')
                                dis = self.cal_dis_nodes(cpi, self.map.intersectionRSUid)
                                dis_cps.append(dis)
                            dis_link_blind_spotlist = self.map.get_dis_link_blind_spotlist()
                            flag_link_blind = []
                            for iii in range(len(dis_cps)):
                                if dis_cps[iii] > dis_link_blind_spotlist[iii]:
                                    flag_link_blind.append(0)
                                else:
                                    flag_link_blind.append(1)

                            if sum(flag_link_blind) == 0:
                                # 链路均稀疏，丢包
                                print(f'没有可用链路：{flag_link_blind}')
                                flag = 0
                            else:
                                # print(f'dis_cps:{dis_cps},dis_link_blind_spotlist,{dis_link_blind_spotlist},flag_link_blind_spotlist:{flag_link_blind_spotlist}')
                                flag_link_blind_cps = torch.tensor(flag_link_blind).unsqueeze(0)
                                # delaylist_est
                                # cur_state = torch.cat((flag_link_blind_cps, cps_que, cps_hop), dim=1)
                                # print(f'name:{self.packet},cur_state[flag_link_blind+que+hop]:{cur_state}')
                                # with open(self.resultfile, 'a') as f:
                                #     f.write(f'name:{self.packet},cur_state[flag_link_blind+que+hop]:{cur_state}\n')
                                cur_state = torch.cat((cps_num_inno, flag_link_blind_cps, cps_que, cps_hop), dim=1)
                                print(
                                    f'name:{self.packet},nonce:{self.packet.get_nonce()},cur_state[num_inno+flag_link_blind+que+hop]:{cur_state}')
                                with open(self.resultfile, 'a') as f:
                                    f.write(
                                        f'name:{self.packet},cur_state[num_inno+flag_link_blind+que+hop]:{cur_state}\n')

                                # cur_state= torch.cat((flag_link_blind_cps, delaylist_est_cps, cps_hop), dim=1)
                                # print(f'name:{self.packet},cur_state[flag_link_blind+delay+hop]:{cur_state}')
                                # with open(self.resultfile, 'a') as f:
                                #     f.write(f'name:{self.packet},cur_state[flag_link_blind+delay+hop]:{cur_state}\n')
                                # print(f'cur_state:{cur_state},cps_density:{cps_density}, flag_link_blind:{flag_link_blind},que_cps:{que_cps}, cps_hop:{cps_hop},cps:{cps}')

                                # 行动空间[]
                                num = 1
                                # (action_index, action_DRL)
                                action_index, action_DRL = self.Rnode.act(self.dqn, cur_state, cps_state, num, will_learn)
                                # action_index=action[0]
                                # action_DRL=action[1]
                                self.packet.cp_selected = action_DRL
                                print(f'action_index:{action_index},action_DRL:{action_DRL}')

                                # 执行动作
                                nextnodes = []
                                (flag, nextnodes) = self.choosenextnode_mindistance_tocp_DRL(nodeid,
                                                                                         self.packet.cp_selected)
                                flag_link_act = flag_link_blind[action_index]
                                # 是否能提供新颖的包
                                flag_inno_act = flag_inno_cps[action_index]

                                # print(f'action_DRL:{action_DRL},{type(action_DRL)},nextnodes:{nextnodes},{type(nextnodes)}')
                                # 奖励
                                # nextnode 是 cp
                                if (flag_link_act==1 and flag_inno_act==1) and action_DRL in nextnodes:
                                    hopdis_cps_act = 0
                                    reward = 40
                                    terminal_flag = 1
                                    # print(f'flag_link_act==1 and action_DRL in nextnodes,action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind},delaylist_est_cps:{delaylist_est_cps},cps_hop:{cps_hop}')
                                    print(f'name:{self.packet},链路通且下一跳就是cp,action_index:{action_index},reward:{reward}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(
                                            f'name:{self.packet},link connect, and next node is cp, action_index:{action_index}, reward:{reward}\n')

                                # 找到nextnode，但不是 cp
                                elif (flag_link_act==1 and flag_inno_act==1) and action_DRL not in nextnodes:
                                    # state:cps' road density
                                    den_max = max(self.roadvehicledensitylist)
                                    den_min = min(self.roadvehicledensitylist)
                                    den_act = self.roadvehicledensitylist[action_index]

                                    # que
                                    que_act = que_cps[action_index]
                                    que_min = min(que_cps)
                                    que_max = max(que_cps)
                                    hop_min = min(hop_cps)
                                    hop_max = max(hop_cps)
                                    hop_act = hop_cps[action_index]
                                    reward_flag_link = 8 * (flag_link_blind[action_index] - min(flag_link_blind))
                                    reward_hop = (hop_min - hop_act) * 3

                                    # inno_num
                                    inno_num_min = min(num_inno_cps_state)
                                    inno_num_max = max(num_inno_cps_state)
                                    inno_num_act = num_inno_cps_state[action_index]
                                    if inno_num_act > inno_num_min:
                                        reward_num_inno = 2
                                    else:
                                        reward_num_inno = -2
                                    # inno_rank
                                    reward_flag_inno = 8 * (
                                                flag_inno_cps[action_index] - min(flag_inno_cps)) + reward_num_inno

                                    # reward_que
                                    if que_max == que_min:
                                        reward_que = 0
                                    # elif que_act - que_min >= 15:
                                    #     reward_que = -7
                                    # elif que_act - que_min >= 10:
                                    #     reward_que = -5
                                    # elif que_act - que_min >= 5:
                                    #     reward_que = -3
                                    # elif que_act - que_min > 0:
                                    #     reward_que = -2
                                    elif que_act - que_min > 0:
                                        # reward_que=-2-0.5*(que_act - que_min)
                                        reward_que=-2-3*math.floor((que_act - que_min)/5)
                                    else:
                                        reward_que = 2
                                    print(f'que_act - que_min:{que_act - que_min},reward_que:{reward_que}')

                                    reward = reward_hop + reward_que + reward_flag_link+ reward_flag_inno

                                    # reward = reward_hop + reward_delay + reward_flag
                                    # reward = reward_delay + reward_flag
                                    # print(f'flag_link_blind:{flag_link_blind},hop_cps:{hop_cps},delaylist_est:{delaylist_est}')
                                    print(
                                        f'name:{self.packet},链路通但下一跳不是cp，action_index:{action_index},reward:{reward},reward_flag_inno:{reward_flag_inno},reward_flag_link:{reward_flag_link},reward_hop:{reward_hop},reward_que:{reward_que}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(
                                            f'name:{self.packet},link connect, but next node is not cp, action_index:{action_index},reward:{reward},reward_flag_inno:{reward_flag_inno},reward_flag_link:{reward_flag_link},reward_hop:{reward_hop},reward_que:{reward_que}\n')
                                    hopdis_cps_act = hop_act - 1
                                    terminal_flag = 0
                                # 没有nextnode
                                elif flag_link_act == 0 or flag_inno_act==0:
                                    reward = -50
                                    hopdis_cps_act = 10
                                    terminal_flag = -1
                                    print(
                                        f'name:{self.packet},选择的链路断，action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(
                                            f'name:{self.packet},link interruption,action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind}\n')

                                # 记录每个时隙，Rnode的奖励
                                self.Rnode.reward.append(reward)
                                # print(f'reward:{reward},density_cps,{self.roadvehicledensitylist},hopdis_cps:{hopdis_cps},action_index:{action_index}')
                                # print(f'reward:{reward},density_cps,{self.roadvehicledensitylist},roadvehicledensitylist:{self.roadvehicledensitylist},hopdis_cps:{hopdis_cps},maxquelist:{self.maxquelist},que_cps:{que_cps},action_index:{action_index}')

                                # next_state
                                hop_cps_ = []
                                que_cps1 = []
                                for indx1 in range(self.map.roadnum):
                                    # print(f'indx1:{indx1}')
                                    if indx1 == action_index:
                                        hop_cps_.append(hopdis_cps_act)
                                        que_cps1.append(que_cps[indx1] + hop_cps[indx1])
                                    else:
                                        hop_cps_.append(hop_cps[indx1] + 1)
                                        que_cps1.append(que_cps[indx1])

                                # 假设道路车辆密度不变
                                flag_link_blind_cps_ = flag_link_blind_cps
                                cps_hop_ = torch.tensor(hop_cps_).unsqueeze(0)
                                que_cps_ = torch.tensor(que_cps1).unsqueeze(0)

                                # print(f'---@@@@@@@--before self.cp_num_inno:{self.cp_num_inno},num_inno_cps_state:{num_inno_cps_state},flag_inno_cps:{flag_inno_cps}')

                                num_inno_cps_ = []
                                flag_inno_cps_ = []
                                for jjjj in range(len(num_inno_cps_state)):
                                    if jjjj == action_index:
                                        # print(f'$$$$$$$$$$$$$')
                                        num_jj = num_inno_cps_state[jjjj] - 1
                                        num_inno_cps_.append(num_jj)
                                        if num_jj > 0:
                                            flag_inno_cps_.append(1)
                                        else:
                                            flag_inno_cps_.append(0)
                                    else:
                                        # print(f'&&&&&&&&&&&')
                                        num_inno_cps_.append(num_inno_cps_state[jjjj])
                                        flag_inno_cps_.append(flag_inno_cps[jjjj])
                                    # print(f'%num_inno_cps_state:{num_inno_cps_state}%%%jjjj:{jjjj}%%%%num_inno_cps_:{num_inno_cps_},flag_inno_cps_:{flag_inno_cps_}')

                                # self.cp_num_inno[name_p] = num_inno_cps_
                                if self.cp_num_inno1[name_p][action_index] > 0:
                                    self.cp_num_inno1[name_p][action_index] = num_inno_cps_[action_index]
                                else:
                                    self.cp_num_inno[name_p][action_index] = num_inno_cps_[action_index]

                                # print(f'-#####---name_p:{name_p},num_inno_cps_state:{num_inno_cps_state},self.cp_num_inno:{self.cp_num_inno},self.cp_num_inno1:{self.cp_num_inno1}')

                                # print(f'self.cpdict:{self.cpdict},self.cp_numdict:{self.cp_numdict},self.cp_num_inno:{self.cp_num_inno}')
                                # print(f'after self.cp_num_inno[name_p]:{self.cp_num_inno[name_p]},num_inno_cps_state:{num_inno_cps_state},flag_inno_cps_:{flag_inno_cps_},')
                                cps_flag_inno_ = torch.tensor(flag_inno_cps_).unsqueeze(0)
                                cps_num_inno_ = torch.tensor(num_inno_cps_).unsqueeze(0)
                                # next_state= torch.cat((cps_flag_inno_, flag_link_blind_cps_, que_cps_, cps_hop_), dim=1)
                                next_state = torch.cat((cps_num_inno_, flag_link_blind_cps_, que_cps_, cps_hop_), dim=1)
                                print(f'next_state[num_inno+flag_link_blind+que+hop]:{next_state}')
                                self.Rnode.learn(self.dqn, cur_state, action_index, reward, next_state, terminal_flag,
                                                 self.linkmatrix)

                        elif int(consumerid)==int(nodeid) and len(cps_state)==2:
                            # 如果消费者侧需要处理的兴趣包，需要选择cp，并选择中继节点
                            # print(f'entering len(cps)==2,cps_state:{cps_state}')

                            """
                            # self.packet.cp_selected = self.getcp_randomchoice(self.packet.get_name())
                            # 选择最小距离的内容源
                            # self.packet.cp_selected=self.getcp_mindis(consumerid,self.packet.get_name())
                            # 选择道路密度最大的内容源
                            self.packet.cp_selected=list(self.cpdict[self.packet.get_name()])[0]
                            (flag, nextnodes) = self.choosenextnode_mindistance_tocp(nodeid, self.packet.cp_selected)
                          """

                            # 获取转发节点列表
                            forwarderlists=self.map.get_forwarderlist(cps_state)

                            # 获取到cps的hop
                            hop_cps=self.get_hoplist(forwarderlists)
                            cps_hop=torch.tensor(hop_cps).unsqueeze(0)

                            # 获取到cps的队列长度
                            que_cps=self.get_quelist_cps(cps,forwarderlists)
                            cps_que=torch.tensor(que_cps).unsqueeze(0)
                            # print(f'forwarderlists:{forwarderlists},hop_cps:{hop_cps},que_cps:{que_cps}')
                            # print(f'--start---num_inno-----')
                            # 从CS_NC中获取新颖数据包的数量，但如果连续两个Interest需要做决策，
                            # self.Vnodelist[cpi].CS_NC_inno可能没有及时更新
                            # 利用Env中更新self.cp_num_inno[self.packet.get_name()]
                            name_p = self.get_nameprefix(self.packet.get_name())
                            # num_inno_cps = self.cp_num_inno[name_p]
                            flag_inno_cps = []
                            for inno_rank in num_inno_cps_state:
                                if inno_rank>0:
                                    flag_inno_cps.append(1)
                                else:
                                    flag_inno_cps.append(0)
                            cps_num_inno=torch.tensor(num_inno_cps_state).unsqueeze(0)
                            cps_flag_inno=torch.tensor(flag_inno_cps).unsqueeze(0)

                            # 链路通断
                            # 车辆稀疏场景
                            dis_cps = []
                            for cpi in cps_state:
                                #     # print(f'node {cpi} positon is {self.Vnodelist[cpi].position}')
                                dis = self.cal_dis_nodes(cpi, self.map.intersectionRSUid)
                                dis_cps.append(dis)
                            dis_link_blind_spotlist=self.map.get_dis_link_blind_spotlist()
                            flag_link_blind=[]
                            for iii in range(len(dis_cps)):
                                if dis_cps[iii]>dis_link_blind_spotlist[iii]:
                                    flag_link_blind.append(0)
                                else:
                                    flag_link_blind.append(1)

                            if sum(flag_link_blind)==0:
                                # 链路均稀疏，丢包
                                print(f'没有可用链路：{flag_link_blind}')
                                flag=0
                            else:
                                # print(f'dis_cps:{dis_cps},dis_link_blind_spotlist,{dis_link_blind_spotlist},flag_link_blind_spotlist:{flag_link_blind_spotlist}')
                                flag_link_blind_cps=torch.tensor(flag_link_blind).unsqueeze(0)

                                # delaylist_est
                                # cur_state= torch.cat((cps_flag_inno,flag_link_blind_cps, cps_que, cps_hop), dim=1)
                                cur_state= torch.cat((cps_num_inno,flag_link_blind_cps, cps_que, cps_hop), dim=1)

                                print(f'name:{self.packet},nonce:{self.packet.get_nonce()},cur_state[num_inno+flag_link_blind+que+hop]:{cur_state}')
                                with open(self.resultfile, 'a') as f:
                                    f.write(f'name:{self.packet},cur_state[num_inno+flag_link_blind+que+hop]:{cur_state}\n')

                                # 行动空间[]
                                num=1
                                # (action_index, action_DRL)
                                action_index,action_DRL=self.Rnode.act(self.dqn,cur_state,cps_state,num,will_learn)
                                self.packet.cp_selected=action_DRL
                                print(f'action_index:{action_index},action_DRL:{action_DRL}')

                                # 执行动作
                                nextnodes = []
                                (flag, nextnodes) = self.choosenextnode_mindistance_tocp_DRL(nodeid, self.packet.cp_selected)
                                print(f'nextnodes:{nextnodes}')
                                flag_link_act=flag_link_blind[action_index]
                                # 是否能提供新颖的包
                                flag_inno_act=flag_inno_cps[action_index]
                                # print(f'action_DRL:{action_DRL},{type(action_DRL)},nextnodes:{nextnodes},{type(nextnodes)}')
                                # 奖励
                                # nextnode 是 cp
                                if (flag_link_act==1 and flag_inno_act==1) and action_DRL in nextnodes:
                                    hopdis_cps_act=0
                                    reward=40
                                    terminal_flag=1
                                    # print(f'flag_link_act==1 and action_DRL in nextnodes,action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind},delaylist_est_cps:{delaylist_est_cps},cps_hop:{cps_hop}')
                                    print(f'name:{self.packet},链路通且下一跳就是cp,action_index:{action_index},reward:{reward}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(f'name:{self.packet},link connect, and next node is cp, action_index:{action_index}, reward:{reward}\n')

                                # 找到nextnode，但不是 cp
                                elif (flag_link_act==1 and flag_inno_act==1) and action_DRL not in nextnodes:
                                    # state:cps' road density
                                    den_max = max(self.roadvehicledensitylist)
                                    den_min = min(self.roadvehicledensitylist)
                                    den_act = self.roadvehicledensitylist[action_index]

                                    # que
                                    que_act=que_cps[action_index]
                                    que_min=min(que_cps)
                                    que_max=max(que_cps)

                                    # hop
                                    hop_min = min(hop_cps)
                                    hop_max=max(hop_cps)
                                    hop_act = hop_cps[action_index]
                                    reward_flag_link=8*(flag_link_blind[action_index]-min(flag_link_blind))
                                    reward_hop=(hop_max - hop_act)*2

                                    # inno_num
                                    inno_num_min=min(num_inno_cps_state)
                                    inno_num_max=max(num_inno_cps_state)
                                    inno_num_act=num_inno_cps_state[action_index]
                                    if inno_num_act>inno_num_min:
                                        reward_num_inno=2
                                    else:
                                        reward_num_inno=-2
                                    # inno_rank
                                    reward_flag_inno=8*(flag_inno_cps[action_index]-min(flag_inno_cps))+reward_num_inno

                                    # reward_que
                                    if que_max==que_min:
                                        reward_que=0
                                    elif que_act-que_min>=15:
                                        reward_que=-7
                                    elif que_act-que_min>=10:
                                        reward_que=-5
                                    elif que_act-que_min>=5:
                                        reward_que=-3
                                    elif que_act-que_min>0:
                                        reward_que=-2
                                    else:
                                        reward_que=2

                                    reward = reward_hop + reward_que + reward_flag_link + reward_flag_inno

                                    # reward = reward_hop + reward_delay + reward_flag
                                    # reward = reward_delay + reward_flag
                                    # print(f'flag_link_blind:{flag_link_blind},hop_cps:{hop_cps},delaylist_est:{delaylist_est}')
                                    print(f'name:{self.packet},链路通但下一跳不是cp，action_index:{action_index},reward:{reward},reward_flag_inno:{reward_flag_inno},reward_flag_link:{reward_flag_link},reward_hop:{reward_hop},reward_que:{reward_que}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(f'name:{self.packet},link connect, but next node is not cp, action_index:{action_index},reward:{reward},reward_flag_inno:{reward_flag_inno},reward_flag_link:{reward_flag_link},reward_hop:{reward_hop},reward_que:{reward_que}\n')

                                    hopdis_cps_act=hop_act-1
                                    terminal_flag=0
                                # 没有nextnode
                                elif flag_link_act==0 or flag_inno_act==0:
                                    reward=-50
                                    hopdis_cps_act=10
                                    terminal_flag=-1
                                    print(f'name:{self.packet},选择的链路断或没有内容，action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind}')
                                    with open(self.resultfile, 'a') as f:
                                        f.write(f'name:{self.packet},link interruption,action_index:{action_index},reward:{reward},flag_link_blind:{flag_link_blind}\n')

                                # 记录每个时隙，Rnode的奖励
                                self.Rnode.reward.append(reward)
                                # print(f'reward:{reward},density_cps,{self.roadvehicledensitylist},hopdis_cps:{hopdis_cps},action_index:{action_index}')
                                # print(f'reward:{reward},density_cps,{self.roadvehicledensitylist},roadvehicledensitylist:{self.roadvehicledensitylist},hopdis_cps:{hopdis_cps},maxquelist:{self.maxquelist},que_cps:{que_cps},action_index:{action_index}')

                                # next_state
                                hop_cps_=[]
                                que_cps1=[]
                                for indx1 in range(self.map.roadnum):
                                    # print(f'indx1:{indx1}')
                                    if indx1 == action_index:
                                        hop_cps_.append(hopdis_cps_act)
                                        que_cps1.append(que_cps[indx1]+hop_cps[indx1])
                                    else:
                                        hop_cps_.append(hop_cps[indx1]+1)
                                        que_cps1.append(que_cps[indx1])

                                # 假设道路车辆密度不变
                                flag_link_blind_cps_=flag_link_blind_cps
                                cps_hop_=torch.tensor(hop_cps_).unsqueeze(0)
                                que_cps_=torch.tensor(que_cps1).unsqueeze(0)

                                # print(f'---@@@@@@@--before self.cp_num_inno:{self.cp_num_inno},num_inno_cps_state:{num_inno_cps_state},flag_inno_cps:{flag_inno_cps}')

                                num_inno_cps_=[]
                                flag_inno_cps_=[]
                                for jjjj in range(len(num_inno_cps_state)):
                                    if jjjj==action_index:
                                        # print(f'$$$$$$$$$$$$$')
                                        num_jj=num_inno_cps_state[jjjj]-1
                                        num_inno_cps_.append(num_jj)
                                        if num_jj>0:
                                            flag_inno_cps_.append(1)
                                        else:
                                            flag_inno_cps_.append(0)
                                    else:
                                        # print(f'&&&&&&&&&&&')
                                        num_inno_cps_.append(num_inno_cps_state[jjjj])
                                        flag_inno_cps_.append(flag_inno_cps[jjjj])
                                    # print(f'%num_inno_cps_state:{num_inno_cps_state}%%%jjjj:{jjjj}%%%%num_inno_cps_:{num_inno_cps_},flag_inno_cps_:{flag_inno_cps_}')

                                # 更新cp_num_inno
                                if self.cp_num_inno1[name_p][action_index] > 0:
                                    self.cp_num_inno1[name_p][action_index] = num_inno_cps_[action_index]
                                else:
                                    self.cp_num_inno[name_p][action_index] = num_inno_cps_[action_index]
                                # print(f'-#####---name_p:{name_p},num_inno_cps_state:{num_inno_cps_state},self.cp_num_inno:{self.cp_num_inno},self.cp_num_inno1:{self.cp_num_inno1}')

                                # print(f'self.cpdict:{self.cpdict},self.cp_numdict:{self.cp_numdict},self.cp_num_inno:{self.cp_num_inno}')
                                # print(f'after self.cp_num_inno[name_p]:{self.cp_num_inno[name_p]},num_inno_cps_state:{num_inno_cps_state},flag_inno_cps_:{flag_inno_cps_},')
                                cps_flag_inno_ = torch.tensor(flag_inno_cps_).unsqueeze(0)
                                cps_num_inno_=torch.tensor(num_inno_cps_).unsqueeze(0)
                                # next_state= torch.cat((cps_flag_inno_, flag_link_blind_cps_, que_cps_, cps_hop_), dim=1)
                                next_state= torch.cat((cps_num_inno_, flag_link_blind_cps_, que_cps_, cps_hop_), dim=1)
                                print(f'next_state[num_inno+flag_link_blind+que+hop]:{next_state}')
                                self.Rnode.learn(self.dqn,cur_state, action_index, reward, next_state,terminal_flag,self.linkmatrix)

                        elif int(consumerid) == int(nodeid) and len(cps_state) == 1:
                            # print(f'entering len(cps_state)==1,cps_state:{cps_state}, background packet')
                            self.packet.cp_selected=cps_state[0]
                            # 选择距离cp更近的节点
                            nextnodes = []
                            (flag, nextnodes) = self.choosenextnode_mindistance_tocp_DRL(nodeid, self.packet.cp_selected)
                            # print(f'nodeid:{nodeid},nextnodes:{nextnodes},roadforwarderlist:{self.map.get_roadforwarderlist()}')

                        else:
                            # print(f'entering node is not consumer')

                            # 不是消费者侧，仅需选择中继节点
                            # 选择距离cp更近的节点
                            nextnodes = []
                            (flag, nextnodes) = self.choosenextnode_mindistance_tocp_DRL(nodeid, self.packet.cp_selected)
                            # print(f'nodeid:{nodeid},nextnodes:{nextnodes}')
                            # print(f'cpid_randomchoice:{cpid_randomchoice}')
                            # print(f'consumerid==nodeid,cp,{self.packet.cp_selected}')

                        # print(f'cp_selected：{self.packet.cp_selected}，flag:{flag},nextnodes:{nextnodes}')
                        if flag:
                            if len(nextnodes)>0:
                                if random.uniform(0, 1) < reliablepacketrate:
                                    for nextnodeidi in nextnodes:
                                        current_packet=copy.deepcopy(self.packet)
                                        # print(f'nodeid:{nodeid},neigh id:{j}')
                                        current_packet.set_nextnode(self.Vnodelist[nextnodeidi])
                                        self.Vnodelist[nextnodeidi].onIncomingInterest(current_packet,nextnodeidi)
                                        # 更新path
                                        self.packet.update_path(nextnodeidi)
                                        # print('self.packet.path',self.packet.path)
                        del self.Vnodelist[nodeid].sendqueuedict[eventtime]

                    elif packettype == "Data":
                        if random.uniform(0, 1) < reliablepacketrate:
                            for jj in neigh:
                                current_packet1=copy.deepcopy(self.packet)
                                # print(f'nodeid:{nodeid},jj:{jj}')
                                current_packet1.set_nextnode(self.Vnodelist[jj])
                                self.Vnodelist[jj].onIncomingData(current_packet1)
                            del self.Vnodelist[nodeid].sendqueuedict[eventtime]
                nodeid=nodeid+1
            # 更新车辆位置，链路矩阵
            # print(f'self.envtime:{self.envtime},quelength:{self.node_queue_lengths},sum:{sum(self.node_queue_lengths)}')


    def update_time(self):
        self.envtime=self.envtime+self.decision_interval
        return self.envtime

    #设置车辆节点
    def setVNodelist(self):
        # print(f'entering setVNodelist, self.map.roadsvehiclesnum:{self.map.roadsvehiclesnum},self.map.roadsvehiclesidpos:{self.map.roadsvehiclesidpos}')
        Vnodelist=[]
        for i in range(self.map.roadsvehiclesnum):
            Vnodelist.append(Vnode(i,self.map.roadsvehiclesidpos[i]))
        # 设置RSU consumer，位置在中心
        # print(f'RSU id:{self.map.roadsvehiclesnum},pos:{self.map.intersectionpos}')

        # for i in range(self.map.roadsvehiclesnum):
        #     print(f'printVNode,nodeid\t{Vnodelist[i].nodeid},position\t{Vnodelist[i].position},neighbors\t{Vnodelist[i].neighbors}')
        # Vnodelist.append(Vnode(self.map.roadsvehiclesnum,self.map.intersectionpos))
        return Vnodelist

    def printVNode(self):
        # self.map.printmap()
        for i in range(self.map.roadsvehiclesnum):
            print(f'printVNode,nodeid\t{self.Vnodelist[i].nodeid},position\t{self.Vnodelist[i].position},neighbors\t{self.Vnodelist[i].neighbors}')

    #设置链路矩阵，节点i，j，如果在通信范围之内，1，不在通信范围之内，float("-inf")，同一节点0。
    def setlinkmatrix(self):
        for ii in range(0,self.num_of_vehicles):
            for jj in range(ii,self.num_of_vehicles):
                # print(f'{ii},{self.Vnodelist[ii].position},{self.Vnodelist[ii].position[0]},{self.Vnodelist[ii].position[1]}')
                d=math.sqrt((self.Vnodelist[ii].position[0]-self.Vnodelist[jj].position[0])**2+(self.Vnodelist[ii].position[1]-self.Vnodelist[jj].position[1])**2)
                # print(f'setlinkmatrix,{ii},{jj},{d}')
                if ii!=jj:
                    if d<self.comm_range:
                        self.linkmatrix[ii][jj]=1
                        self.linkmatrix[jj][ii] = 1
                    else:
                        self.linkmatrix[ii][jj]=float("-inf")
                        self.linkmatrix[jj][ii] = float("-inf")
                else:
                    self.linkmatrix[ii][jj]=0
        # print(f'self.linkmatrix\n{self.linkmatrix}')

    #根据链路矩阵更新车辆的邻居节点
    def setneighs(self):
        # print(f'entering setneighs,num_of_vehicles:{self.num_of_vehicles}')
        self.setlinkmatrix()
        for vid in range(0,self.num_of_vehicles):
            a=self.linkmatrix[vid]
            neigh_vid=[]
            indx=0
            for i in a:
                if i==1:
                    neigh_vid.append(indx)
                indx=indx+1
            self.Vnodelist[vid].neighbors=neigh_vid

    #根据链路矩阵获得车辆vid的邻居车辆
    def getneigh(self,vid):
        a=self.linkmatrix[vid]
        neigh_vid=[]
        indx=0
        for i in a:
            if i==1:
                neigh_vid.append(indx)
            indx=indx+1
        self.Vnodelist[vid].neighbors=neigh_vid
        return neigh_vid
        # print(ll) #[0, 2, 3, 4, 5, 6]


    def randomgeneratepacket_n_BG(self,num_of_packets=20,prefix="BG"):
        # print(f'entering randomgeneratepacket_n')
        event_npacket={}
        # print('self.map.roadsvehiclesidpos',self.map.roadsvehiclesidpos)

        """
        # 以30%,70%的概率选择道路内容源
        rd = self.roadvehicledensitylist
        road_BG=random.choices([0,1], weights=rd, k=num_of_packets)
        print(f'rr_BG:{road_BG}')        
        """

        # 以100%的概率选择道路密度最大的内容源
        roadid=len(self.roadvehicledensitylist)-1
        road_BG = [roadid for _ in range(num_of_packets)]
        # print(f'rr_BG:{road_BG}')
        for i in range(num_of_packets): #共n个包
            # print(i) #如果n=10,i为 0-9
            #name
            # prefix="movie"
            contnam=prefix+"/"+str(i)
            # print('contnam',contnam)
            #nonce
            nonce = random.randint(1, 100000000)
            # print('nonce\t',nonce,'\ti\t',i )
            # consumer='0'
            # consumer=str(random.randint(0,self.num_of_vehicles-1))
            consumer = str(self.num_of_vehicles-1)

            # 选择随机选择道路密度最大的车辆作为内容源
            roadi=road_BG[i]
            cplist=[]
            vcp=self.map.Roadlist[roadi].vehi_randomchoice()
            cplist.append(vcp)
            # print(f'roadlist:{roadlist},prl:{prl},roadi:{roadi},vcp:{vcp}')
            self.cpdict[contnam]=cplist
            sendtime=self.starttime+(self.endtime-self.starttime)*i/num_of_packets
            # print(f'BG,roadi:{roadi},vcp:{vcp},sendtime:{sendtime}')
            # print(f'第{i}个包\tconsumer:{consumer}\tcplist:{cplist}\tsendtime:{sendtime}')
            event_npacket[i]=(sendtime,contnam,nonce,consumer,cplist)
        return event_npacket


    #随机生成n个包，一个RSU节点和三个cp节点匹配
    #RSU节点位于map的中心（3000，3000）
    #三个cp节点分别位于三条道路上
    def randomgeneratepacket_n(self,num_of_packets=20,prefix="movie"):
        # print(f'entering randomgeneratepacket_n')
        event_npacket={}
        # print('self.map.roadsvehiclesidpos',self.map.roadsvehiclesidpos)
        for i in range(num_of_packets): #共n个包
            # print(i) #如果n=10,i为 0-9
            #name
            # prefix="movie"
            contnam=prefix+"/"+str(i)
            # print('contnam',contnam)
            #nonce
            nonce = random.randint(1, 100000000)
            consumer = str(self.num_of_vehicles-1)
            cplist=[]
            for j in range(self.map.roadnum):
                vcp=self.map.Roadlist[j].vehi_randomchoice()
                cplist.append(vcp)
                # print(f'vcp:{vcp}\tpos:{self.Vnodelist[vcp].position}')
            # print(f'cplist:{cplist}')
            self.cpdict[contnam]=cplist
            sendtime=self.starttime+(self.endtime-self.starttime)*i/self.num_of_packets
            # print(f'第{i}个包\tconsumer:{consumer}\tcplist:{cplist}\tsendtime:{sendtime}')
            event_npacket[i]=(sendtime,contnam,nonce,consumer,cplist)
        return event_npacket

    def randomgeneratepacket_n_NC(self,num_of_packets=20,num_chunks=5,NCprefix="NC/movie"):
        # print(f'entering randomgeneratepacket_n')
        event_npacket={}
        # print('self.map.roadsvehiclesidpos',self.map.roadsvehiclesidpos)
        # 采用网络编码后，编码包分为num_of_packets代，每一代有num_chunk个块,同一代内的数据包编码
        # 与原始VNDN不同，要为每一代的每一个包设置事件的时间等

        for i in range(num_of_packets): #共n个代
            """-------------某一个前缀---设置一个内容源，设置多个内容缓存------------------------"""
            contnampref = NCprefix + "/" + str(i)
            golData.set_value(contnampref,[[]])
            cplist = []
            num_cs_cplist = []
            # print(f'roadvehicledensity:{self.map.roadvehicledensitylist}')
            for j in range(self.map.roadnum):
                vcp=self.map.Roadlist[j].vehi_choicedismax()
                cplist.append(vcp)
                # print(f'*******---road {j},vcp:{vcp},{self.Vnodelist[vcp].position}')
                # self.map.roadvehicledensitylist
                # 密度最大，内容多, 内容和加起来大于n
                # if self.map.roadvehicledensitylist[j] == max(self.map.roadvehicledensitylist):
                if self.map.roadvehicledensitylist[j] == self.map.roadvehicledensitylist[1]:
                    # num_cs_cp = random.randint(int(0.8 * num_chunks), num_chunks)
                    num_cs_cp = random.randint(int(0.6 * num_chunks), num_chunks)
                    # num_cs_cp = num_chunks
                else:
                    num_cs_cp = random.randint(0, int(0.5 * num_chunks))
                    # num_cs_cp = num_chunks

                num_cs_cplist.append(num_cs_cp)

            cplist1 = []
            num_cs_cplist1 = []
            # print(f'roadvehicledensity:{self.map.roadvehicledensitylist}')
            for j in range(self.map.roadnum):
                vcp1 = self.map.Roadlist[j].vehi_randomchoice()
                if vcp1 == cplist[j]:
                    vcp1 = self.map.Roadlist[j].vehi_randomchoice()
                cplist1.append(vcp1)
                # self.map.roadvehicledensitylist
                # 密度最大，内容多, 内容和加起来大于n
                # if self.map.roadvehicledensitylist[j] == max(self.map.roadvehicledensitylist):
                if self.map.roadvehicledensitylist[j] == self.map.roadvehicledensitylist[1]:
                    num_cs_cp1 = random.randint(0.2*num_chunks, 0.8*num_chunks)
                    # num_cs_cp1 = random.randint(int(0.6 * num_chunks), 0.8*num_chunks)
                    # num_cs_cp1 = num_chunks

                else:
                    num_cs_cp1 = random.randint(0, int(0.5 * num_chunks))
                    # num_cs_cp1 = num_chunks

                num_cs_cplist1.append(num_cs_cp1)
            # print(f'vcp:{vcp}\tpos:{self.Vnodelist[vcp].position}')
            # print(f'cplist:{cplist},num_cs_cplist:{num_cs_cplist}')
            # self.cpdict[contnampref] = cplist

            for chunk in range(num_chunks):
                pac=i*num_chunks+chunk
                contnam=contnampref+"/" + str(chunk)

                # print(f'第{i}代里的第{chunk}块，第{pac}个包') #如果n=10,i为 0-9
                # 第0代里的第0块，第0个包
                # 第0代里的第1块，第1个包
                #name
                # prefix="movie"
                # print('contnam',contnam)
                #nonce
                nonce = random.randint(1, 100000000)
                # print('nonce\t',nonce,'\ti\t',i )
                # consumer='0'
                # consumer=str(random.randint(0,self.num_of_vehicles-1))
                consumer = str(self.num_of_vehicles-1)

                # sendtime=self.starttime+(self.endtime-self.starttime)*i/self.num_of_packets
                sendtime=self.starttime+(self.endtime-self.starttime)*pac/(self.num_of_packets*num_chunks)

                # print(f'第{i}代里的第{chunk}块，第{pac}个包,name:{contnam}\tconsumer:{consumer}\tcplist:{cplist}\tnum_cs_cplist:{num_cs_cplist}\tsendtime:{sendtime}')
                event_npacket[pac]=(sendtime,contnam,nonce,consumer,cplist,num_cs_cplist,cplist1,num_cs_cplist1)
        return event_npacket

    def get_nameprefix(self,name):
        na = name.split("/")
        # ['NC', 'movie', '1', '5']
        # ['NC', 'movie', '10', '5']
        # print(na)
        nameprefix = ""
        for i in range(len(na) - 1):
            nameprefix = nameprefix + na[i] + "/"
            # print(nameprefix)
        nap = nameprefix[:-1]
        # print(f'name:{name},nameprefix:{nap}')
        return nap


    # event_npacket[pac] = (sendtime, contnam, nonce, consumer, cplist, num_cs_cplist, cplist1, num_cs_cplist1)
    def setInterest_CS_NC(self,event_npacket):
        # 设置在什么时间，请求什么名字的兴趣包，其内容提供者是谁，有几个内容。
        # 编码数据包不同，
        self.event_npacket=event_npacket
        print(f'roadsvehiclesidpos:{self.map.roadsvehiclesidpos}')

        for i in event_npacket.keys():  # 共n个包, pac=i*num_chunks+chunk
            # print(f'i:{i},{type(i)}')
            # print(f'{event_npacket[i]}')
            # eventtime
            eventtime = event_npacket[i][0]
            contnam = event_npacket[i][1]
            nameprefix = self.get_nameprefix(contnam)

            nonce = event_npacket[i][2]
            consumer = event_npacket[i][3]
            sendtime = eventtime

            # 最远的内容源
            cplist = event_npacket[i][4]
            self.cpdict[nameprefix] = cplist  # 某名字前缀对应的cp
            cp_numlist = event_npacket[i][5]
            self.cp_numdict[nameprefix] = cp_numlist
            self.cp_num_inno[nameprefix] = cp_numlist
            # 较近的内容源
            cplist1 = event_npacket[i][6]
            self.cpdict1[nameprefix] = cplist1  # 某名字前缀对应的cp
            cp_numlist1 = event_npacket[i][7]
            self.cp_numdict1[nameprefix] = cp_numlist1
            self.cp_num_inno1[nameprefix] = cp_numlist1
            #set CS
            # for vcp in cplist:
                # self.Vnodelist[vcp].addCS(contnam)
            # print(f'self.cpdict:{self.cpdict},self.cpdict1:{self.cpdict1}')

            # set CS_NC
            # 第nn个数据源的cp id：cplist[nn]]；
            for nn in range(len(cplist)):
                num_chunks = gol.get_value('num_chunks')
                self.Vnodelist[cplist[nn]].initCS_num_random_NC(nameprefix, cp_numlist[nn], num_chunks)

            # set CS_NC
            # 第mm个数据源的cp id：cplist1[mm]]；
            for mm in range(len(cplist1)):
                num_chunks = gol.get_value('num_chunks')
                self.Vnodelist[cplist1[mm]].initCS_num_random_NC(nameprefix, cp_numlist1[mm], num_chunks)


            # set interest
            lastnodeid='999'
            currentnodeid='app'
            nextnodeid=consumer

            # consumer
            # print(f'consumer CS_NC:{self.Vnodelist[int(consumer)].CS_NC},nameprefix:{nameprefix}')
            # if nameprefix in self.Vnodelist[int(consumer)].CS_NC.keys():
            #     coefs_cs = self.Vnodelist[int(consumer)].CS_NC[nameprefix[0:10]]
            #     contnam = nameprefix + np.array2string(coefs_cs)
            # else:
            #     contnam=nameprefix

            # (name, nonce, consumer, sendtime, lastnodeid, currentnodeid, nextnodeid):
            # interest1 = Interest('movie/1', 1289, 123, 1, '999', 'app', '1')
            interest1 = Interest(eventtime,contnam, nonce, consumer, sendtime, lastnodeid, currentnodeid, nextnodeid)
            interest1.update_path(int(consumer))
            self.Vnodelist[int(consumer)].onIncomingInterest(interest1)

        # print(f'setInterest_CS self.node_queue_lengths:{self.node_queue_lengths}')
            # print(f'self.Vnodelist[int(consumer)].sendqueuedictf:{self.Vnodelist[int(consumer)].sendqueuedict},{self.Vnodelist[int(consumer)].recievequeuedict}')
            # print(f'self.Vnodelist[vcp].sendqueuedict:{self.Vnodelist[vcp].sendqueuedict},{self.Vnodelist[vcp].recievequeuedict}')


    def setInterest_CS(self,event_npacket):
        self.event_npacket=event_npacket
        for i in range(len(event_npacket)):  # 共n个包
            #eventtime
            eventtime=event_npacket[i][0]
            contnam=event_npacket[i][1]
            nonce=event_npacket[i][2]
            consumer=event_npacket[i][3]
            sendtime=eventtime
            cplist=event_npacket[i][4]
            self.cpdict[contnam]=cplist

            #set CS
            for vcp in cplist:
                self.Vnodelist[vcp].addCS(contnam)
            # set interest
            lastnodeid='999'
            currentnodeid='app'
            nextnodeid=consumer
            # (name, nonce, consumer, sendtime, lastnodeid, currentnodeid, nextnodeid):
            # interest1 = Interest('movie/1', 1289, 123, 1, '999', 'app', '1')
            interest1 = Interest(eventtime,contnam, nonce, consumer, sendtime, lastnodeid, currentnodeid, nextnodeid)
            interest1.update_path(int(consumer))
            self.Vnodelist[int(consumer)].onIncomingInterest(interest1)

        # print(f'setInterest_CS self.node_queue_lengths:{self.node_queue_lengths}')
            # print(f'self.Vnodelist[int(consumer)].sendqueuedictf:{self.Vnodelist[int(consumer)].sendqueuedict},{self.Vnodelist[int(consumer)].recievequeuedict}')
            # print(f'self.Vnodelist[vcp].sendqueuedict:{self.Vnodelist[vcp].sendqueuedict},{self.Vnodelist[vcp].recievequeuedict}')



    #从当前节点中的邻居节点中选择到cp距离更近的节点
    def choosenextnode_mindistance_tocp1(self,nodeid,cpid):
        # print(f'nodeid:{nodeid},pos:{self.map.roadsvehiclesidpos[nodeid]}')
        neighs = self.getneigh(nodeid)
        # print(f'neighs:{neighs}')

        flag_NCInter = re.match(r"NC", self.packet.get_name())
        # print(flag_NCInter,interest.get_name(),type(interest.get_name()))
        if flag_NCInter:
            cplist=self.cpdict[self.get_nameprefix(self.packet.get_name())]
        else:
            cplist=self.cpdict[self.packet.get_name()]
        cp=[i for i in set(cplist) if i in set(neighs)]
        flag=0
        if len(cp)>0:
            # print(f'下一跳是cp')
            nextnodeid=cp
            flag=1
            # print(f'nodeid：{nodeid}，cplist:{cplist},neighs:{neighs},cp:{cp}')
        else:
            # print(f'选择中继节点')
            nextnodeid = []
            # 选择距离cp最近的邻居
            distocp_min=self.cal_dis_nodes(nodeid,cpid)
            neitocp_min=9999
            for nei in neighs:
                nipos = self.Vnodelist[nei].getposition()
                # print(f'nei:{nei},pos:{nipos}')
                # dis=math.sqrt((nipos[0]-pos_cp[0])**2+(nipos[1]-pos_cp[1])**2)
                dis=self.cal_dis_nodes(cpid,nei)
                if dis<distocp_min:
                    distocp_min=dis
                    neitocp_min=nei
                    flag=1
            if flag:
                nextnodeid.append(neitocp_min)

        if len(nextnodeid)==0:
            # print(f'nodeid:{nodeid},position:{self.Vnodelist[nodeid].position},cp position:{self.Vnodelist[cpid].position},neighs:{neighs}')
            # for nei1 in neighs:
            #     print(f'neighbor:{nei1},nei position:{self.Vnodelist[nei1].position}')
            return (0,[9999])
        else:
            # print(f'nextnodeid:{nextnodeid},pos:{self.Vnodelist[nextnodeid[0]].position}')
            return (1,nextnodeid)


    #从当前节点中的邻居节点中选择到cp距离更近的节点
    # 不是cps距离更近的节点
    def choosenextnode_mindistance_tocp_DRL(self,nodeid,cpid):
        # print(f'nodeid:{nodeid},pos:{self.map.roadsvehiclesidpos[nodeid]}')
        neighs = self.getneigh(nodeid)
        # print(f'neighs:{neighs}')

        if cpid in set(neighs):
            flag=1
        else:
            flag=0
        nextnodeid = []
        if flag==1:
            # print(f'下一跳是cp')
            nextnodeid.append(cpid)
            # print(f'nodeid：{nodeid}，cplist:{cplist},neighs:{neighs},cp:{cp}')
        else:
            # print(f'选择中继节点')
            # 选择距离cp最近的邻居
            distocp_min=self.cal_dis_nodes(nodeid,cpid)
            neitocp_min=9999
            for nei in neighs:
                nipos = self.Vnodelist[nei].getposition()
                # print(f'nei:{nei},pos:{nipos}')
                # dis=math.sqrt((nipos[0]-pos_cp[0])**2+(nipos[1]-pos_cp[1])**2)
                dis=self.cal_dis_nodes(cpid,nei)
                if dis<distocp_min:
                    distocp_min=dis
                    neitocp_min=nei
                    flag=1
            if flag:
                nextnodeid.append(neitocp_min)

        if len(nextnodeid)==0:
            # print(f'nodeid:{nodeid},position:{self.Vnodelist[nodeid].position},cp position:{self.Vnodelist[cpid].position},neighs:{neighs}')
            # for nei1 in neighs:
            #     print(f'neighbor:{nei1},nei position:{self.Vnodelist[nei1].position}')
            return (0,[9999])
        else:
            # print(f'nextnodeid:{nextnodeid},pos:{self.Vnodelist[nextnodeid[0]].position}')
            return (1,nextnodeid)


    #从当前节点中的邻居节点中选择到cp距离更近的节点
    def choosenextnode_mindistance_tocp(self,nodeid,cpid):
        # print(f'nodeid:{nodeid},pos:{self.map.roadsvehiclesidpos[nodeid]}')
        neighs = self.getneigh(nodeid)
        # print(f'neighs:{neighs}')

        flag_NCInter = re.match(r"NC", self.packet.get_name())
        # print(flag_NCInter,interest.get_name(),type(interest.get_name()))
        if flag_NCInter:
            cplist=self.cpdict[self.get_nameprefix(self.packet.get_name())]
        else:
            cplist=self.cpdict[self.packet.get_name()]
        cp=[i for i in set(cplist) if i in set(neighs)]
        flag=0
        if len(cp)>0:
            # print(f'下一跳是cp')
            nextnodeid=cp
            flag=1
            # print(f'nodeid：{nodeid}，cplist:{cplist},neighs:{neighs},cp:{cp}')
        else:
            # print(f'选择中继节点')
            nextnodeid = []
            # 选择距离cp最近的邻居
            distocp_min=self.cal_dis_nodes(nodeid,cpid)
            neitocp_min=9999
            for nei in neighs:
                nipos = self.Vnodelist[nei].getposition()
                # print(f'nei:{nei},pos:{nipos}')
                # dis=math.sqrt((nipos[0]-pos_cp[0])**2+(nipos[1]-pos_cp[1])**2)
                dis=self.cal_dis_nodes(cpid,nei)
                if dis<distocp_min:
                    distocp_min=dis
                    neitocp_min=nei
                    flag=1
            if flag:
                nextnodeid.append(neitocp_min)

        if len(nextnodeid)==0:
            # print(f'nodeid:{nodeid},position:{self.Vnodelist[nodeid].position},cp position:{self.Vnodelist[cpid].position},neighs:{neighs}')
            # for nei1 in neighs:
            #     print(f'neighbor:{nei1},nei position:{self.Vnodelist[nei1].position}')
            return (0,[9999])
        else:
            # print(f'nextnodeid:{nextnodeid},pos:{self.Vnodelist[nextnodeid[0]].position}')
            return (1,nextnodeid)


    # 选择转发方向上的中继节点，该节点比当前节点距离消费者更远
    def choosenextnode_maxdistance_FD_NC(self,consumerid,nodeid,Forwardingdirection):
        # print(f'entering choosenextnode_maxdistance_FD_NC')
        pos_cur=self.Vnodelist[nodeid].getposition()
        neighs = self.getneigh(nodeid)
        # print(f'neighs:{neighs}')
        # cplist=self.cpdict[self.packet.get_name()]

        flag_NCInter = re.match(r"NC", self.packet.get_name())
        # print(flag_NCInter,interest.get_name(),type(interest.get_name()))
        if flag_NCInter:
            cplist = list(self.cpdict[self.get_nameprefix(self.packet.get_name())])
            cplist1= list(self.cpdict1[self.get_nameprefix(self.packet.get_name())])
        else:
            if self.packet.get_name() in self.cpdict.keys():
                cplist = list(self.cpdict[self.packet.get_name()])
            else:
                cplist=[]
            if self.packet.get_name() in self.cpdict1.keys():
                cplist1 = list(self.cpdict1[self.packet.get_name()])
            else:
                cplist1=[]
        # print(f'self.packet.get_name():{self.packet.get_name()},self.cpdict:{self.cpdict},self.cpdict1:{self.cpdict1},cplist:{cplist},cplist1:{cplist1}')
        cplist.extend(cplist1)
        # print(f'cplist:{cplist}')
        # FDlist = [0, math.pi, math.pi / 2, -math.pi / 2]
        # cp=[i for i in set(cplist) if i in set(neighs)]
        # print(f'Forwardingdirection:{Forwardingdirection},cplist:{cplist}')

        cp=[]
        pos_consumer = self.Vnodelist[int(consumerid)].getposition()
        for cpi in set(cplist):
            pos_cpi = self.Vnodelist[cpi].getposition()
            # print(f'cpi:{cpi},pos:{pos_cpi}')
            if cpi in set(neighs):
                if Forwardingdirection==0 and pos_cpi[0]-pos_consumer[0]>0:
                    cp.append(cpi)
                elif Forwardingdirection==3.141592653589793 and pos_cpi[0]-pos_consumer[0]<0:
                    cp.append(cpi)
                elif Forwardingdirection == 1.5707963267948966 and pos_cpi[1] - pos_consumer[1] > 0:
                    cp.append(cpi)
                elif Forwardingdirection == -1.5707963267948966 and pos_cpi[1] - pos_consumer[1] < 0:
                    cp.append(cpi)
        # print(f'cp:{cp}')
        # print(f'Forwardingdirection:{Forwardingdirection},cplist:{cplist}')

        if len(cp)>0:
            nextnodeid=cp
            # print(f'cplist:{cplist},neighs:{neighs},cp:{cp}')
        else:
            neighFD=[]
            nextnodeid=[]
            maxdis=0
            nei_max=0
            for nei in neighs:
                nipos = self.Vnodelist[nei].getposition()
                # print(f'nei:{nei}')
                FDirection=(nipos[0]-pos_cur[0])*math.cos(Forwardingdirection)+(nipos[1]-pos_cur[1])*math.sin(Forwardingdirection)
                disnei=self.cal_dis_nodes(nodeid, nei)
                if int(FDirection)>0 :
                    neighFD.append(nei)
                    if disnei > maxdis:
                        maxdis=disnei
                        # nextnodeid.append(nei)
                        nei_max=nei
                # print(f'nei:{nei},nodeidpos:{pos_cur},neipos:{nipos},Forwardingdirection:{Forwardingdirection},FDirection:{FDirection}\tis in the FD:{int(FDirection)>0}')
            nextnodeid.append(nei_max)
            # print(f'nodeid:{nodeid},Forwardingdirection:{Forwardingdirection},neighFD:{neighFD},maxdis:{maxdis},nextnodeid:{nextnodeid}')
        if len(nextnodeid)==0:
            return (0,9999)
        else:
            return (1,nextnodeid)

    # 选择转发方向上的中继节点，该节点比当前节点距离消费者更远
    def choosenextnode_maxdistance_FD(self,consumerid,nodeid,Forwardingdirection):
        # print(f'entering choosenextnode_maxdistance_FD')
        pos_cur=self.Vnodelist[nodeid].getposition()
        neighs = self.getneigh(nodeid)
        # print(f'neighs:{neighs}')
        # cplist=self.cpdict[self.packet.get_name()]

        flag_NCInter = re.match(r"NC", self.packet.get_name())
        # print(flag_NCInter,interest.get_name(),type(interest.get_name()))
        if flag_NCInter:
            cplist = self.cpdict[self.get_nameprefix(self.packet.get_name())]
        else:
            cplist = self.cpdict[self.packet.get_name()]

        # print(f'cplist:{cplist}')
        cp=[i for i in set(cplist) if i in set(neighs)]

        if len(cp)>0:
            nextnodeid=cp
            # print(f'cplist:{cplist},neighs:{neighs},cp:{cp}')
        else:
            neighFD=[]
            nextnodeid=[]
            maxdis=0
            for nei in neighs:
                # print(f'nei:{nei}')
                nipos = self.Vnodelist[nei].getposition()
                FDirection=(nipos[0]-pos_cur[0])*math.cos(Forwardingdirection)+(nipos[1]-pos_cur[1])*math.sin(Forwardingdirection)
                disnei=self.cal_dis_nodes(nodeid, nei)
                if int(FDirection)>0 :
                    neighFD.append(nei)
                    if disnei > maxdis:
                        maxdis=disnei
                        nextnodeid.append(nei)
                # print(f'nei:{nei},nodeidpos:{pos_cur},neipos:{nipos},Forwardingdirection:{Forwardingdirection},FDirection:{FDirection}\tis in the FD:{int(FDirection)>0}')
            # print(f'nodeid:{nodeid},Forwardingdirection:{Forwardingdirection},neighFD:{neighFD},maxdis:{maxdis},nextnodeid:{nextnodeid}')
        if len(nextnodeid)==0:
            return (0,9999)
        else:
            return (1,nextnodeid)

    # 更新节点队列长度
    def Compute_Nodes_queue_length(self):
        self.node_queue_lengths=[]
        for nodei in self.Vnodelist:
            # print('nodei.getRNodeid()', nodei.getRNodeid())
            # print(f'nodei.getNodeid:{nodei.getNodeid()}')
            #两个字典合并,将接收队列合并到发送队列
            # print(f'before {nodei.getRNodeid()} nodei.sendqueuedict:{nodei.sendqueuedict.keys()},nodei.recievequeuedict:{nodei.recievequeuedict.keys()}')
            bk=list(nodei.recievequeuedict.keys())
            if len(bk) > 0:
                for bb in nodei.recievequeuedict.keys():
                    if bb in nodei.sendqueuedict.keys():
                        # print('bb', bb)
                        bb1 = bb + round(random.uniform(0.0001, 0.002), 5)
                        nodei.recievequeuedict[bb1] = nodei.recievequeuedict[bb]
                        del nodei.recievequeuedict[bb]
            nodei.sendqueuedict.update(nodei.recievequeuedict)
            nodei.recievequeuedict={}
            # print(f'after {nodei.nodeid} nodei.sendqueuedict:{nodei.sendqueuedict.items()}')
            queue_size = len(nodei.getsendqueue())
            self.node_queue_lengths.append(queue_size)  #各个节点的队列列表,包含未处理的所有包
        gol.set_value('node_queue_lengths', self.node_queue_lengths)
        # print('----node_queue_lengths',self.node_queue_lengths)
        # print('nodei.sendqueuedict',nodei.sendqueuedict)

    # 计算node1和node2之间的距离
    def cal_dis_nodes(self, node1, node2):
        pos1 = self.Vnodelist[int(node1)].getposition()
        pos2 = self.Vnodelist[int(node2)].getposition()
        dis12=math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return dis12

