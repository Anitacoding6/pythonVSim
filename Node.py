import numpy as np
import gol
import golData
import golISD
import golSIR
import golHCN
import goldeaddataCV
import golcps_innonum
import golnameprefix_cps

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from wjxlog import Logpath
import math
from RScode import RScode
import re
import json

class Channel():
    def __init__(self,i,j):
        #6 M bit/s
        self.transmissionrate = 6*10**6
        self.payload=1024*8
        self.transdelay=0.0014
        self.snode=i
        self.dnode=j

    def calc_transrate(self):
        transdelay=self.payload/self.transmissionrate
        return transdelay

    def calc_propdelay(self):
        dis = math.sqrt((self.Vnodelist[self.snode].position[0] - self.Vnodelist[self.dnode].position[0]) ** 2 + (
                    self.Vnodelist[self.snode].position[1] - self.Vnodelist[self.dnode].position[1]) ** 2)
        propdelay = dis / 299792458
        return propdelay

# vehicle or RSU node
class Node(object):
    def __init__(self,nodeid,pos=[0,0]):
        self.nodeid=nodeid
        # self.vehijiange=vehijiange
        self.position = pos
        self.roadlength=5000
        # self.roaddirection=getroaddirection()
        self.initposition()
        self.neighbors = self.setneighbor(nodeid)
        # self.CS = {}  # 格式：名字，秩，系数矩阵 #name:(rank,coff)
        self.CS = self.setCS(0)   #集合
        self.CS_NC={}  #字典格式：longest name prefix：/NC/movie/1 +编码系数矩阵: coef。name:/NC/movie/1/coef。
        self.CS_NC_inno={} #字典格式：longest name prefix：/NC/movie/1 +num_inno新颖个数。
        self.encodedData_con={} #字典格式：longest name prefix：/NC/movie/1 +编码系数矩阵: coef。name:/NC/movie/1/coef。

        self.RS = RScode()
        # self.num_chunks = gol.get_value('num_chunks') #编码块数
        self.DeadNonceset=set()  #集合  此节点上处理过的兴趣包随机数集合
        self.PIT= {}  ## 字典格式：#name:(lastnode,sendtime)，名字，上一跳节点，插入时间（在这里约等于发送时间，检查是否失效）
        self.PIT_NC = {}  #字典格式：nameprefix/nonce + (rank,lastnode,sendtime)
                            #暂时未考虑聚合
        self.sendqueue_Inerest=[]  #列表格式
        self.sendqueue_Data=[]
        # self.Rsendqueue=[]  # include self.sendqueue_Inerest and self.sendqueue_Data
        self.sendqueuedict={} # 按照 eventtime 排序，包括 self.sendqueue_Inerest and self.sendqueue_Data
        self.recievequeuedict={}  #不断会有从其他节点接收的数据包
        self.sendqueuedict_currentinterval={}
        self.sendqueuedict_nextinterval_wait={}
        self.queue={}  #格式：name: eventtime
        self.maxqueue=50  #最大队列长度
        self.FIP=0
        self.FDP=0
        self.filename = 'Logfilepath' + '.txt'

    def __str__(self):
        return self.nodeid

    def printsendque(self):
        for key, value in self.sendqueuedict.items():
            print(f'{key}: {value.get_name()}\t {value.get_packettype()}\t {value.get_path()}')
        return len(self.sendqueuedict)

    def addpacket(self,packet):
        #self.Rsendqueue.append(packet)
        # self.sendqueuedict[packet.get_eventtime()]=packet
        # print(f'node id ",self.getNodeid(),before add packet:{self.recievequeuedict},{self.recievequeuedict.keys()}')
        tt=packet.get_eventtime()
        # print(f'tt:{tt}')
        if tt in self.recievequeuedict.keys():
            tt=tt+0.001
            # print(f'after tt:{tt}')
            self.recievequeuedict[tt] = packet
        else:
            self.recievequeuedict[packet.get_eventtime()]=packet  #对应Env，节点都是nextnode，新接收的等待排队发送的包，以便判断是否有当前周期需要更新处理的包
        # print("node id ",self.getNodeid(), "**********add packet****** The packet name is ",packet.get_name(),"\tpacket type\t",packet.get_packettype(),"Event Time is ",packet.get_eventtime())
        # print(f'after add packet:{self.recievequeuedict}')

    def getsendqueue(self):
        return self.sendqueuedict

    def getNodeid(self):
        return self.nodeid

    def initposition(self):
        pass

    def setposition(self,position):
        self.position=position

    def getposition(self):
        return self.position

    def setneighbor(self,nodeid):
        if nodeid=='0':
            Rn=set()
            Rn.add('1')
            Rn.add('5')
            return Rn
        elif nodeid=='1':
            Rn=set()
            Rn.add('0')
            Rn.add('2')
            return Rn
        elif nodeid=='2':
            Rn=set()
            Rn.add('1')
            Rn.add('3')
            Rn.add('5')
            return Rn
        elif nodeid=='3':
            Rn=set()
            Rn.add('2')
            Rn.add('4')
            return Rn
        elif nodeid=='4':
            Rn=set()
            Rn.add('3')
            Rn.add('5')
            return Rn
        elif nodeid=='5':
            Rn=set()
            Rn.add('0')
            Rn.add('2')
            Rn.add('4')
            return Rn

    def getneighbors(self):
        return self.neighbors

    def add_DeadNonceSet(self,nonce):
        self.DeadNonceset.add(nonce)

    def If_Nonce_duplicate(self,nonce):
        if nonce in self.DeadNonceset:
            return 1
        else:
            return 0

    # 初始化设置CS内容
    def setCS(self,num):
        cs=set()
        i=0
        while i<num:
            prefix="movie"
            contnam=prefix+"/"+str(i)
            # print('contnam',contnam)
            cs.add(contnam)
            i=i+1

        # print(cs)
        # print(cs)
        return cs

    # CS添加内容
    def addCS(self,name):
        # print('entering RNode addCS')
        #如果 selt.CS 没有该name，则添加，
        if name not in self.CS:
            # print('CS不存在name', name)
            # print("before")
            # print(self.CS)
            self.CS.add(name)
            # print("after")
            # print(self.CS)

    def addCS_NC_random(self,name):

        # 找到名字的最长前缀作为key，系数矩阵作为value
        # NC/movie/1；coef
        name_prefix=self.get_nameprefix(name)
        name_coef=name[11:]
        print(name_prefix)
        print(name_coef)
        # 如果没有找到相同key,插入新条目
        if name_prefix not in self.CS_NC.keys():
            self.CS_NC[name_prefix] = np.array(name_coef)
        else:
            # 如果存在相同key，且线性无关，则coefs+coef组成新的矩阵作为coefs
            coefs=self.CS_NC[name_prefix]
            coefs_=np.row_stack(coefs,coef)
            print(f'coefs:{self.RS.RS_rank(coefs)},coefs_:{self.RS.RS_rank(coefs_)}')

    def string2vec(self,coef_s):
        # coef_s:[1 0 5 1 7],numlist:[1, 0, 5, 1, 7]
        numlist = []
        coef_sa = coef_s.split("[")
        # print(f'coef_sa:{coef_sa}')
        coef_sa1 = coef_sa[1].split("]")
        # print(f'coef_sa1:{coef_sa1}')
        coef_sa2 = coef_sa1[0].split(" ")
        # print(f'coef_sa2:{coef_sa2}')
        for s in coef_sa2:
            if s.isdigit():
                numlist.append(int(s))
        # print(f'numlist:{numlist}')
        return numlist

    def string2mat(self, coef_s, num_chunks):
        # print(f'---num_chunks:{num_chunks}')
        # coef_s:[1 0 5 1 7],numlist:[1, 0, 5, 1, 7]
        # print(f'coef_s:{coef_s}')
        numlist = []
        coef_sa = coef_s.split("[")
        # print(f'coef_sa:{coef_sa}')
        for jj in coef_sa:
            # print(f'jj:{jj}')
            if jj:
                coef_sb = jj.split("]")
                # print(f'coef_sb:{coef_sb}')
                coef_sc = coef_sb[0].split(" ")
                for ss in coef_sc:
                    if ss.isdigit():
                        numlist.append(int(ss))
        # print(f'numlist:{numlist}')
        mat = np.array(numlist).reshape(-1, num_chunks)
        # print(f'mat:{mat}')
        # num_chunks
        return mat

    #初始化设置，名字前缀为nameprefix，随机生成num个 每一代块数是num_chunks 的编码数据包
    def initCS_num_random_NDN1(self,contnam_NDN_prefix,num,num_chunks=5):
        contnamlist=[]
        if num!=0:
            # 随机生成num个后缀,[0, 3, 1, 2]
            # a=random.sample(range(num_chunks),num)
            a=list(range(num_chunks,num_chunks-num,-1))
            for i in a:
                conname=contnam_NDN_prefix+'/'+str(i)
                self.CS.add(conname)
                contnamlist.append(conname)
        return contnamlist


    #初始化设置，名字前缀为nameprefix，随机生成num个 每一代块数是num_chunks 的编码数据包
    def initCS_num_random_NDN(self,contnam_NDN_prefix,num,num_chunks=5):
        contnamlist=[]
        if num!=0:
            # 随机生成num个后缀,[0, 3, 1, 2]
            # a=random.sample(range(num_chunks),num)
            a=list(range(num))
            for i in a:
                conname=contnam_NDN_prefix+'/'+str(i)
                self.CS.add(conname)
                contnamlist.append(conname)
        return contnamlist

     #初始化设置，名字前缀为nameprefix，随机生成num个 每一代块数是num_chunks 的编码数据包
    def initCS_num_random_NC(self,nameprefix,num,num_chunks=5):
        if num!=0:
            mat=[]
            for i in range(num):
                coef=self.RS.coefficients_generate(num_chunks)
                # print(f'{coef}---np.sum(coef):{np.sum(coef)}')
                if np.sum(coef)!=0:
                    mat.append(coef)
                else:
                    # print('********')
                    coef = self.RS.coefficients_generate(num_chunks)
                    if np.sum(coef) != 0:
                        mat.append(coef)

            # self.CS_NC[name_prefix] = np.array(name_coef_vec)
            self.CS_NC[nameprefix] = mat
            self.CS_NC_inno[nameprefix]=num


    def addCS_NC(self,name):
        # print(f'entering addCS_NC, adding name{name},当前CS_NC为：{self.CS_NC}')
        # 找到名字的最长前缀作为key，系数矩阵作为value
        # NC/movie/1；coef
        # name: NC / movie / 10 / 5, nameprefix: NC / movie / 10, namesuffix: 5
        name_prefix=self.get_nameprefix(self.get_nameprefix(name))
        name_coef=self.get_namesuffix(name)
        # print(f'name_prefix:{name_prefix}')
        # print(f'name_coef:{name_coef}')
        # 如果没有找到相同key,插入新条目
        if name_prefix not in self.CS_NC.keys():
            # print(f'CS_NC没有关于该内容的缓存')
            if bool(name_coef):
                name_coef_vec=self.string2vec(name_coef)
                # print(f'no find key, name_coef:{name_coef},name_coef_vec:{name_coef_vec}')
                mat=[]
                mat.append(name_coef_vec)
                # self.CS_NC[name_prefix] = np.array(name_coef_vec)
                self.CS_NC[name_prefix] = mat
                self.CS_NC_inno[name_prefix]=1
                # print(f'999rank_cs:{self.RS.RS_rank(mat)}')
        else:
            # print(f'find key,当前CS_NC：{self.CS_NC}')
            # 如果存在相同key，且线性无关，则coefs+coef组成新的矩阵作为coefs
            coefs_cs=self.CS_NC[name_prefix]
            # name_coef_vec=self.string2vec(name_coef)
            # print(f'-----acoefs_cs:{coefs_cs},{type(coefs_cs)},rank_cs:{self.RS.RS_rank(coefs_cs)}')
            # coefs_=np.row_stack((np.array(coefs_cs), name_coef_vec))
            num_chunks = gol.get_value('num_chunks')
            name_coef_mat=self.string2mat(name_coef,num_chunks)
            coefs_=np.row_stack((np.array(coefs_cs), name_coef_mat))

            # print(f'666coefs_:{coefs_}')
            # print(f'444rank coefs_cs:{self.RS.RS_rank(coefs_cs)}')
            # print(f'555rank coefs_:{self.RS.RS_rank(np.array(coefs_))}')

            if self.RS.RS_rank(coefs_)>self.RS.RS_rank(coefs_cs):
                # 新颖数据包，插入CS_NC
                # print('新颖数据包，插入CS_NC')
                self.CS_NC[name_prefix] = np.array(coefs_)
                self.CS_NC_inno[name_prefix]=self.CS_NC_inno[name_prefix]+1
            else:
                pass
                # print('不是新颖包')
        # print(f'after adding, CS_NC:{self.CS_NC}')


    # 返回CS内容
    def getCS(self):
        return self.CS

    # CS是否命中
    def CShit(self,name):
        if name in self.CS:
            # print("cs hit")
            return 1
        else:
            # print("cs miss")
            return 0


    # 采用网络编码后，CS缓存命中,返回编码系数，以及提供新颖包的能力
    def CShit_NC1(self,name):
        flag_hit=0
        num_innov_=0
        coef_data=0
        name_prefix=name[0:10]
        name_coef=name[11:]
        # print(f'---check--whether-CShit_NC, name_prefix:{name_prefix},name:{name_coef}')

        # 缓存命中，随机生成一个编码数据包返回
        if name_prefix in self.CS_NC.keys():
            # if 系数矩阵coefs 对于 name_coef是新颖的innovative，即增加秩，增加秩的个数为rank_innov
            coefs_cs=self.CS_NC[name_prefix]
            # print(f'------coefs_cs:{coefs_cs},{type(coefs_cs)},name_coef:{name_coef},{type(name_coef)}')

            # 事实上，Interest并没有携带系数矩阵。
            """
            # 兴趣包名字系数
            if bool(name_coef):
                print(f'--------****-name_coef:{name_coef}-------------')
                num_chunks = gol.get_value('num_chunks')
                name_coef_mat=self.string2mat(name_coef,num_chunks)
                # print(f'---name_coef:{name_coef}---name_coef_mat:{name_coef_mat}')
                coefss_ = np.array(np.row_stack((np.array(coefs_cs), name_coef_mat)), dtype=np.int64)

                # print(f'1111coefss_:{coefss_},{type(coefss_)}')
                # print(f'name_coef_mat:{name_coef_mat},消费者已有的内容个数：{self.RS.RS_rank(name_coef_mat)}')
                # print(f'总秩：{self.RS.RS_rank(coefss_)}')
                num_innov=self.RS.RS_rank(coefss_)-self.RS.RS_rank(name_coef_mat)
                # print(f'num_innov:{num_innov}')
            elif np.any(coefs_cs):
                # print(f'-------coefs_cs不为空,name_coef为空----:{name_coef},self.CS_NC:{self.CS_NC}')
                num_innov = self.RS.RS_rank(np.array(coefs_cs))
            else:
                # print(f'--else-----name_coef为空:{name_coef},self.CS_NC:{self.CS_NC}')
                num_innov=0
            """

            num_innov=self.CS_NC_inno[name_prefix]

            if num_innov==1 and self.RS.RS_rank(np.array(coefs_cs))==1:
                flag_hit=1
                # 计算后续可能提供有用的个数rank_innov
                num_innov_ = num_innov - 1
                del self.CS_NC[name_prefix]
                self.CS_NC_inno[name_prefix]=0
                #调用函数，对CS中的缓存编码数据包中的系数矩阵随机编码
                coef_data=coefs_cs[0]
                # print(f'--1---coef_data:{coef_data},{type(coef_data)}')

            elif num_innov>0:
                flag_hit=1
                # 计算后续可能提供有用的个数rank_innov
                num_innov_ = num_innov - 1
                self.CS_NC_inno[name_prefix]=num_innov_
                #调用函数，对CS中的缓存编码数据包中的系数矩阵随机编码
                # print(f'-----coefs_cs:{coefs_cs},{type(coefs_cs)}')
                # coefs_cs: [[1 2 7 2 0]
                #            [0 1 4 0 7]
                #             [0 0 1 5 7]]
                num_chunks=gol.get_value('num_chunks')
                coef_data=self.RS.RS_reencode_mat(coefs_cs,num_chunks)
                # print(f'--2---coef_data:{coef_data},{type(coef_data)}')

        return (flag_hit,num_innov_,coef_data)

    # name: NC / movie / 10 / 5, nameprefix: NC / movie / 10, namesuffix: 5
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

    # name: NC / movie / 10 / 5, nameprefix: NC / movie / 10, namesuffix: 5
    def get_namesuffix(self,name):
        na = name.split("/")
        return na[-1]

    # 采用网络编码后，CS缓存命中,返回编码系数，以及提供新颖包的能力
    def CShit_NC(self,interest):
        name=interest.get_name()
        router=interest.router
        flag_hit=0
        num_innov_=0
        coef_data=0
        name_prefix=self.get_nameprefix(name)
        # name_chunk=name[11:]
        # print(f'---check--whether-CShit_NC, name_prefix:{name_prefix},self.CS_NC:{self.CS_NC}')

        if router=="multicp_probe":
            pass
        # 路由器等于multicp_NC，利用CM携带到Interest上
        elif router=="multicp_NC":
            # print(f'router=="multicp_NC"self.CS_NC:{self.CS_NC}，name_prefix:{name_prefix}')
            # 名字前缀的前缀
            if name_prefix in self.CS_NC.keys():
                # print(f'CS_NC hit and router=="multicp_NC"self.CS_NC:{self.CS_NC}，name_prefix:{name_prefix}')
                coefs_cs = self.CS_NC[name_prefix]
                # print(f'**********coefs_cs:{coefs_cs}********')
                if np.any(coefs_cs):
                    rank_cs=self.RS.RS_rank(np.array(coefs_cs))
                    # print(f'**********coefs_cs:{coefs_cs},rank_cs:{rank_cs}')
                    Data_consumer=golData.get_value(name_prefix)
                    # print(f'**********Data_consumer:{Data_consumer}')
                    rank_Interest=self.RS.RS_rank(np.array(Data_consumer))
                    if rank_Interest==0:
                        rank_Interest_cs=rank_cs
                    else:
                        coefs_ = np.row_stack((np.array(coefs_cs), np.array(Data_consumer)))
                        rank_Interest_cs=self.RS.RS_rank(coefs_)

                    # print(f'coefs_cs:{coefs_cs},rank_cs:{rank_cs},Data_consumer:{Data_consumer},rank_Interest:{rank_Interest},rank_Interest_cs:{rank_Interest_cs}')

                    # 可以提供新颖包
                    if rank_Interest_cs>rank_Interest:
                        flag_hit = 1
                        if rank_cs==1:
                            coef_data = coefs_cs[0]
                        else:
                            num_chunks = gol.get_value('num_chunks')
                            coef_data = self.RS.RS_reencode_mat(coefs_cs, num_chunks)
                    # print(f'coef_data:{coef_data}')

        # 缓存命中，随机生成一个编码数据包返回
        # 并将新颖度减1，多源存在缓存资源浪费的问题
        elif name_prefix in self.CS_NC.keys() and self.CS_NC_inno[name_prefix]>0:
            # if 系数矩阵coefs 对于 name_coef是新颖的innovative，即增加秩，增加秩的个数为rank_innov
            coefs_cs=self.CS_NC[name_prefix]
            # print(f'------coefs_cs:{coefs_cs},{type(coefs_cs)},name_coef:{name_coef},{type(name_coef)}')

            # 事实上，Interest并没有携带系数矩阵。
            num_innov=self.CS_NC_inno[name_prefix]
            if num_innov==1 and self.RS.RS_rank(np.array(coefs_cs))==1:
                flag_hit=1
                # 计算后续可能提供有用的个数rank_innov
                num_innov_ = num_innov - 1
                del self.CS_NC[name_prefix]
                self.CS_NC_inno[name_prefix]=0
                #调用函数，对CS中的缓存编码数据包中的系数矩阵随机编码
                coef_data=coefs_cs[0]
                # print(f'--1---coef_data:{coef_data},{type(coef_data)}')

            elif num_innov>0:
                flag_hit=1
                # 计算后续可能提供有用的个数rank_innov
                num_innov_ = num_innov - 1
                self.CS_NC_inno[name_prefix]=num_innov_
                #调用函数，对CS中的缓存编码数据包中的系数矩阵随机编码
                # print(f'-----coefs_cs:{coefs_cs},{type(coefs_cs)}')
                # coefs_cs: [[1 2 7 2 0]
                #            [0 1 4 0 7]
                #             [0 0 1 5 7]]
                num_chunks=gol.get_value('num_chunks')
                coef_data=self.RS.RS_reencode_mat(coefs_cs,num_chunks)
                # print(f'--2---coef_data:{coef_data},{type(coef_data)}')

        return (flag_hit,num_innov_,coef_data)


    def longestCommonPrefix1(self, strs):
        lcp = ""
        for tmp in zip(*strs):
            if len(set(tmp)) == 1:
                lcp += tmp[0]
            else:
                break
        return lcp

    def PIThit(self,interest):
        nam=interest.get_name()
        if nam in self.PIT.keys():
            pass
            # print("PIT hit")
        else:
            pass
            # print("PIT miss")

    # PIT里没有该内容，则添加
    def addPIT_NC1(self,interest):
        nam=interest.get_name()
        # print(f'---before--- adding Interest:{nam}, print PIT_NC:{self.PIT_NC}')
        name_prefix = nam[0:10]
        name_coef= nam[11:]
        nonce=interest.get_nonce()
        # name_prefix+nonce
        pit_key=name_prefix+'/'+str(nonce)

        # 兴趣包名字系数
        if bool(name_coef):
            num_chunks = gol.get_value('num_chunks')
            name_coef_mat = self.string2mat(name_coef, num_chunks)
            # print(f'---name_coef:{name_coef}---name_coef_mat:{name_coef_mat}')
            rank_coef_Int=self.RS.RS_rank(name_coef_mat)
        else:
            rank_coef_Int=0

        lastnodeid=interest.get_currentnode().getNodeid()  #lastnode 是兴趣包当前节点，兴趣包选择节点是当前节点id
        # print('lastnodeid',lastnodeid)
        sendtime=interest.get_sendtime()
        cosumerid=interest.get_consumer()

        """
        if name_prefix not in self.PIT_NC.keys():
            self.PIT_NC[name_prefix]=(lastnodeid,sendtime,rank_coef_Int,cosumerid)
        else:
            print(f'-----in PIR_NC, PIT聚合')
            #如果存在，PIT聚合怎么聚合？
            # 如果发送时间或者消费者不同的话，聚合
            pass
        """
        if pit_key not in self.PIT_NC.keys():
            self.PIT_NC[pit_key]=(lastnodeid,sendtime,rank_coef_Int,cosumerid)
        else:
            print(f'-----in PIR_NC, PIT聚合')
            #如果存在，PIT聚合怎么聚合？
            # 如果发送时间或者消费者不同的话，聚合
            pass
        # print(f'---after--- adding Interest:{nam}, print PIT_NC:{self.PIT_NC}')

    # PIT里没有该内容，则添加
    def addPIT_NC(self,interest):
        nam=interest.get_name()
        # print(f'---before--- adding Interest:{nam}, print PIT_NC:{self.PIT_NC}')
        # name_prefix = nam[0:10]
        # name_coef= nam[11:]
        # nonce=interest.get_nonce()
        # name_prefix+nonce
        # pit_key=name_prefix+'/'+str(nonce)

        # 兴趣包名字系数
        # if bool(name_coef):
        #     num_chunks = gol.get_value('num_chunks')
        #     name_coef_mat = self.string2mat(name_coef, num_chunks)
        #     # print(f'---name_coef:{name_coef}---name_coef_mat:{name_coef_mat}')
        #     rank_coef_Int=self.RS.RS_rank(name_coef_mat)
        # else:
        #     rank_coef_Int=0

        lastnodeid=interest.get_currentnode().getNodeid()  #lastnode 是兴趣包当前节点，兴趣包选择节点是当前节点id
        # print('lastnodeid',lastnodeid,type(lastnodeid))
        sendtime=interest.get_sendtime()
        cosumerid=interest.get_consumer()

        """
        if name_prefix not in self.PIT_NC.keys():
            self.PIT_NC[name_prefix]=(lastnodeid,sendtime,rank_coef_Int,cosumerid)
        else:
            print(f'-----in PIR_NC, PIT聚合')
            #如果存在，PIT聚合怎么聚合？
            # 如果发送时间或者消费者不同的话，聚合
            pass
        """
        rank_coef_Int=0
        if nam not in self.PIT_NC.keys():
            self.PIT_NC[nam]=(lastnodeid,sendtime,rank_coef_Int,cosumerid)
        else:
            print(f'-----in PIR_NC, PIT聚合')
            #如果存在，PIT聚合怎么聚合？
            # 如果发送时间或者消费者不同的话，聚合
            pass
        # print(f'---after--- adding Interest:{nam}, lastnodeid:{lastnodeid}, print PIT_NC:{self.PIT_NC}')

    # PIT里没有该内容，则添加
    def addPIT(self,interest):
        nam=interest.get_name()
        lastnodeid=interest.get_currentnode().getNodeid()  #lastnode 是兴趣包当前节点，兴趣包选择节点是当前节点id
        # print('lastnodeid',lastnodeid)
        sendtime=interest.get_sendtime()
        # print(f'{interest.get_consumer()}')

        if nam not in self.PIT.keys():
            self.PIT[nam]=(lastnodeid,sendtime)
        else:
            #如果存在，PIT聚合怎么聚合？
            pass
        # print("print PIT",self.PIT)

    def deletPIT(self,name):
        if name in self.PIT.keys():
            # print(self.PIT)
            del self.PIT[name]
            # print(self.PIT)
        else:
            pass
            # print("error! deletPIT name:\t", name, "but it is not in PIT")

    def deletPIT_NC_int(self,interest):
        # self.PIT_NC[name_prefix] = (lastnodeid, sendtime, rank_coef_Int, cosumerid)
        name=interest.get_name()
        # name_prefix=name[0:10]
        # nonce=interest.get_nonce()
        # pit_key = name_prefix + '/' + str(nonce)
        if name in self.PIT_NC.keys():
            # print(f'before delete self.PIT_NC:{self.PIT_NC}')
            del self.PIT_NC[name]
            # print(f'after delete self.PIT_NC:{self.PIT_NC}')
        else:
            pass
            # print(f'error! deletPIT name: {name}, but it is not in PIT')

    def get_lastnodeidfromPIT_NC(self,interest):
        nameofin=interest.get_name()
        # print(f'nameofin:{nameofin},PIT_NC:{self.PIT_NC}')
        # name_prefix = nameofin[0:10]
        # nonce = interest.get_nonce()
        # name_prefix+nonce
        # pit_key = name_prefix + '/' + str(nonce)
        lastnodeid=self.PIT_NC[nameofin][0]
        return lastnodeid

    def get_lastnodeidfromPIT(self,interest):
        nameofin=interest.get_name()
        # print(f'nameofin:{nameofin},PIT:{self.PIT}')
        lastnodeid=self.PIT[nameofin][0]
        return lastnodeid


    # 节点收到兴趣包Interest
    def onIncomingInterest(self,interest,nextnode='999'):
        # print(f'----entering onIncomingInterest,node id:{self.getNodeid()},name:{interest.get_name()}, nonce:{str(interest.get_nonce())},consumer:{ interest.get_consumer()},')
        # print(f'sendtime:{str(interest.get_sendtime())}, lastnode:{str(interest.get_lastnode())}')
        # print(f'path:{interest.get_path()},{self.position}')
        lastnodeid_ofInt = interest.get_currentnode().getNodeid()  # 实际上是上一个节点  #兴趣包上的节点 last,current,next,在后面选择nextnode时统一更新
        selectnodeid_ofInt = interest.get_nextnode().getNodeid()  # 实际上是上一个节点选择的中继节点

        # 首先判断 当前节点是否是 上一个节点选择的中继节点 或者 消费者（兴趣包来自“app”）
        curnodeid = self.getNodeid()  # RSU current node id
        interest.path=interest.path+'->'+str(curnodeid)
        # print(f'path:{interest.path}')

        # print("curnodeid", curnodeid, "lastnodeidofInt", lastnodeid_ofInt, "selectnodeidofInt", selectnodeid_ofInt,"path",interest.get_path())
        # print("curnodeid",type(curnodeid),"lastnodeidofInt",type(lastnodeid_ofInt),"selectnodeidofInt",type(selectnodeid_ofInt))
        # print('curnodeid==selectnodeidofInt',curnodeid==selectnodeid_ofInt)
        # print("lastnodeidofInt==app",lastnodeid_ofInt=="app")

        # 冗余包， Nonce name相同  已经存储在 Nonceset 里了
        # 测试随机数Nonce是否相同
        if interest.get_nonce() in self.DeadNonceset:
            # print("name\t",interest.get_name(),"\t nonce duplicate","\t return and drop this Interest packet")
            return
        else:
            self.add_DeadNonceSet(interest.get_nonce())
            # print("name\t",interest.get_name(),"\t nonce not duplicate")

        # CS hit
        # 判断兴趣包是否是请求NC的Interest
        # movie/NCNDN/coef
        # movie/NDN/1
        # print('-----------11111111111---------------')
        flag_NCInter = re.match(r"NC", interest.get_name())
        # print(flag_NCInter,interest.get_name(),type(interest.get_name()))
        if flag_NCInter:
            # 请求编码数据包
            # print("request NCNDN Interest packet")
            flag_hit, num_innov_, coef_data=self.CShit_NC(interest)
            # print(f'-----curnodeid:-----CS_hit:{flag_hit}, num_innov_:{num_innov_}, coef_data:{coef_data}')
        # NC，CS缓存命中
        # 如果网络编码，并且缓存命中
        if flag_NCInter and flag_hit:
            # print(f'---curnodeid:{curnodeid}--CS_NC hit:{flag_hit}, ----num_innov_:{num_innov_}, coef_data:{coef_data}')
            self.addPIT_NC(interest)
            nextnodeidofData=self.get_lastnodeidfromPIT_NC(interest)
            # 将数据包发送到上一跳
            # 编码数据包格式变化:新增字段num_innov_；名字变化：name_coef改为coef_data
            self.replyData_NC(interest,nextnodeidofData,num_innov_,coef_data)
            # print('nextnodeidofData',nextnodeidofData)
            # 删除PIT请求
            self.deletPIT_NC_int(interest)

        # 请求没有编码的NDN数据包，并且缓存命中
        elif interest.get_name() in self.CS:
            # print(f'curnodeid:{curnodeid}',"CS hit")
            with open(self.filename, 'a') as f:
                f.write(f'name:{interest.get_name()}, node id:{self.getNodeid()}, CS hit replyData, eventtime:{interest.get_eventtime()}, path:{interest.path},nextnode:{nextnode}\n')
            #return data packet, 转发数据包到上一跳
            # 聚合 or 不聚合，如果不聚合，从PIT获得的 lastnodeid 和 从兴趣包获得的是一样的；考虑聚合的话，需要先插入再查看。
            self.addPIT(interest)

            # 查看PIT，上一跳节点id
            #考虑PIT聚合是个结合，不考虑 nextnodeidofData=lastnodeid_ofInt
            nextnodeidofData=self.get_lastnodeidfromPIT(interest)
            # print('nextnodeidofData',nextnodeidofData,'lastnodeid_ofInt',lastnodeid_ofInt,'nextnodeidofData=lastnodeid_ofInt',nextnodeidofData==lastnodeid_ofInt)

            # 将数据包发送到上一跳
            self.replyData(interest,nextnodeidofData)
            # print('nextnodeidofData',nextnodeidofData)
            # 删除PIT请求
            self.deletPIT(interest.get_name())

        # CS miss
        else:
            # print("CS miss")
            if curnodeid == selectnodeid_ofInt or lastnodeid_ofInt == "app":
                # print(f'curnodeid == selectnodeid_ofInt or lastnodeid_ofInt == app')
                # print(f'curnodeid:{curnodeid},lastnodeid_ofInt:{lastnodeid_ofInt}')
                # curnodeid:60,lastnodeid_ofInt:app
                # curnodeid:4,lastnodeid_ofInt:0

                # PIT insert
                # PIT 聚合
                if flag_NCInter:
                    self.addPIT_NC(interest)
                else:
                    # PIT 插入新的兴趣包
                    self.addPIT(interest)
                #更新兴趣包节点id，lastnode,currentnode
                interest.updatenodeid_interest()
                # 调用DRL选择下一跳,添加到兴趣包上
                # nextnode="999"
                interest.set_nextnode(nextnode)
                node_queue_lengths=gol.get_value('node_queue_lengths')
                # print(f'Node:{node_queue_lengths},neighbors:{self.getneighbors()}')
                neigh = self.getneighbors()
                # print(f'neigh:{neigh}')
                neigh_num = len(neigh)
                neighqueue_len = 0
                for nn in neigh:
                    neighqueue_len = neighqueue_len + node_queue_lengths[int(nn)]
                # print(f'nodeid:{nodeid},neigh:{neigh},neigh_len:{neigh_len},len:{len(neigh)}')
                # print(f'curnodeid:{curnodeid},neighqueue_len:{neighqueue_len},neigh_num:{neigh_num}')
                # 设置丢包率
                alpha = 0.0015
                # reliablepacketrate = 1 - alpha * neighqueue_len
                reliablepacketrate=1
                # print(f'Node onIncomingInterest, neighqueue_len:{neighqueue_len},reliablepacketrate,{reliablepacketrate}')

                # reliablepacketrate = 0.7
                if random.uniform(0, 1) < reliablepacketrate:

                    #计算并判断队列长度是否已超过节点最大队列长度self.maxqueue
                    que_cur=len(self.recievequeuedict)+len(self.sendqueuedict)
                    # print(f'recievequeuedict:{self.recievequeuedict},sendqueuedict:{self.sendqueuedict}')
                    # print(f'maxqueue:{self.maxqueue},que_cur:{que_cur}')

                    RSUID=gol.get_value('RSUID')

                    # 小于最大队列长度,转发；超过最大队列长度，丢包
                    if curnodeid==RSUID or que_cur<self.maxqueue:
                        # print(f'插入队列。RSUID:{RSUID},节点{curnodeid}的当前队列长度：{que_cur}，最大队列长度：{self.maxqueue}')
                        #转发兴趣包到该接口
                        self.onOutgoingInterest(interest,nextnode)
                    else:
                        pass
                        # print(f'队列已满，丢包。RSUID:{RSUID},节点{curnodeid}的当前队列长度：{que_cur}，最大队列长度：{self.maxqueue}')
            else:
                print("I am not the selected node")




    def onOutgoingInterest(self,interest,nextnode):
        # print(f'node id:{self.getNodeid()}, onOutgoingInterest, eventtime:{interest.get_eventtime()},name:{interest.get_name()}, path:{interest.path},nextnode:{nextnode}')
        # 更新eventtimes
        interest.quedelay_cur=(len(self.getsendqueue())+1) * 0.005
        evtime = interest.get_eventtime() +interest.quedelay_cur
        interest.set_eventtime(evtime)

        # update hopcount
        interest.update_hopcount()

        # self.filename = 'Logfile' + '.txt'
        # with open(self.filename, 'a') as f:
        #     f.write(f'node id:{self.getNodeid()}, onOutgoingInterest, eventtime:{interest.get_eventtime()},name:{interest.get_name()}, path:{interest.path},nextnode:{nextnode}\n')

        # 发送兴趣包到链路层，转信道排队
        # Env 管理带宽资源，包的排队等
        # RNode(nextnode).addpacket(interest.get_name())
        # print('RNode(nextnode).Rsendqueue',RNode(nextnode).Rsendqueue)

        self.addpacket(interest)
        # print('self.sendqueuedict', self.sendqueuedict)

        # 如果发送成功，转发兴趣包的个数加1，FIP=FIP+1
        #......FIP........#
        # print(f'name:{interest.get_name()},type:{type(interest.get_name())}')
        BGs="BG"
        intername=interest.get_name()
        if(intername.find(BGs)!=-1):
            pass
            # print(f'{intername} find {BGs}')
        else:
            # print(f'{intername} not find {BGs}')
            FIP = gol.get_value('FIP')
            FIP=FIP+1
            gol.set_value('FIP', FIP)
        # print(f'FIP:{FIP}')

    def replyData_NC(self, interest, lastnode, num_innov_, coef_data):
        # 编码数据包格式变化:新增字段num_innov_；名字变化：name_coef改为coef_data
        # print(f'-------replyData_NC----start-------------')
        name=interest.get_name()
        # name_prefix=name[0:10]
        # name_coef=name[11:]
        # print(f'-----name_prefix:{name_prefix},coef_data:{coef_data},{type(coef_data)}')
        nam=name+'/'+np.array2string(coef_data)
        # print(f'in replyData_NC, interest name:{name}, data name:{nam},coef_data:{coef_data},num_innov_:{num_innov_}')
        cp1=interest.get_nextnode()
        hopcount_in=interest.get_hopcount()
        # print(f'-cp:{cp1.__str__()}--name:{name}-hopcount_in:{hopcount_in}--replyData_NC：{interest.path}')
        namesuffix = self.get_namesuffix(name)
        name_prefix=self.get_nameprefix(name)
        nam_cp = name + "+" + str(cp1.getNodeid())
        namprefix_cp = name_prefix + "+" + str(cp1.getNodeid())

        # print(f'before deaddataCV:{goldeaddataCV.printdeaddataCV()}-nodeid:{nodeid}--name:{data.get_name()}--path:{data.path}->{nextnodeidofdata}')
        # if not goldeaddataCV.find(suffixCV_cp) and data.get_consumer()==nextnodeidofdata and int(namesuffix)<=3:
        # if int(namesuffix) <= 3:
        goldeaddataCV.add(nam_cp)

        namesuffix = self.get_namesuffix(name)
        contnampref = self.get_nameprefix(name)
        key1 = name + "+" + str(cp1.getNodeid())
        # -------start--update--innonum----------------
        # 可能接收到来自同一个数据源的不同的编码数据包

        if int(namesuffix) <= 3:
            # 被consumer真正接收到，更新num_inno
            # golnameprefix_cps.print_golnameprefix_cps()
            # print(f'key:{name},value:{cp1.getNodeid()}')
            # golcps_innonum.append(key1)
            golnameprefix_cps.add_ele(name,cp1.getNodeid())
            # golnameprefix_cps.print_golnameprefix_cps()


        # print(f'golcps_innonum:{golcps_innonum}')

        # print(f'-start--name:{data.get_name()}--path:{data.path}---suffixCV:{suffixCV}-node id:{nodeid}-dataname_prefix:{dataname_prefix},namesuffix:{namesuffix},recieve data, update golISD, golSIR')
        # print(f'----replyData_NC--deaddataCV:{goldeaddataCV.printdeaddataCV()}')
        # ----------golSIR-----------------------
        delay = interest.get_eventtime() + 0.005 - interest.get_sendtime()
        golISD.append_value(namprefix_cp, delay)
        golSIR.add_recieve(namprefix_cp)
        hopco = interest.get_hopcount()
        golHCN.set_value(namprefix_cp, hopco)
        # print(f'update golISD, golSIR, golHCN')
        # golSIR.printgolSIR()
        # golISD.printgolISD()
        # golHCN.printgolHCN()
        # print(f'--replyData_NC----end------')
        # ----------end-----------------------------
        # print(f'replyData:{self.getNodeid()},cp:{cp1.__str__()}')
        sendtime=interest.get_sendtime()
        # replytime=currenttime
        # replytime=10   #currenttime
        #eventtime
        delaydata=0.005
        replytime=interest.get_eventtime()
        eventtime=replytime+delaydata
        currenttime = eventtime
        # print(f'cp:{cp1.__str__()} replyData eventtime:{eventtime},sendtime:{sendtime}')
        # nextnode=interest.get_lastnode()  #查询PIT或者interest上记录的
        payload=1024
        data1=Data(eventtime,nam, cp1, sendtime, replytime, currenttime, lastnode, payload, num_innov_)
        data1.set_consumer(interest.get_consumer())
        data1.path= data1.path+str(cp1.getNodeid())
        data1.set_hopcount(0)
        # 在节点lastnode排队
        self.onOutgoingData(data1,lastnode)


    def replyData(self,interest,lastnode):
        # print("entering replyData")
        nam=interest.get_name()
        cp1=interest.get_nextnode().getNodeid()
        # print(f'replyData:{self.getNodeid()},cp:{cp1.__str__()}')
        sendtime=interest.get_sendtime()
        # replytime=currenttime
        # replytime=10   #currenttime
        #eventtime
        delaydata=0.005
        replytime=interest.get_eventtime()
        eventtime=replytime+delaydata
        currenttime = eventtime
        # print(f'cp:{cp1.__str__()} replyData eventtime:{eventtime},sendtime:{sendtime}')

        # nextnode=interest.get_lastnode()  #查询PIT或者interest上记录的
        payload=1024
        data1=Data(eventtime,nam, cp1, sendtime, replytime, currenttime, lastnode, payload)

        data1.path= data1.path+str(cp1)
        data1.set_hopcount(0)
        # 在节点lastnode排队
        self.onOutgoingData(data1,lastnode)

    def aftersendData(self,name):
        #发送到链路层，转信道排队
        pass

    def updategolData(self,data):
        name=data.get_name()
        name_prefix=self.get_nameprefix(self.get_nameprefix(name))
        name_coef=self.get_namesuffix(name)
        golDataname=golData.get_value(name_prefix)
        print(f'updategolData,name:{name},name_prefix:{name_prefix},name_coef:{name_coef},golDataname:{golDataname}')

    def addencodedData_con(self,data):
        name=data.get_name()
        # print(f'entering addencodedData_con, adding name{name},当前encodedData_con为：{self.encodedData_con}')
        # 找到名字的最长前缀作为key，系数矩阵作为value
        # NC/movie/1；coef
        # name: NC / movie / 10 / 5, nameprefix: NC / movie / 10, namesuffix: 5
        name_prefix=self.get_nameprefix(self.get_nameprefix(name))
        name_coef=self.get_namesuffix(name)
        # print(f'name_prefix:{name_prefix}')
        # print(f'name_coef:{name_coef}')
        # 如果没有找到相同key,插入新条目
        if name_prefix not in self.encodedData_con.keys():
            # print(f'CS_NC没有关于该内容的缓存')
            if bool(name_coef):
                name_coef_vec=self.string2vec(name_coef)
                # print(f'no find key, name_coef:{name_coef},name_coef_vec:{name_coef_vec}')
                mat=[]
                mat.append(name_coef_vec)
                golData.set_value(name_prefix,mat)
                qq= golData.get_value(name_prefix)
                # print(f'golData name_prefix:{name_prefix}, qq:{qq}')
                # self.CS_NC[name_prefix] = np.array(name_coef_vec)
                self.encodedData_con[name_prefix] = mat
                # np.save("encodedData_con.npy", np.array(mat))
                # d1 = np.load("encodedData_con.npy", allow_pickle=True)
                # print(f'encodedData_con  d1:{d1}')

                # self.CS_NC_inno[name_prefix]=1
                # print(f'999rank_cs:{self.RS.RS_rank(mat)}')
        else:
            # print(f'find key,当前CS_NC：{self.CS_NC}')
            # 如果存在相同key，且线性无关，则coefs+coef组成新的矩阵作为coefs
            coefs_Interest=self.encodedData_con[name_prefix]
            # name_coef_vec=self.string2vec(name_coef)
            # print(f'-----acoefs_cs:{coefs_cs},{type(coefs_cs)},rank_cs:{self.RS.RS_rank(coefs_cs)}')
            # coefs_=np.row_stack((np.array(coefs_cs), name_coef_vec))
            num_chunks = gol.get_value('num_chunks')
            name_coef_mat=self.string2mat(name_coef,num_chunks)
            coefs_=np.row_stack((np.array(coefs_Interest), name_coef_mat))

            # print(f'666coefs_:{coefs_}')
            # print(f'444rank coefs_cs:{self.RS.RS_rank(coefs_cs)}')
            # print(f'555rank coefs_:{self.RS.RS_rank(np.array(coefs_))}')

            if self.RS.RS_rank(coefs_)>self.RS.RS_rank(coefs_Interest):
                # 新颖数据包，插入CS_NC
                # print('新颖数据包，插入CS_NC')
                self.encodedData_con[name_prefix] = np.array(coefs_)
                # self.CS_NC_inno[name_prefix]=self.CS_NC_inno[name_prefix]+1
                # print(f'app, encodedData_con:{self.encodedData_con}')
                # # 将字典数据写入文件
                # with open("encodedData_con.json", "w") as file:
                #     json.dump(self.encodedData_con, file)
                # np.save("encodedData_con.npy",self.encodedData_con)
                # np.save("encodedData_con.npy", np.array(coefs_))
                # d1 = np.load("encodedData_con.npy", allow_pickle=True)
                # print(f'encodedData_con  d1:{d1}')

                golData.set_value(name_prefix,np.array(coefs_))
                qq= golData.get_value(name_prefix)
                # print(f'golData name_prefix:{name_prefix}, qq:{qq}')
                # self.CS_NC[name_prefix] = np.array(name_coef_vec)
                self.encodedData_con[name_prefix] = np.array(coefs_)
            else:
                pass
                # print('不是新颖包')

    # 供缓存命中的时候调用
    def getencodedData_con(self):
        # 从文件中读取字典数据
        # with open("encodedData_con.json", "r") as file:
        #     encodedData_con = json.load(file)
        # print(f'encodedData_con:{encodedData_con}')
        d1=np.load("encodedData_con.npy",allow_pickle=True)
        return d1

    def onIncomingData(self,data):
        # print(f'-----entering onIncomingData, node id:{self.getNodeid()}, data name:{ data.get_name()}')
        #PIT匹配
        nam=data.get_name()
        flag_NCInter = re.match(r"NC", nam)
        if flag_NCInter:
            # print(f'------start---onIncomingData---NC------------')
            # self.addCS_NC(nam)
            # print(f'----self.CS_NC:{self.CS_NC},PIT_NC:{self.PIT_NC},self.PIT_NC.keys():{self.PIT_NC.keys()}')
            # PIT entry 匹配
            # data_coef 与 PIT_coef 线性无关，则转发数据包到下一个节点
            # 如果线性相关则丢弃
            dataname_prefix=self.get_nameprefix(nam)
            # dataname_coef=self.get_namesuffix(nam)
            # print(f'data prefix:{dataname_prefix},coef:{dataname_coef},self.PIT_NC.keys():{self.PIT_NC.keys()}')
            # self.PIT_NC.keys(): dict_keys(['NC/movie/0/55910827', 'NC/movie/0/54778980'])
            # 暂时没有考虑PIT聚合
            if dataname_prefix in self.PIT_NC.keys():
                # print(f'PIT_NC hit,{dataname_prefix}')
                nextnodeidofdata = self.PIT_NC[dataname_prefix][0]
                """
                nodeid=self.getNodeid()
                appstr = "app"
                # -------收集探测兴趣包从多个方向带回来的数据包的信息------
                # print(f'nextnodeidofdata:{nextnodeidofdata},consumer:{data.get_consumer()}')
                namesuffix = self.get_namesuffix(dataname_prefix)
                # 下一个节点是key node，但是keynode如果没有相应PIT，将丢弃数据包
                # deaddataCV=set()
                suffixCV=self.get_namesuffix(nam)
                cp = data.get_cp().getNodeid()
                nam_cp=nam+"+"+str(cp)
                # print(f'before deaddataCV:{goldeaddataCV.printdeaddataCV()}-nodeid:{nodeid}--name:{data.get_name()}--path:{data.path}->{nextnodeidofdata}')
                # if not goldeaddataCV.find(suffixCV_cp) and data.get_consumer()==nextnodeidofdata and int(namesuffix)<=3:
                if not goldeaddataCV.find(nam_cp) and data.get_consumer() == nextnodeidofdata and int(namesuffix) <= 3:
                    goldeaddataCV.add(nam_cp)
                    # print(f'-start--name:{data.get_name()}--path:{data.path}---suffixCV:{suffixCV}-node id:{nodeid}-dataname_prefix:{dataname_prefix},namesuffix:{namesuffix},recieve data, update golISD, golSIR')
                    print(f'-onIncomingData--deaddataCV:{goldeaddataCV.printdeaddataCV()}')
                    # ----------golSIR-----------------------
                    contnampref=self.get_nameprefix(self.get_nameprefix(data.get_name()))
                    # cp=data.get_cp().getNodeid()
                    golkey = contnampref + "+" +str(cp)
                    golISD_ = golISD.get_value(golkey)
                    delay = data.get_eventtime() + 0.005 - data.get_sendtimeofInt()
                    golISD.append_value(golkey, delay)
                    golSIR.add_recieve(golkey)
                    hopco = data.get_hopcount()+1
                    golHCN.set_value(golkey,hopco)
                    # print(f'--onIncomingData----end------')
                # ----------end-----------------------------
                """
                self.onOutgoingData(data, nextnodeidofdata)
                # self.deletPIT_NC(nam)
                # print(f'before delete self.PIT_NC: {self.PIT_NC}')
                # 删除PIT_NC条目
                del self.PIT_NC[dataname_prefix]
                # print(f'after delete self.PIT_NC: {self.PIT_NC}')

            """
            # 聚合多播
            for i in self.PIT_NC.keys():
                if re.match(dataname_prefix,i):
                    print(f'PIT_CS hit,{dataname_prefix} match {i}')
                    nextnodeidofdata = self.PIT_NC[i][0]
                    self.onOutgoingData(data, nextnodeidofdata)
                    print(f'nextnodeidofdata:{nextnodeidofdata}')
                    # self.deletPIT_NC(nam)
                    # print(f'before delete self.PIT_NC: {self.PIT_NC}')
                    # 删除PIT_NC条目
                    del self.PIT_NC[i]
                    # print(f'after delete self.PIT_NC: {self.PIT_NC}')
                    break
            # print(f'------end---onIncomingData---NC------------')
            """
        else:
            # CS 插入内容
            self.addCS(nam)
            #为了测试，临时插入PIT
            #查询PIT，是否请求过。请求过。
            if nam in self.PIT.keys():
                # print("PIT hit")
                nextnodeidofdata=self.get_lastnodeidfromPIT(data)
                self.onOutgoingData(data,nextnodeidofdata)
                # print(f'nextnodeidofdata:{nextnodeidofdata}')
                self.deletPIT(nam)
            else:
            #data 没有请求过，丢弃数据包
                # print("PIT miss")
                pass
                #下面三行，仅为了测试
                # self.PIT[nam] = (lastnodeid, sendtime)
                # self.PIT[nam] = (data.get_nextnode(), data.get_sendtimeofInt())
                #self.onOutgoingData(data, data.get_nextnode())

    def onOutgoingData(self,data,nextnode):
        # print("----- onOutgoingData","node\t", self.getNodeid(), "send Data packet to nextnode",nextnode)
        data.path=data.path+'->'+str(nextnode)
        # print(f'node id:{self.getNodeid()}, onOutgoingData, name:{data.get_name()},path:{data.path},eventtime:{data.get_eventtime()}, nextnode:{nextnode}')
        with open(self.filename, 'a') as f:
            f.write(
                f'name:{data.get_name()},node id:{self.getNodeid()},nextnode:{nextnode}, onOutgoingData, eventtime:{data.get_eventtime()},sendtime:{data.get_sendtimeofInt()}, path:{data.path},nextnode:{nextnode}\n')

        # 发送数据包到链路层，转信道排队
        # Env 管理带宽资源，包的排队等
        appstr="app"
        if nextnode==appstr:
            # print("appstr=app")
            # print(f'name:{data.get_name()},node id:{self.getNodeid()},nextnode:{nextnode}, onOutgoingData, eventtime:{data.get_eventtime()},sendtime:{data.get_sendtimeofInt()}, path:{data.path},nextnode:{nextnode}\n')
            # self.filename = 'Logfile' + '.txt'
            with open(self.filename, 'a') as f1:
                f1.write(f'name:{data.get_name()},node id:{self.getNodeid()}, onOutgoingData, eventtime:{data.get_eventtime()},sendtime:{data.get_sendtimeofInt()}, path:{data.path},nextnode:{nextnode}\n')

            # print(f'onOutgoingData, node id is {self.getNodeid()},nextnode is app')
            #SI=SI+1 #Env中的满足的兴趣包个数加1

            BGs = "BG"
            intername = data.get_name()
            if (intername.find(BGs) != -1):
                pass
                # print(f'{intername} find {BGs}, and success')
            else:
                # print(f'{intername} not find {BGs}')
                #app接到内容
                # print(f'------start---onOutgoingData---NC------------')
                # 如果是编码包，则需缓存到CS中
                flag_NCInter = re.match(r"NC", intername)
                if flag_NCInter:
                    # 请求编码数据包
                    # print("NCNDN Data packet")
                    nodeid=self.getNodeid()
                    consumerid=data.get_consumer()
                    # print(f'nodeid:{nodeid},{type(nodeid)},consumerid:{consumerid},{type(consumerid)}')
                    if nodeid==int(consumerid):
                        self.addencodedData_con(data)

                    # # ----------golSIR-----------------------
                    # nam = data.get_name()
                    # dataname_prefix = self.get_nameprefix(nam)
                    # namesuffix = self.get_namesuffix(dataname_prefix)
                    # contnampref = self.get_nameprefix(self.get_nameprefix(data.get_name()))
                    # cp = data.get_cp().getNodeid()
                    # key1 = nam + "+" + str(cp)
                    # # -------start--update--innonum----------------
                    # # 可能接收到来自同一个数据源的不同的编码数据包
                    # if int(namesuffix)<=3:
                    #     # 被consumer真正接收到，更新num_inno
                    #     golcps_innonum.append(key1)

                        """
                        # print(f'nam:{nam},namesuffix:{namesuffix},cp:{cp}&&&&printgolcps_innonum:{golcps_innonum.printgolcps_innonum()}')
                        if not goldeaddataCV.find(key1):
                            goldeaddataCV.add(key1)
                            print(f'--onOutgoingData--goldeaddataCV:{goldeaddataCV}')
                            # ----------golSIR-----------------------
                            # cp=data.get_cp().getNodeid()
                            golkey = contnampref + "+" + str(cp)
                            delay = data.get_eventtime() + 0.005 - data.get_sendtimeofInt()
                            golISD.append_value(golkey, delay)
                            golSIR.add_recieve(golkey)
                            hopco = data.get_hopcount()
                            golHCN.set_value(golkey, hopco)
                            # print(f'---onOutgoingData---end------')
                        # --------end-------------------------                      
                      """
                        """
                    golkey = contnampref + "+" + str(cp)
                    if int(namesuffix)>3:
                        golISD_ = golISD.get_value(golkey)
                        delay = data.get_eventtime() + 0.005 - data.get_sendtimeofInt()
                        # print(f'name:{data.get_name()},nodeid:{nodeid},recieve data, update golISD, golSIR')
                        golISD.append_value(golkey, delay)
                        golSIR.add_recieve(golkey)
                        hopco = data.get_hopcount()
                        golHCN.set_value(golkey,hopco)
                    """
                    # print(f'before adding, nodeid:{nodeid},CS_NC:{self.CS_NC}')
                    # self.addCS_NC(intername)
                    # print(f'after adding, nodeid:{nodeid},CS_NC:{self.CS_NC}')
                # print(f'------end---onOutgoingData---NC------------')

                SIN = gol.get_value('SIN')
                SIN = SIN + 1
                gol.set_value('SIN', SIN)
                ISD = gol.get_value('ISD')
                delay=data.get_eventtime()+0.005-data.get_sendtimeofInt()
                ISD=((SIN-1)*ISD+delay)/SIN
                gol.set_value('ISD', ISD)
                HCN=gol.get_value('HCN')
                hopco=data.get_hopcount()
                HCN=((SIN-1)*HCN+hopco)/SIN
                gol.set_value('HCN',HCN)

                # 缓存在关键节点，关键节点统计
                # 内容源对应的跳数、时延、成功率

                # print(f'name:{intername},success,delay:{delay},hopco:{hopco}')
                resultfile = 'map4Roadv2vresult_DRL' + '.txt'
                with open(resultfile, 'a') as f:
                    f.write(f'name:{intername},success,path:{data.path},delay:{delay},hopco:{hopco}\n')

        else:
            # print(f'onOutgoingData, node id is {self.getNodeid()},nextnode is {nextnode}')
            # print('before self.getsendqueue()',self.getsendqueue())
            node_queue_lengths = gol.get_value('node_queue_lengths')
            # print(f'Node:{node_queue_lengths},neighbors:{self.getneighbors()}')
            neigh = self.getneighbors()
            neigh_num = len(neigh)
            neighqueue_len = 0
            for nn in neigh:
                neighqueue_len = neighqueue_len + node_queue_lengths[nn]
            # print(f'nodeid:{nodeid},neigh:{neigh},neigh_len:{neigh_len},len:{len(neigh)}')
            # print(f'neighqueue_len:{neighqueue_len},neigh_num:{neigh_num}')
            # 设置丢包率
            alpha = 0.0015
            # reliablepacketrate = 1 - alpha * neighqueue_len
            reliablepacketrate = 1
            # print(f'Node onOutgoingData, neighqueue_len:{neighqueue_len},reliablepacketrate,{reliablepacketrate}')

            # 设置丢包率
            # undroppacketrate=0.7
            if random.uniform(0, 1) < reliablepacketrate:
                # 更新eventtime
                if self.getNodeid()!=data._cp:
                    data.quedelay_cur = len(self.getsendqueue()) * 0.005
                    #设置时延
                    # trans_delay=0.1
                    channel=Channel(int(self.getNodeid()),int(nextnode))
                    evtime=data.get_eventtime()+data.quedelay_cur+channel.transdelay
                    data.set_eventtime(evtime)

                BGs = "BG"
                intername = data.get_name()
                if (intername.find(BGs) != -1):
                    pass
                    # print(f'{intername} find {BGs}')
                else:
                    # print(f'{intername} not find {BGs}')
                    FDP = gol.get_value('FDP')
                    FDP = FDP + 1
                    gol.set_value('FDP', FDP)
                data.update_hopcount()
                data.set_nextnode(nextnode)
                self.addpacket(data)

                # print('after self.getsendqueue()',self.getsendqueue())
                # FDP=FDP+1 #Env里的转发数据包个数加1

class Vnode(Node):
    def __init__(self,nodeid,pos=[0,0]):
        super().__init__(nodeid,pos)
        self.position=pos

    def initposition(self):
        pass
        # print("entering Vnode setposition")
        #等间隔位置
        # if type(self.nodeid) == int or self.nodeid.isdigit():
        #     xx = int(self.nodeid) * self.vehijiange
        #     self.position = [xx, 0]
        #     # print(f'node id:{self.nodeid},position,{self.position}')
        # if type(self.nodeid) == int or self.nodeid.isdigit():
        #     # xx = int(self.nodeid) * self.vehijiange
        #     xx= random.randint(1, self.roadlength)
        #     self.position = [xx, 0]
        #     print(f'node id:{self.nodeid},position,{self.position}')

class RNode(Node):
    def __init__(self,nodeid,pos=[0,0]):
        super().__init__(nodeid,pos)
        #DRL
        self.EPSILON = 0.3
        self.update_epsilon = False
        self.BATCH_SIZE = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GAMMA = 0.90
        self.reward=[]
        self.actionlist=[]
        self.loss=[]
        self.learn_counter=0
        self.Q_NETWORK_ITERATION=100
        self.num_action_epsilon=0

    # 根据状态采取某个动作
    def act(self, neural_network, state, cps, num, will_learn):
        ''' 我们将随机游走或以概率 epsilon 引用 Q 表 '''
        # if will_learn:
        #     if num < 10:
        #         self.EPSILON = 0.7

        print(f'action_epsilon:{neural_network.policy_net.num_action_epsilon}')
        neural_network.policy_net.num_action_epsilon = neural_network.policy_net.num_action_epsilon + 1

        if will_learn:
            if neural_network.policy_net.num_action_epsilon<100:
                self.EPSILON=0.9
            elif neural_network.policy_net.num_action_epsilon<500:
                self.EPSILON=0.7
            elif neural_network.policy_net.num_action_epsilon<1000:
                self.EPSILON=0.5
            else:
                self.EPSILON=0.1
        else:
            self.EPSILON = 0.01

        # print(f'node id:{self.Rnodeid}, neural_network.policy_net.num_action_epsilon: {neural_network.policy_net.num_action_epsilon},self.EPSILON:{self.EPSILON}')
        # print(f'neural_network.num_action_epsilon: {neural_network.num_action_epsilon},self.EPSILON:{self.EPSILON}')

        if random.uniform(0, 1) < self.EPSILON:
            print(f'willlearn:{will_learn},choice随机')
            resultfile = 'map4Roadv2vresult_DRL' + '.txt'
            with open(resultfile, 'a') as f:
                f.write(f'willlearn:{will_learn},choice random\n')
            ''' 我们将随机游走或以概率 epsilon 引用 Q 表 '''
            if not bool(cps):
                # 检查数据包的当前节点是否有任何可用的邻居
                return None
            else:
                next_step = random.choice(cps)  # 探索行动空间
                next_step_idx=cps.index(next_step)
        else:
            print(f'willlearn:{will_learn},choice最大')
            if not bool(cps):
                return None
            else:
                ''' 通过参考我们的神经网络获得下一个最佳邻居以将数据包从其当前节点移动 '''
                with torch.no_grad():    # 不计算梯度
                    qvals = neural_network.policy_net(state.float())  # state表示当前数据包的所在的卫星节点和当前节点的数据包队列长度和最大队列节点的拼接
                    print(f'qvals:{qvals}')

                    resultfile = 'map4Roadv2vresult_DRL' + '.txt'
                    with open(resultfile, 'a') as f:
                        f.write(f'willlearn:{will_learn},choice max,qvals:{qvals}\n')

                    # next_step_idx = qvals[:, cps].argmax().item()
                    # wjxgai
                    next_step_idx = qvals.argmax().item()
                    next_step = cps[next_step_idx]
                    # print(f'entering act, state:{state},cps:{cps},next_step_idx:{next_step_idx},next_step:{next_step}')

                    if self.update_epsilon:
                        self.EPSILON = self.decay_rate * self.EPSILON
                        self.update_epsilon = False

                    # neural_network.num_action_epsilon=neural_network.num_action_epsilon + 1

        return next_step_idx,next_step

    # 训练神经网络
    def learn(self, nn, current_state, action, reward, next_state, terminal_flag, adjacency_matrix = []):
        if self.learn_counter % self.Q_NETWORK_ITERATION ==0:
            nn.target_net.load_state_dict(nn.policy_net.state_dict())
            # print(f'nn.target_net.load_state_dict(nn.policy_net.state_dict()),{nn.policy_net.state_dict()}')
        self.learn_counter+=1

        ''' 如果没有提供有效动作或没有提供奖励，则跳过 '''
        if (action == None) or (reward == None):
            pass
        else:
            if current_state != None:
                nn.replay_memory.push(current_state, action,  next_state, reward, terminal_flag)
            # print(f'nn.replay_memory:{len(nn.replay_memory),nn.replay_memory.memory}')

            if (nn.replay_memory.can_provide_sample(self.BATCH_SIZE)):
                # experiences = nn.replay_memory.take_recent(self.BATCH_SIZE)
                experiences = nn.replay_memory.sample(self.BATCH_SIZE)
                # print(f'experiences:{len(experiences),experiences}')

                states, actions, next_states, rewards,terminal_flags = self.extract_tensors(experiences)
                current_q_values = self.get_current_QVal(nn.policy_net, states, actions)

                qvalue_max=max(current_q_values)
                qvalue_min=min(current_q_values)

                next_q_values = self.get_next_QTarget(nn.target_net, next_states, actions, terminal_flags, adjacency_matrix,qvalue_max,qvalue_min)
                target_q_values = (next_q_values * self.GAMMA) + rewards
                # print(f'current_q_values:{torch.transpose(current_q_values, 0, 1)},next_q_values:{next_q_values},rewards:{rewards},target_q_values:{target_q_values}')
                loss = F.mse_loss(current_q_values,
                                  torch.transpose(target_q_values, 0, 1))
                nn.optimizer.zero_grad()
                loss.backward()
                nn.optimizer.step()
                return loss.item()

    def extract_tensors(self, experiences):
        # nn.replay_memory.push(current_state, action, next_state, reward, terminal_flag)
        states = torch.cat(tuple(exps[0] for exps in experiences))
        actions = torch.cat(
            tuple(torch.tensor([exps[1]]) for exps in experiences))
        next_states = torch.cat(tuple(exps[2] for exps in experiences))
        rewards = torch.cat(
            tuple(torch.tensor([exps[3]], dtype=torch.float) for exps in experiences))
        # 添加采取行动action，是否终止
        terminal_flags=torch.cat(
            tuple(torch.tensor([exps[4]], dtype=torch.float) for exps in experiences))
        return (states, actions, next_states, rewards,terminal_flags)


    def get_current_QVal(self, policy_net, states, actions):
        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))


    def get_next_QTarget(self, target_net, next_states, actions, terminal_flags, adjacency_matrix,qvalue_max,qvalue_min):

        # 如果到达目的地，next_QTarget=0，否则取最大的Q值
        ''' 在取最大值之前需要将邻居掩码应用于 target_net() '''
        # print(f'entering get_next_QTarget,terminal_flags:{terminal_flags}')
        # non_terminal_idx = (actions != destination)
        temp1 = target_net(next_states.float())

        # print(f'temp1:{temp1}')

        ''' 初始化零值向量 '''
        batch_size = next_states.shape[0]
        values_nextQTarget = torch.zeros(batch_size).view(1, -1).to(self.device)

        for idx in range(values_nextQTarget.size()[1]):
            """
            #original Q value of target network
            values_nextQTarget[0, idx] = torch.max(temp1[idx]).detach()
            """
            """"""
            # 没有终止，计算target 网络 max Q值
            if terminal_flags[idx]==0:
                # wjx
                values_nextQTarget[0, idx] = torch.max(temp1[idx]).detach()
                #可能用来判断action是否存在邻居节点，这样可以考虑两跳邻居节点
                # if len(adjacency_matrix) != 0:
                #     adjs = adjacency_matrix[actions[idx]][0]   #将actions这个节点的邻接矩阵的该行找出来
                #     adjs = (adjs == 1)   # 让其从整型变为boolean型
                #     temp2 = temp1[idx, :].view(1, -1)  #temp1的值
                # 目的地的目标网络的Q值
                #     values_nextQTarget[0, idx] = torch.max(temp2[adjs]).detach()
            # wjx改
            elif terminal_flags[idx]==1:
                # find cp
                values_nextQTarget[0, idx] = qvalue_max
            elif terminal_flags[idx]==-1:
                # 请求失败
                values_nextQTarget[0, idx] = min(qvalue_min,0)
            # print(f'values_nextQTarget:{values_nextQTarget},idx:{idx},qvalue_max:{qvalue_max},qvalue_min:{qvalue_min},flag:{terminal_flags[idx]},Q:{torch.max(temp1[idx]).detach()}')
        return values_nextQTarget


class Interest():
    def __init__(self,eventtime,name,nonce,consumer,sendtime,lastnodeid,currentnodeid,nextnodeid,flag_probe=0):
        self._name = name
        self._nonce = nonce
        self._sendtime = sendtime
        self._consumer = consumer
        self._recievetime = 0
        self._delay = 0
        self._hopcount = 0
        self._CPlist = []
        self._lastnode= Node(lastnodeid)
        # 在Env生成包时，生成包时消费者currentnodeid初始化节点为应用层'app'，nextnodeid初始化为消费者id
        self._currentnode = Node(currentnodeid)
        self._nextnode = Node(nextnodeid)
        self._eventtime=eventtime
        self.path=''
        self.quedelay_cur=0
        self.quedelay_next=0
        self.pathnode=set()
        self.cp_selected=0
        self.router=0
        self.flag_probe=flag_probe

    def __str__(self):
        return self._name

    def get_packettype(self):
        return "Interest"

    def get_flag_probe(self):
        return self.flag_probe

    def set_flag_probe(self,flag_probe):
        self.flag_probe=flag_probe

    def set_eventtime(self,time):
        self._eventtime=time

    def get_eventtime(self):
        return self._eventtime

    def get_path(self):
        return self.path

    def update_path(self,nextnodestr):
        # print(f'before:{self.path}')
        self.path=self.path+'->'+str(nextnodestr)
        # print(f'after update:{self.path}\n')
        # self.pathnode.add(nextnodestr)

    def get_name(self):
        return self._name

    def set_nonce(self, nonce):
        self._nonce = nonce

    def get_nonce(self):
        return self._nonce

    def set_sendtime(self,sendtime1):
        self._sendtime=sendtime1

    def get_sendtime(self):
        return self._sendtime

    def set_recievetime(self,recievetime1):
        self._recievetime=recievetime1

    def get_recievetime(self):
        return self._recievetime


    def set_delay(self):
        self._delay=self._recievetime-self._sendtime

    def get_delay(self):
        return self._delay

    def set_hopcount(self,hopcount):
        self._hopcount=hopcount

    def update_hopcount(self):
        self._hopcount=self._hopcount+1

    def get_hopcount(self):
        return self._hopcount

    def set_consumer(self, consumer):
        self._consumer = consumer

    def get_consumer(self):
        return self._consumer

    def set_lastnode(self,lastnode):
        self._lastnode=lastnode

    def get_lastnode(self):
        return self._lastnode

    def set_currentnode(self,currentnode):
        self._currentnode=currentnode

    def get_currentnode(self):
        return self._currentnode

    def get_nextnode(self):
        return self._nextnode

    def set_nextnode(self,nextnode):
        self._nextnode=nextnode

    def get_nextnode(self):
        return self._nextnode

    # 兴趣包发送到下一个节点，应该有什么功能呢？
    def updatenodeid_interest(self):
        #在onIncomingInterest调用此函数，改变兴趣包上的当前、下一跳节点id；
        # print(f'--updatenodeid_interest---_currentnode:{self._currentnode},nextnode:{self._nextnode}')
        self.set_lastnode(self._currentnode)
        self.set_currentnode(self._nextnode)
        self.set_nextnode('999')

    def dosendInterest(self,nextnode):
        pass


class Data():
    def __init__(self,eventtime,name,contentprovider,sendtime,replytime,currenttime,lastnodeofinterest,payload,num_innov_=0,flag_probe=0):
        self._name = name
        self._payload=payload
        self._cp = contentprovider
        self._sendtimeofInt = sendtime
        self._replytime = replytime  #replytime data from cp
        self._currenttime=currenttime
        self._delay = 0
        self._hopcount = 0  #从cp初始化为0，从cp->consumer开始计算跳数
        self._currentnode = Node('0')
        self._nextnode = lastnodeofinterest
        self._eventtime = eventtime
        self._consumer=9999
        self.path=''
        self.quedelay_cur = 0
        self.quedelay_next = 0
        self.num_innov_=num_innov_
        self.flag_probe = flag_probe

    def __str__(self):
        return self._name

    def get_flag_probe(self):
        return self.flag_probe

    def set_flag_probe(self,flag_probe):
        self.flag_probe=flag_probe

    def get_cp(self):
        return self._cp

    def set_cp(self,cp):
        self._cp=cp


    def get_packettype(self):
        return "Data"

    def set_eventtime(self,time):
        self._eventtime=time

    def get_eventtime(self):
        return self._eventtime

    def get_name(self):
        return self._name

    def get_path(self):
        return self.path

    def update_path(self,nextnodestr):
        print(f'before:{self.path}')
        self.path=self.path+'->'+str(nextnodestr)
        print(f'after update path:{self.path}\n')

    def set_sendtimeofInt(self,sendtime1):
        self._sendtimeofInt=sendtime1

    def get_sendtimeofInt(self):
        return self._sendtimeofInt

    def set_replytime(self,replytime):
        self._replytime=replytime

    def get_replytime(self):
        return self._replytime

    def set_delay(self):
        self._delay=self._currenttime-self._sendtimeofInt

    def get_delay(self):
        return self._delay

    def set_hopcount(self,hopcount):
        self._hopcount=hopcount

    def update_hopcount(self):
        self._hopcount=self._hopcount+1

    def get_hopcount(self):
        return self._hopcount

    def set_consumer(self, consumer):
        self._consumer = consumer

    def get_consumer(self):
        return self._consumer

    def set_lastnode(self,lastnode):
        self._lastnode=lastnode

    def get_lastnode(self):
        return self._lastnode

    def set_currentnode(self,currentnode):
        self._currentnode=currentnode

    def get_currentnode(self):
        return self._currentnode

    def get_nextnode(self):
        return self._nextnode

    def set_nextnode(self,nextnode):
        self._nextnode=nextnode

    def get_nextnode(self):
        return self._nextnode

    # 兴趣包发送到下一个节点，应该有什么功能呢？
    def send_to_nextnode(self):
        self.set_lastnode(self._currentnode)
        self._currentnode=self.set_currentnode(self._nextnode)


class Map():
    def __init__(self,roadnum=4,roadvehicledensitylist=[2,4,6,4]):
        self.roadsvehiclesnum=0
        self.roadnum=roadnum
        self.Roadlist=[]
        self.initmap()
        self.roadvehicledensitylist=roadvehicledensitylist
        self.roadvehiclenumlist=[]
        self.setvehiclenumlist_from_density(self.roadvehicledensitylist)
        self.maxvehicleid=0
        self.addroadsvehicles()
        self.roadsvehiclesidpos = {}
        self.roadsvehiclesidRD={}       #车辆id对应的道路方向
        self.setroadsvehiclesidpos()
        self.intersectionpos=[3000,3000]
        self.comm_range=500
        self.intersectionRSUid=0
        self.addintersectionvehicle()
        # [1728.0, 2934.0, 2802.0, 48.0]
        self.dis_link_blind_spotlist=self.get_dis_link_blind_spotlist()

    def get_dis_link_blind_spotlist(self):
        dis_link_blind_spotlist=[]
        for i in range(self.roadnum):
            # print(f'roaddirection:{self.Roadlist[i].roaddirection},self.Roadlist[i].vehiclesidpos:{ self.Roadlist[i].vehiclesidpos}')
            roaddirection=self.Roadlist[i].roaddirection
            vehiclesposlist=[]
            if roaddirection==0:
                vehiclesposlist.append(self.intersectionpos[0])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][0])
                if vehiclesposlist[1]>self.intersectionpos[0]:
                    vehiclesposlist.sort()
                else:
                    vehiclesposlist.sort(reverse = True)
            else:
                vehiclesposlist.append(self.intersectionpos[1])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][1])
                if vehiclesposlist[1]>self.intersectionpos[1]:
                    vehiclesposlist.sort()
                else:
                    vehiclesposlist.sort(reverse = True)
            # print(f'vehiclesposlist:{vehiclesposlist}')

            blind_spot_dis=0
            for m in range(len(vehiclesposlist)-1):
                if abs(vehiclesposlist[m+1]-vehiclesposlist[m])<self.comm_range:
                    blind_spot_dis=abs(vehiclesposlist[m+1]-vehiclesposlist[0])
                else:
                    break
            dis_link_blind_spotlist.append(blind_spot_dis)
        # print(f'dis_link_blind_spotlist:{dis_link_blind_spotlist}')
        return dis_link_blind_spotlist

    # 获得到节点cp的转发列表
    def get_roadforwarderlist(self):
        # print(f'cps:{cps}')
        forwarderlists = []
        for i in range(self.roadnum):
            # print(f'roaddirection:{self.Roadlist[i].roaddirection},self.Roadlist[i].vehiclesidpos:{ self.Roadlist[i].vehiclesidpos}')
            roaddirection = self.Roadlist[i].roaddirection
            vehiclesposlist = []
            vehiclesidlist=[]
            if roaddirection == 0:
                # vehiclesposlist.append(self.intersectionpos[0])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][0])
                if vehiclesposlist[1] > self.intersectionpos[0]:
                    vehiclesposlist.sort()
                    # print(f'vehiclesposlist[1] > self.intersectionpos[0]:{vehiclesposlist}')
                    # 从小到大排队，从十字路口排队
                else:
                    vehiclesposlist.sort(reverse=True)
                    # 从大到小排队，从十字路口排队
                    # print(f'vehiclesposlist[1] < self.intersectionpos[0]:{vehiclesposlist}')

                # print(f'vehiclesposlist:{vehiclesposlist},cpi_pos:{cpi_pos}')

                # 筛选转发者
                forwarderlisti = []
                posi = []
                # posi.append(vehiclesposlist[0])
                pos0 = self.intersectionpos[0]
                pos=pos0
                for j in range(len(vehiclesposlist)):
                    if abs(pos - vehiclesposlist[j]) <=self.comm_range:
                        pass
                    else:
                        # print(f'pos:{pos},j:{vehiclesposlist[j]},j-1:{vehiclesposlist[j-1]}')
                        pos = vehiclesposlist[j - 1]
                        posi.append(pos)

                # print(f'posi:{posi}')
                # forwarderlists.append(posi)

                forwarder_ids = []
                forwarder_ids.append(self.intersectionRSUid)
                for posii in posi:
                    # print(f'posii:{posii}')
                    # print(f'vehiclesidpos:{self.Roadlist[i].vehiclesidpos}')
                    for vid in self.Roadlist[i].vehiclesidpos:
                        if posii == self.Roadlist[i].vehiclesidpos[vid][0]:
                            posii_id = vid
                            forwarder_ids.append(posii_id)
                            break
                # print(f'forwarder_ids:{forwarder_ids}')
                forwarderlists.append(forwarder_ids)
            else:
                vehiclesposlist.append(self.intersectionpos[1])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][1])
                if vehiclesposlist[1] > self.intersectionpos[1]:
                    vehiclesposlist.sort()
                else:
                    vehiclesposlist.sort(reverse=True)
                # print(f'vehiclesposlist:{vehiclesposlist}')

                # 筛选转发者
                forwarderlisti = []
                posi = []
                # posi.append(vehiclesposlist[0])
                pos = vehiclesposlist[0]
                for j in range(len(vehiclesposlist)):

                    if abs(pos - vehiclesposlist[j]) < self.comm_range:
                        pass
                    else:
                        pos = vehiclesposlist[j - 1]
                        posi.append(pos)

                # print(f'posi:{posi}')
                # forwarderlists.append(posi)

                forwarder_ids = []
                forwarder_ids.append(self.intersectionRSUid)
                for posii in posi:
                    # print(f'posii:{posii}')
                    # print(f'vehiclesidpos:{self.Roadlist[i].vehiclesidpos}')
                    for vid in self.Roadlist[i].vehiclesidpos:
                        if posii == self.Roadlist[i].vehiclesidpos[vid][1]:
                            posii_id = vid
                            forwarder_ids.append(posii_id)
                            break
                # print(f'forwarder_ids:{forwarder_ids}')
                forwarderlists.append(forwarder_ids)

        # print(f'forwarderlists:{forwarderlists}')
        # [[60, 1, 14, 16, 11, 8, 3], [60, 31, 29, 54, 42, 44, 46]]
        return forwarderlists

    # 获得到节点cp的转发列表
    def get_forwarderlist(self,cps):
        # print(f'cps:{cps}')
        forwarderlists = []
        for i in range(self.roadnum):
            # print(f'roaddirection:{self.Roadlist[i].roaddirection},self.Roadlist[i].vehiclesidpos:{ self.Roadlist[i].vehiclesidpos}')
            roaddirection = self.Roadlist[i].roaddirection
            vehiclesposlist = []
            vehiclesidlist=[]
            if roaddirection == 0:
                # vehiclesposlist.append(self.intersectionpos[0])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][0])
                if vehiclesposlist[1] > self.intersectionpos[0]:
                    vehiclesposlist.sort()
                    # print(f'vehiclesposlist[1] > self.intersectionpos[0]:{vehiclesposlist}')
                    # 从小到大排队，从十字路口排队
                else:
                    vehiclesposlist.sort(reverse=True)
                    # 从大到小排队，从十字路口排队
                    # print(f'vehiclesposlist[1] < self.intersectionpos[0]:{vehiclesposlist}')
                cpi = cps[i]
                cpi_pos=self.roadsvehiclesidpos[cpi][0]
                # print(f'vehiclesposlist:{vehiclesposlist},cpi_pos:{cpi_pos}')
                # 筛选转发者
                forwarderlisti = []
                posi = []
                # posi.append(vehiclesposlist[0])
                pos0 = self.intersectionpos[0]
                pos=pos0
                for j in range(len(vehiclesposlist)):
                    if abs(pos0-vehiclesposlist[j])<=abs(pos0-cpi_pos):
                        if abs(pos - vehiclesposlist[j]) <=self.comm_range:
                            pass
                        else:
                            # print(f'pos:{pos},j:{vehiclesposlist[j]},j-1:{vehiclesposlist[j-1]}')
                            pos = vehiclesposlist[j - 1]
                            posi.append(pos)
                    else:
                        break
                # print(f'posi:{posi}')
                # forwarderlists.append(posi)

                forwarder_ids = []
                forwarder_ids.append(self.intersectionRSUid)
                for posii in posi:
                    # print(f'posii:{posii}')
                    # print(f'vehiclesidpos:{self.Roadlist[i].vehiclesidpos}')
                    for vid in self.Roadlist[i].vehiclesidpos:
                        if posii == self.Roadlist[i].vehiclesidpos[vid][0]:
                            posii_id = vid
                    forwarder_ids.append(posii_id)
                forwarder_ids.append(cpi)
                # print(f'forwarder_ids:{forwarder_ids}')
                forwarderlists.append(forwarder_ids)
                # print(f'forwarderlists......:{forwarderlists}')

            else:
                vehiclesposlist.append(self.intersectionpos[1])
                for j in self.Roadlist[i].vehicles:
                    vehiclesposlist.append(self.Roadlist[i].vehiclesidpos[j][1])
                if vehiclesposlist[1] > self.intersectionpos[1]:
                    vehiclesposlist.sort()
                else:
                    vehiclesposlist.sort(reverse=True)
                # print(f'vehiclesposlist:{vehiclesposlist}')
                cpi = cps[i]
                cpi_pos=self.roadsvehiclesidpos[cpi][1]

                # 筛选转发者
                forwarderlisti = []
                posi = []
                # posi.append(vehiclesposlist[0])
                pos = vehiclesposlist[0]
                for j in range(len(vehiclesposlist)):
                    if vehiclesposlist[j]<=cpi_pos:
                        if abs(pos - vehiclesposlist[j]) < self.comm_range:
                            pass
                        else:
                            pos = vehiclesposlist[j - 1]
                            posi.append(pos)
                    else:
                        break
                # print(f'posi:{posi}')
                # forwarderlists.append(posi)

                forwarder_ids = []
                forwarder_ids.append(self.intersectionRSUid)
                for posii in posi:
                    # print(f'posii:{posii}')
                    # print(f'vehiclesidpos:{self.Roadlist[i].vehiclesidpos}')
                    for vid in self.Roadlist[i].vehiclesidpos:
                        if posii == self.Roadlist[i].vehiclesidpos[vid][1]:
                            posii_id = vid
                    forwarder_ids.append(posii_id)
                forwarder_ids.append(cpi)
                # print(f'forwarder_ids:{forwarder_ids}')
                forwarderlists.append(forwarder_ids)

        # print(f'forwarderlists......1:{forwarderlists}')
        return forwarderlists


    def initmap(self):
        if self.roadnum==2:
            self.Roadlist.append(Road([0,3000],[3000,3000],0))
            self.Roadlist.append(Road([3000,3000],[6000,3000],0))
        elif self.roadnum==3:
            self.Roadlist.append(Road([0,3000],[3000,3000],0))
            self.Roadlist.append(Road([3000,3000],[6000,3000],0))
            self.Roadlist.append(Road([3000,3000],[3000,6000],math.pi/2))
        elif self.roadnum==4:
            self.Roadlist.append(Road([0,3000],[3000,3000],0))
            self.Roadlist.append(Road([3000,3000],[6000,3000],0))
            self.Roadlist.append(Road([3000,0],[3000,3000],math.pi/2))
            self.Roadlist.append(Road([3000,3000],[3000,6000],math.pi/2))

    def setvehiclenumlist_from_density(self,roadvehicledensitylist):
        # print(f'roadvehicledensitylist:{roadvehicledensitylist},roadnum:{self.roadnum}')
        for i in range(self.roadnum):
            venum=roadvehicledensitylist[i]*self.Roadlist[i].maxhop
            # print(f'density:{roadvehicledensitylist[i]},num:{venum}')
            self.roadvehiclenumlist.append(int(venum))
        # print(f'vehicle density:{self.roadvehicledensitylist},all roads vehicle num:{self.roadvehiclenumlist}')

    def printmap(self):
        for i in range(len(self.Roadlist)):
            print(f'road {i}, startpos:{self.Roadlist[i].startpos},endpos:{self.Roadlist[i].endpos},roaddirection:{self.Roadlist[i].roaddirection}')
            print(f'vehiclesnum:{self.Roadlist[i].vehiclesnum},vehicles:{self.Roadlist[i].vehicles},roadsvehiclesnum:{self.roadsvehiclesnum}')

    # 打印所有道路上车辆的位置
    def printroadsvehicles(self):
        for i in range(self.roadnum):
            for j in range(self.Roadlist[i].vehiclesnum):
                print(self.Roadlist[i].vehicles)

    # 按照roadvehiclenumlist为每个道路设置相应数量的车辆
    def addroadsvehicles(self):
        for i in range(self.roadnum):
            self.Roadlist[i].addroadvehicles(self.maxvehicleid,self.roadvehiclenumlist[i])
            self.maxvehicleid=self.maxvehicleid+self.roadvehiclenumlist[i]

    # 设置所有道路车辆id的位置
    def setroadsvehiclesidpos(self):
        for i in range(self.roadnum):
            self.roadsvehiclesidpos.update(self.Roadlist[i].vehiclesidpos)
            self.roadsvehiclesidRD.update(self.Roadlist[i].vehiclesidRD)
            # print(f'after adding road {i}, the all roads vehiclesidpos are {self.roadsvehiclesidpos}')
        self.roadsvehiclesnum=len(self.roadsvehiclesidpos)

    def setroadsvehiclesidpos1(self,roadsvehiclesidpos,roadsvehiclesidRD):
        self.roadsvehiclesidpos=roadsvehiclesidpos
        self.roadsvehiclesidRD=roadsvehiclesidRD

    # 得到道路车辆id的位置
    def getroadsvehiclesidpos(self):
        return self.roadsvehiclesidpos

    # 没有用过道路方向列表
    def getRDlist(self,nodeid):
        RDlist=[0,math.pi,math.pi/2,-math.pi/2]
        pos=self.roadsvehiclesidpos[nodeid]
        #判断是否在十字路口
        dis=math.sqrt((pos[0]-self.intersectionpos[0])**2+(pos[1]-self.intersectionpos[1])**2)
        print(f'pos：{pos},dis:{dis}')
        # 如果在十字路口
        if dis<self.comm_range/2:
            return RDlist
        else:
            # 如果不在十字路口
            for road in self.Roadlist:
                if nodeid in road.vehicles:
                    return [road.roaddirection,road.roaddirection-math.pi]

    # 在十字路口部署车辆
    def addintersectionvehicle(self):
        # self.maxvehicleid=self.maxvehicleid+1
        self.roadsvehiclesnum=self.roadsvehiclesnum+1
        self.roadsvehiclesidpos[self.maxvehicleid]=self.intersectionpos
        self.intersectionRSUid=self.maxvehicleid


class Road():
    def __init__(self,startpos,endpos,roaddirection):
        self.startpos=startpos
        self.endpos=endpos
        self.roadlength=self.getroadlength()
        self.roaddirection=roaddirection
        self.vehicles=[]
        self.vehiclesidpos={}
        self.vehiclesidRD={}
        self.vehiclesnum=0
        self.comm_range=500
        self.maxhop=self.roadlength/self.comm_range
        self.pathnode=list()
        self.dis_link_blind_spot=[]


    # 获得车辆道路方向
    def get_vehicleroaddirection(self,vehicleid):
        if vehicleid in self.vehicles:
            return self.roaddirection


    # 道路包含的车辆数目
    def get_vehiclesnum(self):
        return len(self.vehicles)

    # 一跳通信范围内有几个车
    def get_vehicledensity(self):
        den=self.vehiclesnum/self.roadlength*self.comm_range

    # 道路长度
    def getroadlength(self):
        self.roadlength=math.sqrt((self.startpos[0]-self.endpos[0])**2+(self.startpos[1]-self.endpos[1])**2)
        return self.roadlength

    # def setVNodelist(self, num1,num):
    #     Vnodelist = []
    #     for i in range(self.num_of_vehicles):
    #         Vnodelist.append(Vnode(i, pos[i]))
    #     return Vnodelist

    # 为该道路添加num个车辆，从序号num1开始
    def addroadvehicles(self,num1,num):
        for i in range(num):
            self.addvehicle(num1+i)

    # 道路添加车辆，随机设置车辆位置，并返回车辆位置
    def addvehicle(self,vehicleid):
        self.vehicles.append(vehicleid)
        self.vehiclesnum = self.vehiclesnum + 1
        pos = random.randint(1, 1000)/1000
        if self.roaddirection==math.pi or self.roaddirection==0:
            pos1=self.startpos[0]+pos*self.roadlength
            self.vehiclesidpos[vehicleid]=[pos1,3000]
        elif self.roaddirection==math.pi/2 or self.roaddirection==-math.pi/2:
            pos1=self.startpos[1]+pos*self.roadlength
            self.vehiclesidpos[vehicleid]=[3000,pos1]
        self.vehiclesidRD[vehicleid]=self.roaddirection

    def removevehicle(self,vehicleid):
        # 根据值移除元素
        self.vehicles.remove(vehicleid)
        del self.vehiclesidpos[vehicleid]
        del self.vehiclesidRD[vehicleid]
        self.vehiclesnum=self.vehiclesnum-1

    def addvehicles(self,vehiclesid):
        self.vehicles.update(vehiclesid)
        self.vehiclesnum=self.vehiclesnum+len(vehiclesid)

    def vehi_randomchoice(self):
        vehinum=random.choice(self.vehicles)
        # print(f'vehinum:{vehinum}，pos:{self.vehiclesidpos[vehinum]}')
        return vehinum

    def getdis(self,v2):
        p1=[3000,3000]
        p2=self.vehiclesidpos[v2]
        dis=pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)
        # print(f'p1:{p1},v2:{v2},p2:{p2},dis:{dis}')
        return dis

    def vehi_choicedismax(self):
        dismax=0
        vehi_dismax=0
        for vehi in self.vehicles:
            d1=self.getdis(vehi)
            if d1>dismax:
                dismax=d1
                vehi_dismax=vehi
        # print(f'vehi_dismax:{vehi_dismax}，pos:{self.vehiclesidpos[vehi_dismax]}')
        return vehi_dismax


