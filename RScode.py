import numpy as np

class RScode:
    def __init__(self):
        self.m_alpha_to=[1,2,4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9]
        self.m_index_of=[0,1,4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12, 0]
        # 本原多项式 1+m+m^4=0

    # 两个伽罗华域的元素相加
    def RS_add(self,a,b):
        # print('a',a)
        bina=self.dec2binreverselist(int(a))
        # print('a',a,'bina',bina)
        binb=self.dec2binreverselist(b)
        # print('binb', binb)
        abxor_nixu=[]
        for i in range(4):
            abxor_nixu.append(int(bina[i])^int(binb[i]))
        #print('abxor',abxor_nixu)
        value=0
        for j in range(4):
            value+=abxor_nixu[j]*pow(2,j)

        return value

    # 两个伽罗华域中的元素相乘
    def RS_mul(self,a,b):
        if (a==0)|(b==0):
           c=0
        else:
            a_indx=self.m_index_of[a-1]
            b_indx=self.m_index_of[b-1]
            c_indx=(a_indx + b_indx) % 15
            c=self.m_alpha_to[c_indx]
        return c

    # 伽罗华域中元素的倒数
    def RS_inverse(self,x):
        if x==1:
            return 1
        else:
            for i in range(15):
                if self.m_alpha_to[i]==x:
                    x_indx=i
            y_indx=15-x_indx
            y_inverse=self.m_alpha_to[y_indx]
            return y_inverse

    def RS_reencode_mat(self,mat,num_chunks):
        # coefs_cs: [[1 2 7 2 0]
        #            [0 1 4 0 7]
        #             [0 0 1 5 7]]
        #随机生成一个系数向量
        #矩阵乘以系数向量[1,2,3]
        # print(f'------RS_reencode_mat-------mat:{mat},{type(mat)}')
        coef=self.coefficients_generate(np.array(mat).shape[0])

        # print(f'entering RS_reencode_mat,mat:{mat},coef:{coef}')
        coef_ = np.zeros(num_chunks,dtype=np.int64)
        # print(f'coef_:{coef_}')

        for j in range(num_chunks):
            for i in range(np.array(mat).shape[0]):
                coef_[j]=self.RS_add(coef_[j],self.RS_mul(mat[i][j],coef[i]))
                # print(f'coef_:{coef_}')
            # print(f'coef_:{coef_}')
        # print(f"7777mat:{mat},hang:{mat.shape[0]},lie:{mat.shape[1]},coef:{coef},coef_:{coef_}")
        return coef_


    def RS_rank(self,mat):
        # print(f'entering RS_rank, mat:{mat},{type(mat)}')
        # print(np.size(mat,1))
        # print(np.size(mat,0))
        if np.size(mat,0)<np.size(mat,1):
            matrowcolmin=np.size(mat,0)
        else:
            matrowcolmin=np.size(mat, 1)
        #print(matrowcolmin)
        # print('高斯列消元前：')
        # print(mat)

        for i in range(matrowcolmin):
            hang=i
            if mat[i][i]==0:
                flag=0
                for m in range(i+1,matrowcolmin):
                    if mat[m][i]!=0:
                        hangZero=m
                        flag=1
                        break

                if flag==1:
                    # 交换hangZero行和第i行
                    listss=[]
                    for ss in range(np.size(mat,0)):
                        listss.append(ss)
                    # print(listss)
                    tmp1=listss[i]
                    listss[i]=listss[hangZero]
                    listss[hangZero]=tmp1
                    # print(listss)
                    # print('交换前')
                    # print(mat)
                    mat=mat[listss,:]
                    # print('交换后')
                    # print(mat)

            if mat[i][i]!=0:
                hang=i
                inv=self.RS_inverse(mat[hang][i])

                #inv * 第i行的元素
                for j in range(i,np.size(mat,1)):
                    mat[hang][j]=self.RS_mul(inv,mat[hang][j])

                # print('1')
                # print(mat)

                for k in range(i+1,np.size(mat,0)):

                    diagk=mat[k][i]
                    # print('diagk',diagk)
                    for j in range(i,np.size(mat,1)):
                        mat[k][j]=self.RS_add(mat[k][j],self.RS_mul(mat[hang][j],diagk))

                # print('2')
                # print(mat)

        # 伽罗华域消元后
        # print(mat)

        # 统计矩阵mat秩
        rank=0
        for i in range(np.size(mat,0)):
            if_Allzero=0
            for j in range(np.size(mat,1)):
                if mat[i][j]!=0:
                    if_Allzero=1
                    break

            rank=rank+if_Allzero

        # print(rank)
        return rank


   #二进制转十进制数
    def bin2dec(self,x):
        print(list(str(x)))
        return int('0b'+str(x),2)

    # 十进制转二进制数
    def dec2bin(self,x):
        return int(bin(x).replace('0b', ''))

    # 十进制转二进制数的逆序输出,不够4位数的前面补零
    def dec2binreverselist(self,x):
        # print(f'x:{x},bin(x):{bin(x)}')
        xlist=(list(bin(x).replace('0b', '')))
        # print('xlist',xlist)
        #print('len(xlist)',len(xlist))
        xlist.reverse()
        for i in range(4-len(xlist)):
            xlist.append('0')
        #print('after xlist.reverse',xlist)
        # return int(bin(x).replace('0b', ''))
        return xlist

    #<class 'numpy.ndarray'>
    def coefficients_generate(self,n):
        a=np.random.randint(0,15,n,int)
        # print(f'----coefficients_generate,{a},{type(a)}')
        return a

if __name__ == '__main__':
    RS=RScode()
    # row=5
    # col=5
    # a=[]
    # for i in range(20):
    #     X=[]
    #     for j in range(row):
    #         y=RS.coefficients_generate(col)
    #         X.append(y)
    #     #RS.RS_rank(np.array(X))
    #     a.append(RS.RS_rank(np.array(X)))
    # print(a)
    # print(a.count(5))

    # XX=[[0,11,3,14,12],[11,4,3,2,13],[14,13,9,13,8], [2,13,8,8,3],[ 5,7,14,11,12]]
    XX=[[0,11,3,14,12,0,11,3,14,12],[11,4,3,2,13,11,4,3,2,13],[14,13,9,13,8,14,13,9,13,8]]

    xx=RS.RS_rank(np.array(XX))
    tt=RS.RS_reencode_mat(np.array(XX),10)
    print(f'XX rank:{xx}')

    YY=np.row_stack((XX,tt))
    yy=RS.RS_rank(YY)
    print(f'YY=(XX,tt) rank:{yy}')

    ZZ=np.row_stack((YY,tt))
    zz=RS.RS_rank(ZZ)
    print(f'ZZ=(YY,tt)rank:{zz}')


    # YY=[[11,4,3,2,13],[14,13,9,13,8],[2,13,8,8,3],[5,7,14,11,12],[9,0,13,7,14]]
    # RS.RS_rank(np.array(YY))

    # X = np.array([[5, 2, 3, 4],
    #               [1, 6, 7, 8],
    #               [9, 10, 11, 12]])
    #
    # RS.RS_rank(X)


    # co=RS.coefficients_generate(4)
    # Xnew=np.row_stack((X,co))
    #
    # print(Xnew)
    # RS.RS_rank(Xnew)


    # a=RS.dec2bin(3)
    # print(a)
    # c=RS.dec2binreverselist(4)
    # print(c)
    # a=RS.RS_add(6,2)
    # print(a)
    # mm=RS.RS_mul(10,9)
    # print(mm)
    # rr=RS.RS_inverse(15)
    # print(rr)
    # print(RS.RS_mul(5,0))


