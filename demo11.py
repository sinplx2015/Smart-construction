from cmath import exp, log
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def decode(bounds, n_bits, bitstring):
    """
    字符串解码并转化为对应区间的值
    """
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
        
    return decoded

def crossover(p1, p2, r_cross):
    """
    交叉算子，r_cross为交叉概率
    """
    c1, c2 = p1.copy(), p2.copy()       #保证格式相同
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1)-2)      #交叉方式为单点交叉
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(bitstring, r_mut):
    """
    变异算子，进行基本位变异操作
    """
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]

def NDsorts(R, decoded):
    """
    非支配排序
    """
    #初始化
    value = [[objective1(d), objective2(d).real, objective3(d)] for d in decoded]   #得到函数值，f2去掉虚部
    S=[[] for _ in range(len(R))]
    res =[[] for _ in range(len(R))]
    bit =[[] for _ in range(len(R))]
    for p in range(len(R)): 
        S[p] = []        
        np = 0
        
        #寻找pq支配关系
        for q in range(len(R)):  
            if  (value[p][0] >= value[q][0] and value[p][1] <= value[q][1] and value[p][2] >= value[q][2]) and (
                value[p][0] > value[q][0] or value[p][1] < value[q][1] or value[p][2] > value[q][2]): 
                np += 1
        #按照被支配次数从小到大排序
        for i in range(len(R)):
            if np == i and p not in res[i]:                    
                    res[i].append(decoded[p])
                    bit[i].append(p)
    return res, bit
    
    
def NSGA2(bounds, n_bits, n_iter, n_pop, r_cross, r_mut, isplot): 
    pop = [np.random.randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]        #生成长度为n_bits的随机01比特串
    front =[]
    solution =[]    
    if isplot:
        plt.ion()
        fig = plt.figure()
        subplot = Axes3D(fig)        
        subplot.set_xlabel('f1')
        subplot.set_ylabel('f2')
        subplot.set_zlabel('f3')
        
    for iter in range(n_iter):
        #交叉变异得到新种群R
        Q = []
        for i in range(0, n_pop, 2):			
            p1, p2 = pop[i], pop[i+1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)				
                Q.append(c)                
        R = pop + Q  
        
        #非支配排序
        decoded = [decode(bounds, n_bits, p) for p in R]
        res,index = NDsorts(R, decoded)  
        
        #，由于较难出现两个个体处于同一分类子集，不考虑聚集距离，直接选取排序较小的作为下一代        
        index=sum(index,[])
        pop = [R[i] for i in index[:n_pop]]
        decoded = [decode(bounds, n_bits, p) for p in pop]
        value = [[objective1(d), objective2(d).real, objective3(d)] for d in decoded]
        solution.append(decoded)
        front.append(value)
        
        if isplot:
            value = np.array(value)
            subplot.scatter(value[:,0], value[:,1], value[:,2], marker ='.')
        
        
    print("Down! Totally {} epoch,  Pareto_front = {}".format(iter+1, front[iter]))
    # print("Result = {}".format(solution[-1]))
    if isplot:
            fig = plt.figure()
            subplot = Axes3D(fig)        
            subplot.set_xlabel('f1')
            subplot.set_ylabel('f2')
            subplot.set_zlabel('f3')
            subplot.scatter(value[:,0], value[:,1], value[:,2], marker ='.')
            plt.ioff()
    return front

#目标函数
def objective1(x):
    return 100*x[0] + 1.5*x[1] + 25*x[2]

def objective2(x):
	return 0.8*(x[0]/2e3)**1.5 + 10*log(x[1]/4e5) + exp(x[2]/5e3)/3

def objective3(x):
    t1 = (x[0]-2e3)/8e3
    t2 = (x[1]-4e5)/(2e6-4e5)
    t3 = (x[2]-5e3)/(2e5-5e3)
    return t1 + 2*t2 + 0.4*t3 + 1.5*t1*t2 + (0.4*t3-t1*t2)**2

def main():
    #参数设置
    bounds = [[2e3, 1e4], [4e5, 2e6], [5e3, 2e4],]     #边界条件, 边界维度与n_pop控制pop维度  
    n_iter = 20 #迭代次数
    n_bits = 16   #比特位    
    n_pop = 10 #种群规模    
    r_cross = 0.9  #交叉比例    
    r_mut = 0.1      #变异概率
    isplot = True   #是否绘图

    try:
        front = NSGA2(bounds, n_bits, n_iter, n_pop, r_cross, r_mut, isplot)
    finally:
        pass

    if isplot:
        plt.show()
if __name__ == '__main__':
    
    main()
    
