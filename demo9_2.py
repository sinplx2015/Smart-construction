from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt

def decode(n_bits, bitstring):
    """
    字符串解码并转化为对应区间的值
    """   
    substring = bitstring[0:n_bits] 
    return substring.count(0),substring.count(1),substring.count(2)


def selection(pop, scores, k=3):
    """
    选择算子
    """    
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1, p2, r_cross):
    """
    交叉算子，r_cross为交叉概率
    """
    c1, c2 = p1.copy(), p2.copy()       #保证格式相同
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)      #交叉方式为单点交叉
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(bitstring, r_mut):
    """
    变异算子，进行基本位变异操作
    """
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut,isplot):
    
    pop = [randint(0, 3, n_bits*len(bounds)).tolist() for _ in range(n_pop)]        #生成长度为n_bits的随机01比特串
    best, best_eval = 0, objective(decode(n_bits, pop[0]))
        
    if isplot:
        fig = plt.figure()
        subplot = fig.add_subplot(111)

    for gen in range(n_iter):
        decoded = [decode(n_bits, p) for p in pop]      #解码
        scores = [objective(d) for d in decoded]        #计算适应度
        
        if isplot:            
            subplot.scatter([gen+1]*n_pop, scores, c = 'r', marker= '.')

        #选择结果
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print("{} epoch,  f{} = {}".format(gen,  decoded[i], scores[i]))
        selected = [selection(pop, scores) for _ in range(n_pop)]
		
        #生成子代
        children = list()
        for i in range(0, n_pop, 2):			
            p1, p2 = selected[i], selected[i+1]

			#交叉&变异
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)				
                children.append(c)
                
        pop = children      # 淘汰其他
        
    return [best, best_eval]

#目标函数
def objective(x):
    global value
    value = 1
    width = (total_length - 90*x[1] -200*x[2])/(x[1]+x[2]-1+10e-6)
    # print(width)
    if width < 8 or width > 12 :
        value += 1000      #满足缝隙要求
    value  = value - R1*x[2] - R2*x[1]
    
    return value

def main():
    #参数设置
    global total_length, R1, R2
    total_length = 2800     #墙长度 
    R1 = 10     #奖励函数
    R2 = -25    #惩罚函数
    bounds = [[0, None]]     #边界条件, 边界维度与n_pop控制pop维度  
    n_iter = 100  #迭代次数
    n_bits = 16 #比特位 此处使用的最大砖块数，至少取 total_length/90   
    n_pop = 100   #种群规模    
    r_cross = 0.9  #交叉比例    
    r_mut = 0.1 / (float(n_bits) * len(bounds))     #变异概率
    
    #惩罚函数
    
    isplot = True   #是否绘图
    
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, isplot)
    
    decoded = decode( n_bits, best)
    print('Down!\n', 'f(%s) = %f\n' % (decoded, score))
    print(("墙长{}，90型砖{}块，200型砖{}块").format(total_length,decoded[1], decoded[2]))
    if isplot:
        plt.show()
if __name__ == '__main__':
    main()
    

