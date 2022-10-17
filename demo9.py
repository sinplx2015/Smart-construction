from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt

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
    
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]        #生成长度为n_bits的随机01比特串
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
        
    if isplot:
        fig = plt.figure()
        subplot = fig.add_subplot(111)

    for gen in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in pop]      #解码
        # print(decoded)
        scores = [objective(d) for d in decoded]        #计算适应度
        
        if isplot:            
            subplot.scatter([gen+1]*n_pop, scores, c = 'r', marker= '.')

        #选择结果
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print("{} epoch,  f{} = {}".format(gen,  decoded[i], scores[i]))
        # print(scores)
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
	return x[0]**3 + x[1]**3 + x[2]**3

def main():
    #参数设置
    bounds = [[0, 9.0], [1.0, 7.0], [2.0, 6.0],]     #边界条件, 边界维度与n_pop控制pop维度  
    n_iter = 100   #迭代次数
    n_bits = 16   #比特位    
    n_pop = 100   #种群规模    
    r_cross = 0.9  #交叉比例    
    r_mut = 0.1 / (float(n_bits) * len(bounds))     #变异概率
    isplot = True   #是否绘图
    
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, isplot)
    
    decoded = decode(bounds, n_bits, best)
    print('Down!', 'f(%s) = %f' % (decoded, score))
    if isplot:
        plt.show()
if __name__ == '__main__':
    main()
    