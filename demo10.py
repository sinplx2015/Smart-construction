import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# np.random.seed(1)
def objective(x):
	return x[0]**3 + x[1]**3 + x[2]**3

def decode(bounds, x):
    """
    将01间的随机数转化为对应区间的值 
    """
    for i in range(len(bounds)):
        value = [bounds[i][0] + s * (bounds[i][1] - bounds[i][0]) for s in x]
    return value

def compute_w(t, n_iter, ):
    """
    计算权重w
    """
    return 1.2 - 0.4* t/ n_iter

def compute_vel(t, vel, pos, n_iter, pop_score, ind_score, max_vel):
    """
    计算速度
    """
    v=[]
    for i in range (len(vel)):
        value = compute_w(t, n_iter)*vel[i]+np.random.rand()*2*(pop_score[i]-pos[i])+np.random.rand()*2*(ind_score[i] -pos[i])
        v.append(value)
    
    #超出范围的    
    v[value > max_vel] = max_vel
    v[value < -max_vel] = -max_vel    
    return v

def compute_pos(pos, vel, bounds):
    """
    计算位置
    """
    p = np.sum([pos, vel],axis=0).tolist()   
    for i in range(len(bounds)):
        if p[i] > bounds[i][1]:p[i] = bounds[i][1]
        elif p[i] < bounds[i][0]:p[i] = bounds[i][0]
    return p

def PSO(bounds, n_pop, max_vel, n_iter, isplot):
    randlist = [np.random.random(len(bounds)).tolist() for _ in range(n_pop)]        
    pos = [decode(bounds, p) for p in randlist]             #初始位置  
    scores = [objective(d) for d in pos]                 #初始化适应度，亦为最优适应度
    pop_best_pos = pos[scores.index(min(scores))]        #种群历史最优解
    ind_best_pos = deepcopy(pos)                         #个体历史最优解
    pop_vel = [((2*np.random.rand(len(bounds))-1)*max_vel).tolist() for _ in range(n_pop)]      #初始速度

    if isplot:
        fig = plt.figure()
        subplot = fig.add_subplot(111)

    for iter in range (n_iter):
        for i in range (n_pop):
            vel = compute_vel(i, pop_vel[i], pos[i], n_iter, pop_best_pos, ind_best_pos[i], max_vel)  #更新速度
            current_pos= compute_pos (pos[i], vel, bounds)     #更新位置
            current_score = objective(current_pos)
            
            
            # print(vel)
            if current_score < scores[i]:           #个体最优解
                scores[i] = current_score
                ind_best_pos[i] = current_pos

            if current_score < objective(pop_best_pos):         #种群最优解
                pop_best_pos =  current_pos 
                print("{} epoch,  f{} = {}".format(iter+1,  pop_best_pos, objective(pop_best_pos)))
        
        if isplot:            
            subplot.scatter([iter+1]*n_pop, scores, c = 'r', marker= '.')
    return pop_best_pos, objective(pop_best_pos)

def main():
    #参数设置
    n_pop = 100    #种群规模
    bounds = [[0 ,6.0], [1.0, 7.0], [2.0, 6.0],]     #边界条件
    n_iter = 100    #迭代次数
    max_vel = 2.22    #最大速度    
    isplot = True  #是否绘图
    
    best, score =PSO(bounds, n_pop, max_vel, n_iter, isplot)
    print('Down!\n', 'f(%s) = %f' % (best, score))
    if isplot:
        plt.show()
    
if __name__ == '__main__':
    main()
    