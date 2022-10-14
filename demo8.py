import math
from copy import deepcopy
from random import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np


class vector3d():
    def __init__(self, x, y, z):
        self.deltaX = x 
        self.deltaY = y
        self.deltaZ = z
        self.direction = [0, 0, 0]
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2 + self.deltaZ ** 2) * 1.0
    
    def vector3d_share(self):
        self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2 + self.deltaZ ** 2) * 1.0
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()) and type(self.deltaZ) == type(list()):
            deltaX, deltaY, deltaZ = self.deltaX, self.deltaY, self.deltaZ
            self.deltaX = deltaX
            self.deltaY = deltaY
            self.deltaZ = deltaZ
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length, self.deltaZ / self.length]
            else:
                self.direction = None
        else:
            # self.direction = [self.deltaX / self.length, self.deltaY / self.length, self.deltaZ / self.length]
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length, self.deltaZ / self.length]
            else:
                self.direction = [0,0,0]
                
    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, deltaZ:{} length:{}, direction:{}'.format(self.deltaX, self.deltaY, self.deltaZ, 
                                                                                       self.length,self.direction)
    
    def __truediv__(self, other):
        return self.__mul__(1.0 / other)
    
    def __sub__(self, other):
        vec = vector3d(self.deltaX, self.deltaY, self.deltaZ)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.deltaZ -= other.deltaZ
        vec.vector3d_share()
        return vec

    def __add__(self, other):
        vec = vector3d(self.deltaX, self.deltaY, self.deltaZ)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.deltaZ += other.deltaZ
        vec.vector3d_share()
        return vec
    
    def __mul__(self, other):
        vec = vector3d(self.deltaX, self.deltaY, self.deltaZ)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.deltaZ *= other
        vec.vector3d_share()
        return vec
    
    
# class obstacal():
#     def __init__(self, x, y, z, r):
#         self.x = x
#         self.y = y
#         self.z = z
    

class APF():
    """
    人工势场法
    """

    def __init__(self, start, goal, obstacles, k_att: float, k_rep: float, rr: float,
                 step_size: float, max_iters: int, goal_threshold: float, isimproved=False):
        """
        :param start: 起点
        :param goal: 终点
        :param obstacles: 障碍物
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param step_size: 步长
        :param max_iters: 迭代次数
        :param goal_threshold: 控制值
        """
        self.start = vector3d(start[0], start[1], start[2])
        self.current_pos = vector3d(start[0], start[1], start[2] )
        self.goal = vector3d(goal[0], goal[1], goal[2])
        self.obstacles = [vector3d(OB[0], OB[1],OB[2]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr 
        self.step_size = step_size
        self.max_iters = max_iters
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.isimproved = isimproved
        self.delta_t = 0.01
        self.previous_pos = self.current_pos

    def attractive(self):
        """
        引力
        """
        att = (self.goal - self.current_pos) * self.k_att  
        return att
        
    def repulsion(self):
        """
        斥力
        """
        rep = vector3d(0, 0, 0) 
        for obstacle in self.obstacles:
            # obstacle = Vector3d(0, 0)
            t_vec = self.current_pos - obstacle
            if (t_vec.length > self.rr):  
                pass
            else:
                rep += vector3d(t_vec.direction[0], t_vec.direction[1],t_vec.direction[2]) * self.k_rep * (
                        1.0 / t_vec.length - 1.0 / self.rr) / (t_vec.length ** 2)
                
        return rep

    def path_plan(self):
        """
        path plan
        :return:
        """
        
        while (self.iters < self.max_iters and (self.current_pos - self.goal).length > self.goal_threashold):
            #备份上两个点
            self.pprevious_pos = deepcopy(self.previous_pos)
            self.previous_pos = deepcopy(self.current_pos)

            #计算当前位置的力
            f_vec = self.attractive() + self.repulsion()
            self.current_pos += vector3d(f_vec.direction[0], f_vec.direction[1], f_vec.direction[2]) * self.step_size
            
            #改进方法，陷入局部最小值随机游走一步
            if (self.current_pos-self.pprevious_pos).length <0.1 and self.isimproved == True:
                self.current_pos += vector3d(random(),random(),random()) * self.step_size *0.5
                self.path.append([self.current_pos.deltaX, self.current_pos.deltaY, self.current_pos.deltaZ])
            else:   #未改进          
                self.path.append([self.current_pos.deltaX, self.current_pos.deltaY, self.current_pos.deltaZ])
            self.iters += 1
            # print (self.current_pos)
            # if True:
            #     plt.scatter(self.previous_pos.deltaX, self.previous_pos.deltaY, self.previous_pos.deltaZ)
                # plt.pause(self.delta_t)
        
        if (self.current_pos - self.goal).length <= self.goal_threashold:
            print("Path plan success")
            self.is_path_plan_success = True      
        
        return self.path

def main():
    # cfg
    isimproved = True       #是否改进
    isplot = True           #是否绘图
    k_att, k_rep = 0.1 , 1000000.0
    rr = 21
    step_size, max_iters, goal_threashold = .2, 10000, .2 
    # step_size_ = 2
    
    #path plan
    start = (100, 0, 100)
    goal = (100, 200, 100)
    obs1 = [[100, 50, i] for i in range(200)]
    obs2 = [[i, 150, 100] for i in range(200)]
    obs = obs1 +obs2
    

    #plot
    if isplot:
        fig = plt.figure(figsize=(200,200))
        subplot = Axes3D(fig)
        
        subplot.set_xlabel('X-distance: mm')
        subplot.set_ylabel('Y-distance: mm')
        subplot.set_zlabel('Z-distance: mm')
        subplot.scatter(start[0], start[1], start[2]) 
        subplot.scatter(goal[0], goal[1], goal[2])
        # 障碍物z方向
        u = np.linspace(0, 2 * np.pi, 50)
        h = np.linspace(0,200,50)
        x1 = 10 * np.outer(np.cos(u), np.ones(len(h)))+100
        y1 = 10 * np.outer(np.sin(u), np.ones(len(h)))+50
        z1 = np.outer(np.ones(len(u)),h)
        # subplot.plot_surface(x1, y1, z1)
        # 障碍物x方向
        y2 = 10 * np.outer(np.cos(u), np.ones(len(h)))+150
        z2 = 10 * np.outer(np.sin(u), np.ones(len(h)))+100
        x2 = np.outer(np.ones(len(u)),h)
        subplot.plot_surface(x1, y1, z1)
        subplot.plot_surface(x2, y2, z2)
        # plt.show()
        
        

        # for OB in obs:
            # subplot.scatter(OB[0], OB[1],OB[2])
        u = np.linspace(0,2*np.pi,20)
        h = np.linspace(0,200,20)
        x =10*np.outer(np.cos(u),np.ones(len(h)))+100
        y = 10*np.outer(np.sin(u),np.ones(len(h)))+50
        z = np.outer(h,np.ones(len(u)))
        subplot.plot_surface(x,y,z)
        
    
    apf = APF(start, goal, obs, k_att, k_rep, rr, step_size, max_iters, goal_threashold, isimproved)
    path = np.array(apf.path_plan())
    if isplot:
        subplot.scatter3D(path[:,0], path[:,1], path[:,2])
        plt.show()
        
    
if __name__ == '__main__':
    main()