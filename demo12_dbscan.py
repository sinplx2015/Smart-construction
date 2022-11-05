import numpy as np
from copy import deepcopy
import pyvista as pv
np.set_printoptions(suppress=True)
import open3d as o3d

def dbscan(data, r, minPts):
    C = []
    m = data.shape[0]
    N =[[] for _ in range(m)]
    #确定邻域与核心对象
    for j in range(m):
        for i in range(m):
            if np.linalg.norm(data[i]-data[j])< r:
                N[j].append(list(data[i]))
        Omega = [list(data[i]) for i in range(m) if len(N[i]) >= minPts]
    #初始化
    k = 0
    Gama = deepcopy(data).tolist()
    while len(Omega) != 0:
        Gama_old = deepcopy (Gama)      #记录未访问样本
        o = Omega.pop()     #随机取核心对象
        Q = [o]     
        Gama.remove(o)
        #找出密度可达样本生成聚类簇
        while Q:
            q = Q[0]
            Q.remove(q)
            index = np.argwhere(data==q)[0,0]
            if len(N[index]) >= minPts:
                delta = [x for x in Gama if x in N[index]]
                Q += delta
                Gama = [x for x in Gama if x not in delta]
        k += 1
        C_k = [x for x in Gama_old if x not in Gama]
        C.append(C_k)
        Omega = [x for x in Omega if x not in C_k]
    return C, k

def draw(C, k):
    figure = pv.Plotter()  
    color = ['blue', 'purple', 'green', 'yellow']
    for j in range(k):
        selected_pts = C[j]
        figure.add_mesh(pv.PolyData(selected_pts), color=color[j])
    figure.set_background('w')
    figure.show()
    return
    

def main():         
    # 0.2 30 1000
    r = 0.3        #半径
    minPts= 300     #最小点
    
    pcd = o3d.io.read_point_cloud(r"C:\Users\Administrator\Desktop\four steel members.xyz")
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 100)
    data = np.array(pcd_new.points)
    C, k = dbscan(data, r, minPts)
    draw(C, k)
    
    
if __name__ == "__main__":
    
    main()