import numpy as np
from copy import deepcopy
import pyvista as pv
np.set_printoptions(suppress=True)

def kmeans(data, k, n_iter):    
    center =  np.array([data[i] for i in np.random.choice(data.shape[0], k, replace= True)])    #初始化中心  3*3*k
    dist = [[np.linalg.norm(a - b) for b in center] for a in data ]     #计算每个点到k个中心的距离   k*n
    label = [dist[i].index(min(dist[i])) for i in range(data.shape[0])]     #根据距离分配标签   1*n
    last_label = []
    epoch = 0
    while label != last_label and epoch < n_iter: 
        last_label = deepcopy(label)
        #选择同一簇的点重新计算中心点
        for j in range(k):
            selected_pts = [data[i] for i in range(data.shape[0]) if label[i] == j] 
            center[j] = np.mean(selected_pts, axis=0)
        dist = [[np.linalg.norm(a - b) for b in center] for a in data ]     #更新距离
        label = [dist[i].index(min(dist[i])) for i in range(data.shape[0])]     #更新标签
        print(f" {epoch + 1} epoch ,\n  Center_points = {center}")
        print(f" num of label = {count_label(label)} \n")
        epoch += 1
    return label

def count_label(label):
    """
    计算标签个数
    """
    unique, counts = np.unique(label, return_counts=True)
    return dict(zip(unique, counts))
    

def draw(data, label, k):
    figure = pv.Plotter()  
    color = ['blue', 'purple', 'green', 'yellow']
    for j in range(k):
        selected_pts = [data[i] for i in range(data.shape[0]) if label[i] == j]
        figure.add_mesh(pv.PolyData(selected_pts), color=color[j])
    figure.set_background('w')
    figure.show()
    return
    
def main():         
    k = 4       
    n_iter = 25    
    data = np.loadtxt(r"C:\Users\Administrator\Desktop\four steel members.xyz",usecols=(0,1,2))
    label = kmeans(data, k, n_iter)
    draw(data, label, k)
    
if __name__ == "__main__":
    
    main()