import numpy as np
from scipy import linalg as LA

def PCA(data, dims_rescaled_data=2):     

    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = LA.eigh(R)
    print(evals, evecs)
    
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :dims_rescaled_data]
    
    return np.dot(evecs.T, data.T).T

def plot_pca(data,):
    from matplotlib import pyplot as plt
    data_resc = PCA(data)
    print(data_resc)
    plt.scatter(data_resc[:, 0], data_resc[:, 1] ,marker='.')
    plt.show()

def main():
    data = np.loadtxt(r'C:\Users\Administrator\\Desktop\\2_Precast_Laminated_Panels.asc', usecols=(0,1,2))
    plot_pca(data)

if __name__ == '__main__':
    main()
