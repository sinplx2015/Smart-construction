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

def plot_pca(data):
    from matplotlib import pyplot as MPL
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1])
    MPL.show()

def main():
    data = np.loadtxt(r'C:\Users\Administrator\\Desktop\\2_Precast_Laminated_Panels.asc')
    data = data[:,:3]
    plot_pca(data)

if __name__ == '__main__':
    main()
