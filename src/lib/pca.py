''' PCA Implementation in NumPy'''
import numpy as np


class PCA():
    """
    Implements Principal Component Analysis using NumPy.

    Arguments:
        n_components (int): The number of components (k) used to reduce 
            the dimension of the input data.
    Example:
        >>> from lib.pca import PCA
        >>> x = np.array([[1, 2], [3, 4]])
        >>> pca  = PCA(n_components=1)
        >>> y = pca.transform(x)
    """

    def __init__(self, n_components=1):
        self.k = n_components


    def transform(self, x):
        ''' 
        Transforms input data using PCA Algorithm.
        
        Algorithm: 
            0. Get number of samples(N) and feature dimension(D)
            1. Center the input data
            2. Get scaled co-variance between features (not N samples)
            3. Eigen Decomposition of Co-variance matrix
            4. Sort eigen-vectors in descending order of eigen-values
            5. Project input using first 'k' eigen-vectors
        
        Arguments:
            x (numpy.ndarray): Input data to be transformed
        '''

        N, D = x.shape
        x_centered = x - x.mean(axis=0)         #1.
        x_cov = np.cov(x_centered.T) / (N)      #2.
        eig_val, eig_vec = np.linalg.eig(x_cov) #3.

        indices = eig_val.argsort()[::-1]       #4.
        eig_val = eig_val[indices]
        eig_vec = eig_vec[:, indices]
        self.components_, self.singular_values_ = eig_vec[:, :self.k].T, eig_val[:self.k]

        x_pca = x_centered.dot(eig_vec[:, :self.k]) #5.
        return x_pca


    def inverse_transform(self, x):
        pass