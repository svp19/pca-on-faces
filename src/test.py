import numpy as np
from sklearn.decomposition import PCA
from lib.pca import PCA as MyPCA


def test_against_sklearn(x, k=3, verbose=False):
    pca = PCA(n_components=k)
    my_pca = MyPCA(n_components=k)
    y = pca.fit_transform(x)
    z = my_pca.transform(x)

    if verbose:
        print('='*37, 'TESTING AGAINST SKLEARN', '='*38)
        print('SKLEARN COMPONENTS\n', pca.components_)
        print('SKLEARN SINGULAR VALUES\n', pca.singular_values_)
        print('PCA COMPONENTS\n', my_pca.components_)
        print('PCA SINGULAR VALUES\n', my_pca.singular_values_)
        print('SKLEARN OUTPUT: \n', y)
        print('PCA OUTPUT: \n', z)
        print('-'*100, '\n')

    np.testing.assert_array_almost_equal(np.abs(y), np.abs(z))


def test_reconstruction(x, k=3, verbose=False):
    my_pca = MyPCA(n_components=k)
    y = my_pca.transform(x)
    z = my_pca.inverse_transform(y)
    
    if verbose:
        print('='*37, 'TESTING RECONSTRUCTION', '='*38)
        print('ORIGINAL \n', x)        
        print('RECONSTRUCTED \n', z)
        print('-'*100, '\n')
    np.testing.assert_array_almost_equal(x, z)


if __name__ == '__main__':

    x = np.arange(0, 25).reshape(5, 5)
    # x = np.random.rand(5, 5)
    
    test_against_sklearn(x, verbose=True)
    test_reconstruction(x, verbose=True)
    
    print('✓ (2/2) Tests Passed!')
