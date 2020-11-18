import numpy as np
from sklearn.decomposition import PCA
from lib.pca import PCA as MyPCA


def test_against_sklearn(x, k=3):
    pca = PCA(n_components=k)
    my_pca = MyPCA(n_components=k)
    y = pca.fit_transform(x)
    z = my_pca.transform(x)

    print('SKLEARN COMPONENTS\n', pca.components_)
    print('SKLEARN SINGULAR VALUES\n', pca.singular_values_)
    print('PCA COMPONENTS\n', my_pca.components_)
    print('PCA SINGULAR VALUES\n', my_pca.singular_values_)
    print('sklearn: \n', y)
    print('ours: \n', z)

    if np.sign(y[0,0]) == np.sign(z[0,0]):
        np.testing.assert_array_almost_equal(y, z)
    else:
        np.testing.assert_array_almost_equal(y, -z)

if __name__ == '__main__':
    x = np.arange(0, 16).reshape(4, 4)
    test_against_sklearn(x)
    print('✓(1/1) Tests Passed!')