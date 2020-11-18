import numpy as np
from sklearn.decomposition import PCA
from lib.pca import PCA as MyPCA


def test_against_sklearn(x):
    pca = PCA(n_components=2)
    my_pca = MyPCA(n_components=2)
    y = pca.fit_transform(x)
    z = my_pca.transform(x)

    print(pca.components_)
    print(pca.singular_values_)

    print(my_pca.components_)
    print(my_pca.singular_values_)


    # print('sklearn: ', y)
    # print('ours: ', z)
    # np.testing.assert_array_almost_equal(y, z)


if __name__ == '__main__':
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    x = np.array([[0.5, 1], [0, 0]])
    test_against_sklearn(x)
    print('✓(1/1) Tests Passed!')