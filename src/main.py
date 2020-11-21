import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lib.pca import PCA

def get_inverse_transform(X, n_components):
    """
    Perform PCA and Compute the inverse transform for the input data.
    Also return the original_x obtained through inverse transformation after PCA.

    Arguments:
        X(numpy.ndarray): The input matrix for which the inverse transform is to be computed
        n_components(int): The number of components 

    """
    pca = PCA(n_components=n_components)
    X_red = pca.transform(X)
    X_inv = pca.inverse_transform(X_red)
    X_inv = X_inv.reshape(-1, 64, 64)
    return X_inv

if __name__ == '__main__':
    
    # Load the images into a matrix
    images = []
    for filename in os.listdir('../data/group10/'):
        image = cv2.imread(f'../data/group10/{filename}', cv2.IMREAD_GRAYSCALE)
        images.append(image)

    print(f'There are {len(images)} images to reconstruct')

    # Convert the images to numpy arrays
    X = np.array(images)
    X = X.reshape(-1, 64**2)
    print('Input dataset has shape ', X.shape)

    ## Plot the inverse transformation of the images for k=4096, our min no. of comps x=3 and for k=2 (<x)

    faces = np.zeros((4, 30, 64, 64))
    faces[0] = images
    faces[1] = get_inverse_transform(X, 4096)
    faces[2] = get_inverse_transform(X, 3)
    faces[3] = get_inverse_transform(X, 2)

    titles = ['Original Faces', 'Reconstructed Faces for k=4096', 'Reconstructed Faces for k=3', 'Reconstructed Faces for k=2']

    # Plot the results
    fig = plt.figure(figsize=(20, 30)) 
    outer = gridspec.GridSpec(4, 1, wspace=0.01, hspace=0.05)

    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(3, 10,
                        subplot_spec=outer[i], wspace=0.01, hspace=0)
        
        title_ax = plt.Subplot(fig, outer[i])
        title_ax.set_title(titles[i])
        title_ax.axis('off')
        fig.add_subplot(title_ax)

        for j in range(30):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(faces[i][j], cmap=plt.cm.bone, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    fig.show()