# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Read and Display Face Images

# %%
images = []
for filename in os.listdir('../data/group10/'):
    image = cv2.imread(f'../data/group10/{filename}', cv2.IMREAD_GRAYSCALE)
    images.append(image)

print(f'There are {len(images)} images to reconstruct')


# %%
#Setup a figure 10 inches by 10 inches 
fig = plt.figure(figsize=(10,10)) 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 

# Plot the faces, each image is 64 by 64 pixels 
for i in range(len(images)): 
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[]) 
    ax.imshow(images[i], cmap=plt.cm.bone, interpolation='nearest') 

plt.show() 

# %% [markdown]
# # PCA on Faces

# %%
X = np.array(images)
X = X.reshape(-1, 64**2)
print('Input dataset has shape ', X.shape)


# %%
pca = PCA(n_components=3)
X_red = pca.fit_transform(X)
print('Reduced dimensions of input data after PCA is ', X_red.shape)

# %% [markdown]
# # Reconstruct Images through Inverse Transform and Visualize

# %%
#Inverse transform and visualize
X_inv = pca.inverse_transform(X_red)
X_inv = X_inv.reshape(-1, 64, 64)
print('Reconstructed input data to shape ', X_inv.shape)


# %%
#Setup a figure 10 inches by 10 inches 
fig = plt.figure(figsize=(10,10)) 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 

# Plot the faces, each image is 64 by 64 pixels 
for i in range(len(images)): 
    ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[]) 
    ax.imshow(X_inv[i], cmap=plt.cm.bone, interpolation='nearest') 

plt.show() 


# %%


