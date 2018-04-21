__author__ = 'Dustin Davis'
#AST383 HW06
#April 27, 2018
#
#ML Segmentation


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#a few test images from EGS field
img1 = np.load('r214.5d52.5.npy')
img2 = np.load('r215.0020629d52.96877494.npy')
img3 = np.load('r214.985038d52.97236883.npy')
img4 = np.load('r215.0063129d52.97301577.npy')


#plt.figure()
#plt.imshow(img1,origin='lower')
#plt.show()

#plt.imshow(img2,origin='lower')
#plt.show()

plt.imshow(img3,origin='lower',vmin=0, vmax=0.3)
plt.show()

#plt.imshow(img4,origin='lower')
#plt.show()
