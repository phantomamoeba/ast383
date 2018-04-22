__author__ = 'Dustin Davis'
#AST383 HW06
#April 27, 2018
#
#ML Segmentation


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#todo: actual error handling

#assume image is x by y by channels (like r,g,b = 3 or greyscale=1)
def make_image_patches(image,patch_size = 4):
    ''''''

    shape = np.shape(image)
    if len(shape) == 2:
        channels = 1
    elif len(shape) == 3:
        channels = shape[2]
    else:
        print("Unexpected image Dim ", shape)
        return None

    tf_image = tf.convert_to_tensor(image,dtype=tf.float32)
    if channels == 1:
        tf_image = tf_image[tf.newaxis, :, :, tf.newaxis]
    else:
        tf_image = tf_image[tf.newaxis, :, :, :]

    patches = tf.extract_image_patches(images=tf_image,
                                       ksizes=[1, patch_size, patch_size, 1],
                                       strides=[1, patch_size, patch_size, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')

    number_of_patches = patches.shape[1] * patches.shape[2]
    patches = tf.reshape(patches, [number_of_patches, patch_size, patch_size, channels])

    return patches

def flatten_tensor_2D(tensor): #4D tensor
    return tf.reshape(tensor,[tensor.shape[0],(tensor.shape[1]*tensor.shape[2]*tensor.shape[3])])



def main():
    # a few test images from EGS field
    images = []
    images.append(np.load('r214.5d52.5.npy'))
    images.append(np.load('r215.0020629d52.96877494.npy'))
    images.append(np.load('r214.985038d52.97236883.npy'))
    images.append(np.load('r215.0063129d52.97301577.npy'))


    #debug check the images
    if False:
        for i in images:
            plt.imshow(i, origin='lower')
            plt.show()
            plt.close()

    #todo: change this to an input
    image = images[0] #i.e 41x41x1

    patches = make_image_patches(image) #i.e. (100,4,4,1) 100 patches of 4x4x1

    feature_matrix = flatten_tensor_2D(patches) #i.e. from 100x4x4x1 to 100x16



    print("hi")

if __name__ == '__main__':
    main()