__author__ = 'Dustin Davis'
#AST383 HW06
#April 27, 2018
#
#ML Segmentation


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
#from sklearn.cluster import AgglomerativeClustering as agc
#from sklearn.cluster import SpectralClustering as spc


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


def agg_clustering(X, connectivity=None, title="", num_clusters=3, linkage='ward'):
    model = cluster.AgglomerativeClustering(linkage=linkage,
                    connectivity=connectivity, n_clusters=num_clusters)
    model.fit(X)

    return model.labels_





def segmentation_map(image,patch_size=4,num_clusters=None):

    patches = make_image_patches(image, patch_size=4)  # i.e. (100,4,4,1) 100 patches of 4x4x1
    feature_tensor = flatten_tensor_2D(patches)  # i.e. from 100x4x4x1 to 100x16

    with tf.Session() as sess:
        feature_matrix = sess.run(feature_tensor)

    labels = agg_clustering(feature_matrix)

    #now map labels back onto the (rows) of the features and onto the image
    map = np.zeros(np.shape(image))
    map[:,:] = -1

    #turn 100 1D lables back into 100x4x4x1 (each label becomes 16 lables for 4x4x1)
    #so apply to map

    #could do this as a loop (not strictly efficient, but makes sense)
    #(could try tf.space_to_depth and some .reshape calls
    #need the row strides and column strides ... use patch_size

    #note ... this is a bit backward ...
    row_stride = image.shape[0]//patch_size#how many patches per row
    for r in range(image.shape[0]//patch_size * patch_size):
        for c in range(image.shape[1]//patch_size * patch_size):
            idx = (r//patch_size)*row_stride + (c//patch_size)
            if idx < len(labels):
                map[r][c] = labels[idx]
            else:
                map[r][c] = -1


    return map



def main():
    # a few test images from EGS field
    images = []
    images.append(np.load('r214.5d52.5.npy'))
    images.append(np.load('r215.0020629d52.96877494.npy'))
    images.append(np.load('r214.985038d52.97236883.npy'))
    images.append(np.load('r215.0063129d52.97301577.npy'))


    #debug check the images
    # for i in images:
    #     plt.imshow(i, origin='lower')
    #     plt.show()
    #     plt.close()

    #todo: change this to an input
    image = images[0] #i.e 41x41x1

    map = segmentation_map(image,patch_size=4)

    plt.imshow(map, origin='lower')
    plt.show()
    plt.close()


    print("hi")

if __name__ == '__main__':
    main()