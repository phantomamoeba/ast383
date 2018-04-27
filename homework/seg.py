__author__ = 'Dustin Davis'
#AST383 HW06
#April 27, 2018
#
#ML Segmentation


import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import hdbscan
#from sklearn.cluster import AgglomerativeClustering as agc
#from sklearn.cluster import SpectralClustering as spc
from astropy.visualization import ZScaleInterval

#todo: actual error handling


def get_vrange(vals, scale=1.0, contrast=1.0):
    vmin = None
    vmax = None
    if scale == 0:
        scale = 1.0

    try:
        zscale = ZScaleInterval(contrast=1.0, krej=2.5)  # nsamples=len(vals)
        vmin, vmax = zscale.get_limits(values=vals)
        vmin = vmin / scale
        vmax = vmax / scale
    except:
        pass

    return vmin, vmax

#
# def plot(image, map, filename=None):
#     plt.subplot(131)
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     cmap = plt.get_cmap('gray_r')
#     vmin, vmax = get_vrange(image)
#     plt.title("Gray-R, Z-Scale")
#     plt.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
#
#     plt.subplot(132)
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     plt.title("No scaling")
#     plt.imshow(image, origin='lower')  # ,cmap=cmap)
#
#     plt.subplot(133)
#     plt.gca().set_xticks([])
#     plt.gca().set_yticks([])
#     plt.title("Segmentation Map")
#     plt.imshow(map, origin='lower')
#
#     if filename is not None:
#         plt.savefig(filename)
#     else:
#         plt.show()
#     plt.close()
#


def plot_dict(image, maps_dict, args):

    rows = 1 + int(np.ceil(len(maps_dict)/2))
    cols = 2

    plt.figure(figsize=(8,3*rows))
    plt.suptitle("Patch Size(%d), #Clusters(%d), Min Cluster Size(%d)" % (args.patch,args.clusters,args.size))
    plt.subplots_adjust(top=0.85)

    plt.subplot(rows,cols,1)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cmap = plt.get_cmap('gray_r')
    vmin, vmax = get_vrange(image)
    plt.title("Gray-R, Z-Scale")
    plt.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    plt.subplot(rows,cols,2)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title("No scaling")
    plt.imshow(image, origin='lower')  # ,cmap=cmap)

    index = 2

    for k in maps_dict.keys():
        map = maps_dict[k]
        index += 1
        plt.subplot(rows,cols,index)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title(k)
        plt.imshow(map, origin='lower')

    plt.subplots_adjust(hspace=0.2)
    #plt.subplots_adjust(top=1.05)
 #   plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output)
    else:
        plt.show()
    plt.close()



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


def agg_clustering(X, connectivity=None, num_clusters=3, linkage='ward'):
    model = cluster.AgglomerativeClustering(linkage=linkage,
                    connectivity=connectivity, n_clusters=num_clusters)
    model.fit(X)

    return model.labels_


def spec_clustering(X,num_clusters=3):
    model = cluster.SpectralClustering(n_clusters=num_clusters)
    model.fit(X)

    return model.labels_


#
# def segmentation_map(image,patch_size=4,num_clusters=None):
#
#     patches = make_image_patches(image, patch_size=4)  # i.e. (100,4,4,1) 100 patches of 4x4x1
#     feature_tensor = flatten_tensor_2D(patches)  # i.e. from 100x4x4x1 to 100x16
#
#     with tf.Session() as sess:
#         feature_matrix = sess.run(feature_tensor)
#
#     #todo: switch between different clustering methods
#     #todo: or run them all and return a list of maps
#     labels = agg_clustering(feature_matrix)
#
#     #now map labels back onto the (rows) of the features and onto the image
#     map = np.full(np.shape(image),-1)
#
#     #turn 100 1D lables back into 100x4x4x1 (each label becomes 16 lables for 4x4x1)
#     #so apply to map
#
#     #could do this as a loop (not strictly efficient, but makes sense)
#     #(could try tf.space_to_depth and some .reshape calls
#     #need the row strides and column strides ... use patch_size
#
#     row_stride = image.shape[0]//patch_size#how many patches per row
#     for r in range(image.shape[0]//patch_size * patch_size):
#         for c in range(image.shape[1]//patch_size * patch_size):
#             idx = (r//patch_size)*row_stride + (c//patch_size)
#             if idx < len(labels):
#                 map[r][c] = labels[idx]
#             else:
#                 map[r][c] = -1
#
#
#     return map
#



def segmentation_maps(image,patch_size=4,num_clusters=3,min_size=5):

    maps_dict = {} #use the same keys (could do this instead as a composite dict)
    labels_dict = {}
    print("Making patches ...")
    patches = make_image_patches(image, patch_size=patch_size)  # i.e. (100,4,4,1) 100 patches of 4x4x1
    feature_tensor = flatten_tensor_2D(patches)  # i.e. from 100x4x4x1 to 100x16

    with tf.Session() as sess:
        feature_matrix = sess.run(feature_tensor)

    #todo: add time probes

    #Agglomerative
    print("Agglomerative ...")
    labels_dict['Agglomerative'] = agg_clustering(feature_matrix,num_clusters=num_clusters)
    maps_dict['Agglomerative'] = np.full(np.shape(image),-1)

    #Spectral
    print("Spectral ...")
    labels_dict['Spectral'] = spec_clustering(feature_matrix,num_clusters=num_clusters)
    maps_dict['Spectral'] = np.full(np.shape(image), -1)

    #HDBScan
    print("HDBscan ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
    labels_dict['HDBScan'] = clusterer.fit_predict(feature_matrix)
    maps_dict['HDBScan'] = np.full(np.shape(image), -1)


    #turn 100 1D lables back into 100x4x4x1 (each label becomes 16 lables for 4x4x1)
    #so apply to map

    #could do this as a loop (not strictly efficient, but makes sense)
    #(could try tf.space_to_depth and some .reshape calls
    #need the row strides and column strides ... use patch_size

    print("Constructing maps ...")
    for k in maps_dict.keys():
        labels = labels_dict[k]
        map = maps_dict[k]

        row_stride = image.shape[0]//patch_size#how many patches per row
        for r in range(image.shape[0]//patch_size * patch_size):
            for c in range(image.shape[1]//patch_size * patch_size):
                idx = (r//patch_size)*row_stride + (c//patch_size)
                if idx < len(labels):
                    map[r][c] = labels[idx]
                else:
                    map[r][c] = -1


    return maps_dict




def main():
    desc = "AST381 HW#06 - Clustering for segmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help='Input .npy numpy file', required=True)
    parser.add_argument('-o', '--output',help='Output png summary file', required=False)
    parser.add_argument('-p', '--patch', help='(Square) Patch size', type=int, default=4, required=False)
    parser.add_argument('-c', '--clusters', help='Number of clusters', type=int, default=3, required=False)
    parser.add_argument('-s', '--size', help='Minimum cluster size', type=int, default=5, required=False)

    args = parser.parse_args()

    # #old testing
    # # a few test images from EGS field
    # images = []
    # #2.5 sqr arcsec (normal HETDEX catalog cutout error window)
    # images.append(np.load('r214.5d52.5.npy'))
    # images.append(np.load('r215.0020629d52.96877494.npy'))
    # images.append(np.load('r214.985038d52.97236883.npy'))
    # images.append(np.load('r215.0063129d52.97301577.npy'))
    #
    # images.append(np.load('r214.985d52.972.npy')) #10 sq arcsec
    #
    #
    # #debug check the images
    # # for i in images:
    # #     plt.imshow(i, origin='lower')
    # #     plt.show()
    # #     plt.close()
    #
    # #todo: change this to an input
    # image = images[4] #i.e 41x41x1
    # map = segmentation_map(image,patch_size=4)
    # plot(image,map)#,"testfig.png")

    image = np.load(args.input)


    maps_dict = segmentation_maps(image,patch_size=args.patch,num_clusters=args.clusters,min_size=args.size)

    plot_dict(image,maps_dict,args)



if __name__ == '__main__':
    main()