import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

np.seterr(divide='ignore')

MIN_FLOAT = sys.float_info.min

def aic_hist(h):
    '''
    AIC for histogram
    :param h: array of bin heights (len (h) == k)
    :return: aic (float) value
    '''

    #todo: error checking
    h = np.array(h).astype(float)  # just to be safe and to protect the log
    k = float(len(h))
    n = np.sum(h)

    #get rid of 0 height to protect log
    h[h == 0] = MIN_FLOAT

    try:
        aic = -2 *np.sum( h*np.log(h*k/n) ) + 2*(k-1)
    except:
        #generally h == 0 so, log is not valid
        aic = np.inf
    return aic




def min_aic(data):
    '''
    Pick the best number of bins by minimizing the AIC
    :param data:
    :return: number of bins and corresponding minimum AIC
    '''

    #iterate over all possible number of bins
    #find the minimum aic

    aic = np.zeros(len(data)) # a little waste, but then the index is the number of bins
    aic += np.inf

    #todo: this is a dumb way to search ... should make better choice on min and max k values
    #call to np.histogram (reminder, if bins=<int> is number of bins, else if array, it defines the edges
    for k in range(2,len(data)):
        hist, edges = np.histogram(data,bins=k)
        aic[k] = aic_hist(hist)
        #print(k,aic[k])

    #get rid of nans
    aic[np.isnan(aic)] = np.inf
    return int(np.argmin(aic)),min(aic)

def main():

    zero_point = 0.0
    N = 4
    data = []
    best_bins = np.zeros(4,dtype=int)
    best_aic = np.zeros(4)
    for i in range(N):
        data.append(stats.cauchy.rvs(loc=zero_point,size=10**(i+1)))
        #data.append(stats.norm.rvs(size=10 ** (i + 1)))

    for i in range(N):
        best_bins[i], best_aic[i] = min_aic(data[i])
        print("AIC: N(%d), Bins(%d), MinAIC(%f)" %(10**(i+1), best_bins[i], best_aic[i]))
        #sanity check with np guess on bins ... no good alogrithm, though, for cauchy distro
        #hist, edges = np.histogram(data[i],bins='doane')
        #print("numpy:",len(hist))

    #plots
    zoom_range = (zero_point-25,zero_point+25)
    fig = plt.figure(figsize=(14, 14))
    fig.subplots_adjust(hspace=.8)
    fig.subplots_adjust(wspace=.5)

    fig.suptitle("Cauchy Distribution Histograms\nwith Bins Set by Minimizing AIC", fontsize=15)

    plt.subplot(221)
    plt.title("N=10 Bins=%d" %(best_bins[0]))
    plt.ylabel('Counts')
    plt.xlabel('X')
    plt.hist(data[0],bins=best_bins[0])#,range=zoom_range) #no need to zoom


    plt.subplot(222)
    plt.title("N=100 Bins=%d\n(zoomed)" %(best_bins[1]))
    plt.ylabel('Counts')
    plt.xlabel('X')
    plt.hist(data[1], bins=best_bins[1],range=zoom_range)


    plt.subplot(223)
    plt.title("N=1000 Bins=%d\n(zoomed)" %(best_bins[2]))
    plt.ylabel('Counts')
    plt.xlabel('X')
    plt.hist(data[2], bins=best_bins[2],range=zoom_range)


    plt.subplot(224)
    plt.title("N=10,000 Bins=%d\n(zoomed)" %(best_bins[3]))
    plt.ylabel('Counts')
    plt.xlabel('X')

    plt.hist(data[3], bins=best_bins[3],range=zoom_range)

    plt.savefig("./out/ddavis_hw04.pdf")
    plt.show()




if __name__ == '__main__':
    main()

