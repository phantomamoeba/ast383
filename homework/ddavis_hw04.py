import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.seterr(divide='ignore')

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

   # summing_term = h * np.log(float(k)/float(n)*h)
    #get rid of 0 height to protect log
    h[h == 0] = 10**-10

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
    :return:
    '''

    #iterate over all possible number of bins (from 2 to len(data))
    #find the minimum aic
    #?? is this a single mode, so once we find a turn over, we are done?
    #   i.e. once aic[x+1] > aix[x], we have found the minimum?

    aic = np.zeros(len(data)) # a little waste, but then the index is the number of bins
    aic += np.inf
    #call to np.histogram (reminder, if bins=<int> is number of bins, else if array, it defines the edges
    for k in range(2,len(data)):
        hist, edges = np.histogram(data,bins=k)
        aic[k] = aic_hist(hist)
     #   print(k,aic[k])

    #get rid of nans
    aic[np.isnan(aic)] = np.inf
    return np.argmin(aic),min(aic)

def main():

    #test, some random data
    N = 4
    data = []
    for i in range(N):
        data.append(stats.cauchy.rvs(size=10**(i+1)))

    for i in range(N):
        print("AIC", min_aic(data[i]))
        hist, edges = np.histogram(data[i],bins='doane')
        print("numpy:",len(hist))
     #   print( data[i])


if __name__ == '__main__':
    main()

