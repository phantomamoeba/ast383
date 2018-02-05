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
    h = np.array(h)  # just to be safe
    k = len(h)
    n = np.sum(h)

   # summing_term = h * np.log(float(k)/float(n)*h)

    return -2 *np.sum(  h * np.log(float(k)/float(n)*h)  ) + 2*(k-1)




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

    aic = np.zeros(len(data))
    #call to np.histogram (reminder, if bins=<int> is number of bins, else if array, it defines the edges
    for k in range(2,len(data)):
        hist, edges = np.histogram(data,bins=k)
        aic[k] = aic_hist(hist)
        print(k,aic[k])

    #get rid of nans
    aic[np.isnan(aic)] = np.inf
    return np.argmin(aic)+2,min(aic)

def main():

    #test, some random data
    data = stats.norm.rvs(size=10)

    print( min_aic(data))


if __name__ == '__main__':
    main()

