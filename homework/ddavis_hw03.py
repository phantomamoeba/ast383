__author__ = 'Dustin Davis'
#AST383 HW03
#January 29, 2018

import sys
sys.path.append('../NPEET')

import entropy_estimators as ee
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def main():
    """

     """

    N = [10,100,1000,10000]
    num_of_samples = 1000
    true_entropy = stats.norm.entropy() #loc=0,scale=1

    #test for convergence
    # treat entropy as a random variable
    # take means of samples and show the sd of those means approaches zero

    #N x num_of_samples
    sample_entropy = np.empty((len(N),num_of_samples)).tolist()
    means = np.zeros(4)
    sds = np.zeros(4)
    biases = np.empty((len(N),num_of_samples))

    for i in range(len(N)):
        print(i)
        for j in range(num_of_samples):
            #reshape so can pass a list of list of floats (one float in each sub-list) to ee.entropy
            sample = np.reshape(stats.norm.rvs(size=N[i]),(N[i],1)).tolist()
            entropy_est = ee.entropy(sample)
            sample_entropy[i][j] = entropy_est

            #get the bias of the entropy estimate for this one sample of size N
            bias = true_entropy - entropy_est
            biases[i][j] = bias
            #relative_bias = bias/true_entropy
            #print(relative_bias, N[j])


    #does the bias converage, use CLT ... expect the std to approach 0.0
    for i in range(len(N)):
        means[i] = np.mean(biases[i]) #means of the biases
        sds[i] = np.std(biases[i])

        print(i,means[i],sds[i])
        #hmmm .... biases don't seem to converge ... std remains essentially constant across num_of_samples

    #plotting
    plt.figure()
    plt.title(r'Relative Bias of Entropy Estimator for Normal Distribution' 
              '\nNum of Samples (%d)' %(num_of_samples))
    plt.xlabel(r"$Log_{10}(N)$ (N = Number of Points per Sample)")
    plt.ylabel("mean of relative bias")

    x = np.arange(1, 5)
    #unexpected ... relative bias actually increases (in abs) with the size of N
    plt.plot(x, means/true_entropy, c='b', label="mean of relative biases")

    #plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

    plt.tight_layout()
    plt.savefig("./out/ddavis_hw03_%d.pdf" %num_of_samples, bbox_inches="tight")

    plt.show()

if __name__ == '__main__':
    main()