__author__ = 'Dustin Davis'
#AST383 HW03
#January 26, 2018

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
    num_of_samples = 100
    #samples = [[]*num_of_samples]*len(N) #np.empty((num_of_samples,len(N))).tolist() #2d array

    true_entropy = stats.norm.entropy() #loc=0,scale=1


    for i in range(num_of_samples):
        for j in range(len(N)):
            sample = []
            for k in range(N[j]):
                sample.append([stats.norm.rvs()])
            bias = true_entropy - ee.entropy(sample)
            relative_bias = bias/true_entropy
            print(relative_bias, N[j])


    exit(0)

    #plotting
    plt.figure()
    plt.title(r'')
    plt.xlabel(r"")
    plt.ylabel("")

    #plt.plot(x, sample_sd, c='b', label="sd of sample sums")
    #plt.plot(x, clt_sd, c='r', linestyle=":", label=r'CLT $\sigma \approx \sigma_{\chi^2} \cdot \sqrt{N}$')

    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

    plt.tight_layout()
    plt.savefig("./out/ddavis_hw03.pdf", bbox_inches="tight")

    #plt.show()

if __name__ == '__main__':
    main()