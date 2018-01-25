__author__ = 'Dustin Davis'
#AST383 HW03
#January 26, 2018

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def main():
    """

     """

    samples = [None]*100 #list of arrays
    means = [None]*100

    for i in range(100):
        samples[i] = stats.norm.rvs(size=100)
        means[i] = np.mean(samples[i])
        #todo .... step 3 and on




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