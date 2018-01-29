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

    #test for convergence
    # treat entropy as a random variable
    # take means of samples and show the sd of those means approaches zero

    if False:
        sample_entropy = np.empty((num_of_samples,len(N))).tolist() #[[None]*4] #4 lists (one for each N) of the entropies of each sample
        means = np.zeros(4)
        sds = np.zeros(4)


        for i in range(num_of_samples):
            print(i)
            for j in range(len(N)):
                sample = []
                for k in range(N[j]):
                    sample.append([stats.norm.rvs()])


                sample_entropy[i][j] = ee.entropy(sample)

                #bias = true_entropy - ee.entropy(sample)
                #relative_bias = bias/true_entropy
                #print(relative_bias, N[j])


        for j in range(len(N)):
            means[j] = np.mean(sample_entropy[j])
            sds[j] = np.std(sample_entropy[j])

            print(j,means[j],sds[j])
            #hmmm .... small sample Ns seem to converge (10 and 100), the 1000 and 10^4 don't seem to


    #pass in 100 lists each of 10, 100, 1000, 10000 samples
    #get 4 entropies (one for each of 10, 100, 1000, 10000 samples)
    if True:
        sample_rvs = np.empty((len(N),num_of_samples)).tolist() #4x100
        sample_entropy = np.empty((num_of_samples, len(N))).tolist()  # [[None]*4] #4 lists (one for each N) of the entropies of each sample
        means = np.zeros(4)
        sds = np.zeros(4)

        for j in range(len(N)):
            for i in range(num_of_samples):
                sample_rvs[j][i] = stats.norm.rvs(size=N[j])

                # bias = true_entropy - ee.entropy(sample)
                # relative_bias = bias/true_entropy
                # print(relative_bias, N[j])

        entropies = np.zeros(len(N))
        for j in range(len(N)):
            entropies[j] = ee.entropy(sample_rvs[j])

            print(j,entropies[j])

            #means[j] = np.mean(sample_entropy[j])
            #sds[j] = np.std(sample_entropy[j])

            #print(j, means[j], sds[j])
            # hmmm .... small sample Ns seem to converge (10 and 100), the 1000 and 10^4 don't seem to

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