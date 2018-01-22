__author__ = 'Dustin Davis'
#AST383 HW02
#January 22, 2018

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
     shows both the validity of CLT and the increase in sd of SUM with increase in N
     (reminder to self: if using means of samples, the sd would decrease with sqrt(N))
     scatter of sd of means for 1000 runs each of 10,100,1000,10000 samples
     """

    t10_1 = np.zeros(1000)
    t10_2 = np.zeros(1000)
    t10_3 = np.zeros(1000)
    t10_4 = np.zeros(1000)
    dof = 50

    # std of chi2 = sqrt(2*df), so 10.0 in this case
    chi2_sd = np.sqrt(2*dof)

    for i in range(1000):
        # take the sum of the points for each of 1000 sample runs
        t10_1[i] = np.sum(stats.chi2.rvs(df=dof, size=10))
        t10_2[i] = np.sum(stats.chi2.rvs(df=dof, size=100))
        t10_3[i] = np.sum(stats.chi2.rvs(df=dof, size=1000))
        t10_4[i] = np.sum(stats.chi2.rvs(df=dof, size=10000))

    sample_sd = [np.std(t10_1), np.std(t10_2), np.std(t10_3), np.std(t10_4)]
    clt_sd = [chi2_sd * np.sqrt(10), chi2_sd * np.sqrt(10 ** 2), chi2_sd * np.sqrt(10 ** 3), chi2_sd * np.sqrt(10 ** 4)]
    x = np.arange(1, 5)

    plt.figure()

    plt.title(r'$\chi^2$ Distribution (DoF=50)'
              '\nCLT Predicted $\sigma$ vs sd of Sample Means')
    plt.xlabel(r"$Log_{10}(N)$ (N = Number of Sample Points)")
    plt.ylabel("Standard Deviation of Sample Means")

    plt.plot(x, sample_sd, c='b', label="sd of sample means")
    plt.plot(x, clt_sd, c='r', linestyle=":", label=r'CLT $\sigma \approx \sigma_{\chi^2} \cdot \sqrt{N}$')

    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

    plt.tight_layout()
    #plt.savefig("./out/ddavis_hw02.pdf", bbox_inches="tight")

    plt.show()

if __name__ == '__main__':
    main()