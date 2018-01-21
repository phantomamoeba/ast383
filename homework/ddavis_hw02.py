__author__ = 'Dustin Davis'
#AST383 HW02
#January 22, 2018

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

#outfile = './out/ddavis_hw02.pdf'

#fixed seed just so runs will be repeatable
#np.random.seed(1138)


#this or 4 plots (one for each of the #samples) with the sample sd vs sd/sqrt(N))?

def method1():
    """
    scatter of sample sd for 1000 runs each of 10,100,1000,10000 samples
    :return:
    """
    x = np.arange(0, 1000)
    t10_1 = np.zeros(1000)
    t10_2 = np.zeros(1000)
    t10_3 = np.zeros(1000)
    t10_4 = np.zeros(1000)
    dof = 50

    # std of chi2 = sqrt(2*df), so 10.0 in this case
    clt_sd = 10.0

    for i in x:
        t10_1[i] = np.std(stats.chi2.rvs(df=dof, size=10))
        t10_2[i] = np.std(stats.chi2.rvs(df=dof, size=100))
        t10_3[i] = np.std(stats.chi2.rvs(df=dof, size=1000))
        t10_4[i] = np.std(stats.chi2.rvs(df=dof, size=10000))

    plt.figure()

    plt.title(r'$\chi^2$ Distribution (DoF=50) CLT $\sigma$ vs Sample sd ')
    plt.xlabel("Sample Run Number")
    plt.ylabel("Standard Deviation")
    plt.xlim(-5, 1005)  # just to give the points a little room

    plt.scatter(x, t10_1, c='r', s=5, marker="^", label=r"$10^1$ samples")
    plt.scatter(x, t10_2, c='g', s=5, marker="s", label=r"$10^2$ samples")
    plt.scatter(x, t10_3, c='b', s=5, marker="v", label=r"$10^3$ samples")
    plt.scatter(x, t10_4, c='k', s=5, marker="o", label=r"$10^4$ samples")
    plt.axhline(y=clt_sd, c='y', label=r"CLT prediction (N $\rightarrow \infty$)")

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0)

    plt.tight_layout()
    plt.savefig("./out/ddavis_hw02_sd_of_samples.pdf", bbox_inches="tight")

    #plt.show()


def method2():
    """
    scatter of sd of means for 1000 runs each of 10,100,1000,10000 samples
    :return:
    """

    t10_1 = np.zeros(1000)
    t10_2 = np.zeros(1000)
    t10_3 = np.zeros(1000)
    t10_4 = np.zeros(1000)
    dof = 50

    # std of chi2 = sqrt(2*df), so 10.0 in this case
    clt_sd = 10.0

    for i in range(1000):
        #take the mean of the points for each of 1000 sample runs
        t10_1[i] = np.mean(stats.chi2.rvs(df=dof, size=10))
        t10_2[i] = np.mean(stats.chi2.rvs(df=dof, size=100))
        t10_3[i] = np.mean(stats.chi2.rvs(df=dof, size=1000))
        t10_4[i] = np.mean(stats.chi2.rvs(df=dof, size=10000))

    sds = [np.std(t10_1),np.std(t10_2),np.std(t10_3),np.std(t10_4)]
    x = np.arange(0,4)

    plt.figure()

    plt.title(r'$\chi^2$ Distribution (DoF=50) CLT $\sigma$ vs Sample sd' '\n of sd of means')
    plt.xlabel("Number of Samples (Points)")
    plt.ylabel("Standard Deviation of Means")

    plt.plot(x, sds)

    plt.tight_layout()
    #plt.savefig("./out/ddavis_hw02_sd_of_means.pdf", bbox_inches="tight")

    plt.show()

def method3():
    """
    scatter of sd of sums? for 1000 runs each of 10,100,1000,10000 samples
    :return:
    """

    t10_1 = np.zeros(1000)
    t10_2 = np.zeros(1000)
    t10_3 = np.zeros(1000)
    t10_4 = np.zeros(1000)
    dof = 50

    # std of chi2 = sqrt(2*df), so 10.0 in this case
    clt_sd = 10.0

    for i in range(1000):
        #take the sum of the points for each of 1000 sample runs
        t10_1[i] = np.sum(stats.chi2.rvs(df=dof, size=10))
        t10_2[i] = np.sum(stats.chi2.rvs(df=dof, size=100))
        t10_3[i] = np.sum(stats.chi2.rvs(df=dof, size=1000))
        t10_4[i] = np.sum(stats.chi2.rvs(df=dof, size=10000))

    sds = [np.std(t10_1),np.std(t10_2),np.std(t10_3),np.std(t10_4)]
    x = np.arange(0,4)

    plt.figure()

    plt.title(r'$\chi^2$ Distribution (DoF=50) CLT $\sigma$ vs Sample sd' '\n of sd of sums')
    plt.xlabel("Number of Samples (Points)")
    plt.ylabel("Standard Deviation of Sums")

    plt.plot(x, sds)

    plt.tight_layout()
    #plt.savefig("./out/ddavis_hw02_sd_of_sums.pdf", bbox_inches="tight")

    plt.show()

def main():


#or a single 1000 sample rvs  and pull 10,100,1000,10000 sub samples taking the sd of the means of each?
# ... that does not make sense as the 1000 sub-sample is the population, and 10000 duplicates

#not sure what is meant by plotting sd as a funtion of log_10
# ... log_10(N) for each of the four sets is a constant (1,2,3,4)
# ... could put on a log scale, but that seems pointless


 #   method1()
  #  method2()
    method3()


if __name__ == '__main__':
    main()