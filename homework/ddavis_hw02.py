__author__ = 'Dustin Davis'
#AST383 HW02
#January 22, 2018

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

outfile = './out/ddavis_hw02.pdf'

#fixed seed just so runs will be repeatable
#np.random.seed(1138)

def main():

    x = np.arange(0,1000)
    t10_1 = np.zeros(1000)
    t10_2 = np.zeros(1000)
    t10_3 = np.zeros(1000)
    t10_4 = np.zeros(1000)
    dof = 50

    # std of chi2 = sqrt(2*df), so 10.0 in this case
    clt_sd = 10.0

    for i in x:
        t10_1[i] = np.std(stats.chi2.rvs(df=dof,size=10))
        t10_2[i] = np.std(stats.chi2.rvs(df=dof,size=100))
        t10_3[i] = np.std(stats.chi2.rvs(df=dof,size=1000))
        t10_4[i] = np.std(stats.chi2.rvs(df=dof,size=10000))


    plt.figure()

    plt.title(r'$\chi^2$ Distribution (DoF=50) CLT vs Sample $\sigma$ ')
    plt.xlabel("Sample Run Number")
    plt.ylabel("Standard Deviation")
    plt.xlim(-5,1005) #just to give the points a little room

    plt.scatter(x, t10_1, c='r',s=5,marker="^", label=r"$10^1$ samples")
    plt.scatter(x, t10_2, c='g',s=5,marker="s", label=r"$10^2$ samples")
    plt.scatter(x, t10_3, c='b',s=5,marker="v", label=r"$10^3$ samples")
    plt.scatter(x, t10_4, c='k',s=5,marker="o", label=r"$10^4$ samples")
    plt.axhline(y=clt_sd, c='y',    label="CLT prediction")

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight")

    #plt.show()


if __name__ == '__main__':
    main()