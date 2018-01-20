__author__ = 'Dustin Davis'
#AST383 HW01
#January 19, 2018

#examples: python hw01.py --dist norm --rvs 0 size=1000 --pdf 0
#          python hw01.py --dist gamma --rvs 1 size=1000 --pdf 1


#todo: to make 'real' should implement:
# proper error control
# help (list all possible distributions from stats)
# command line options for range (minx,maxx,stepsize)
# command line option for function (distribution) parameters
# command line option to set output name
# optional egs random or fixed seed
# optional plot statistics (if appropriate: mean, median, mode, variance, std, etc)
# option to make x axis symmetric
# option to show plot

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

outfile = './out/ddavis_hw01.pdf'

#fixed seed just so runs will be repeatable
#np.random.seed(1138)

def plot_pdf(dist_name,rvs_args=None,pdf_args=None):

    #samples = dist.rvs(*dist_args,size=1000)
    cmd = 'stats.' + dist_name + '.rvs('
    if (rvs_args is not None) and (len(rvs_args) > 0):
        cmd += ','.join(rvs_args) + ')'
    else:
        cmd += ')'
    samples = eval(cmd)

    xmin = np.min(samples)
    xmax = np.max(samples)
    x = np.linspace(xmin, xmax, 1000)

    cmd = 'stats.' + dist_name + '.pdf(x'
    if (pdf_args is not None) and (len(pdf_args) > 0):
        cmd += ',' + ','.join(pdf_args) + ')'
    else:
        cmd += ')'
    pdf = eval(cmd)

    plt.figure()
    plt.title("%s Distribution\nhistogram (30 fixed bins from 1000 samples) and pdf" % dist_name)
    plt.xlim(xmin,xmax)
    plt.ylabel("normed")
    plt.xlabel("x")

    #note: plt.hist uses numpy.histogram, but can call it explicitly if needed
    #e.g. values, edges = np.histogram(samples,bins=30,normed=True)
    plt.hist(samples, 30, normed=True, facecolor='r',label="histogram")
    plt.plot(x, pdf,c='b',label='pdf')

    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0)

    plt.savefig(outfile)
    print("Output written to %s" % outfile)

    #plt.show()


def main():

    desc ="AST381 HW#01"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-d','--dist', help='Distribution name',required=True)
    parser.add_argument('-r','--rvs', nargs='*', help='Distribution (rvs) arguments', required=False)
    parser.add_argument('-p','--pdf', nargs='*', help='Distribution (pdf) arguments', required=False)

    args = parser.parse_args()

    try:
        dist = getattr(stats,args.dist)
    except AttributeError:
        print('Not a valid distribution')
        exit(0)
    except: #catchall
        print ("There was a problem: %s" % sys.exc_info()[0])
        exit(-1)

    #sanity check
    try:
        _ = getattr(dist, "pdf")
        _ = getattr(dist, "rvs")
    except AttributeError:
        print('Distribution (%s) has no defined pdf() or rvs() function' % args.dist)
        exit(0)
    except: #catchall
        print ("There was a problem: %s" % sys.exc_info()[0])
        exit(-1)

    plot_pdf(args.dist,args.rvs, args.pdf)


if __name__ == '__main__':
    main()