"""This file contains code for use with "Think Bayes".

By Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from dice import Dice
import thinkplot


class Train(Dice):
    """
    Container class.

    The likelihood function for the train problem is the same as for the
    Dice problem.
    """


def mean(suite):
    """mean."""
    total = 0
    for hypo, prob in suite.Items():
        total += hypo * prob
    return total


def make_posterior(high, dataset):
    """make_posterior."""
    hypos = range(1, high + 1)
    suite = Train(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.update(data)

    thinkplot.pmf(suite)
    return suite


def main():
    """main."""
    dataset = [30, 60, 90]

    for high in [500, 1000, 2000]:
        suite = make_posterior(high, dataset)
        print(high, suite.mean())

    thinkplot.save(root='train2',
                   xlabel='Number of trains',
                   ylabel='Probability')


if __name__ == '__main__':
    main()
