"""This file contains code for use with "Think Bayes".

By Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkbayes
import thinkplot

from thinkbayes import Pmf, percentile
from dice import Dice


class Train(Dice):
    """Represents hypotheses about how many trains the company has."""


class Train2(Dice):
    """Represents hypotheses about how many trains the company has."""

    def __init__(self, hypos, alpha=1.0):
        """Initialize the hypotheses with a power law distribution.

        hypos: sequence of hypotheses
        alpha: parameter of the power law prior
        """
        Pmf.__init__(self)
        for hypo in hypos:
            self.set(hypo, hypo**(-alpha))
        self.normalize()


def make_posterior(high, dataset, constructor):
    """Make and updates a Suite.

    high: upper bound on the range of hypotheses
    dataset: observed data to use for the update
    constructor: function that makes a new suite

    Returns: posterior Suite
    """
    hypos = range(1, high + 1)
    suite = constructor(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.update(data)

    return suite


def compare_priors():
    """Run the analysis with two different priors and compares them."""
    dataset = [60]
    high = 1000

    thinkplot.clear_figure()
    thinkplot.pre_plot(num=2)

    constructors = [Train, Train2]
    labels = ['uniform', 'power law']

    for constructor, label in zip(constructors, labels):
        suite = make_posterior(high, dataset, constructor)
        suite.name = label
        thinkplot.pmf(suite)

    thinkplot.save(root='train4',
                   xlabel='Number of trains',
                   ylabel='Probability')


def main():
    """main."""
    compare_priors()

    dataset = [30, 60, 90]

    thinkplot.clear_figure()
    thinkplot.pre_plot(num=3)

    for high in [500, 1000, 2000]:
        suite = make_posterior(high, dataset, Train2)
        print(high, suite.mean())

    thinkplot.save(root='train3',
                   xlabel='Number of trains',
                   ylabel='Probability')

    interval = percentile(suite, 5), percentile(suite, 95)
    print(interval)

    cdf = thinkbayes.make_cdf_from_pmf(suite)
    interval = cdf.percentile(5), cdf.percentile(95)
    print(interval)


if __name__ == '__main__':
    main()
