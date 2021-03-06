"""This file contains code for use with "Think Bayes".

By Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

This file contains a partial solution to a problem from
MacKay, "Information Theory, Inference, and Learning Algorithms."

    Exercise 3.15 (page 50): A statistical statement appeared in
    "The Guardian" on Friday January 4, 2002:

        When spun on edge 250 times, a Belgian one-euro coin came
        up heads 140 times and tails 110.  'It looks very suspicious
        to me,' said Barry Blight, a statistics lecturer at the London
        School of Economics.  'If the coin were unbiased, the chance of
        getting a result as extreme as that would be less than 7%.'

MacKay asks, "But do these data give evidence that the coin is biased
rather than fair?"
"""

import thinkbayes
import thinkplot


class Euro(thinkbayes.Suite):
    """Represents hypotheses about the probability of heads."""

    def likelihood(self, data, hypo):
        """Compute the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: string 'H' or 'T'
        """
        x = hypo / 100.0
        if data == 'H':
            return x
        else:
            return 1 - x


class Euro2(thinkbayes.Suite):
    """Represents hypotheses about the probability of heads."""

    def likelihood(self, data, hypo):
        """Compute the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: tuple of (number of heads, number of tails)
        """
        x = hypo / 100.0
        heads, tails = data
        like = x**heads * (1 - x)**tails
        return like


def version1():
    """version1."""
    suite = Euro(range(0, 101))
    heads, tails = 140, 110
    dataset = 'H' * heads + 'T' * tails

    for data in dataset:
        suite.update(data)

    return suite


def version2():
    """version2."""
    suite = Euro(range(0, 101))
    heads, tails = 140, 110
    dataset = 'H' * heads + 'T' * tails

    suite.update_set(dataset)
    return suite


def version3():
    """version3."""
    suite = Euro2(range(0, 101))
    heads, tails = 140, 110

    suite.update((heads, tails))
    return suite


def main():
    """main."""
    suite = version3()
    print(suite.mean())

    thinkplot.pmf(suite)
    thinkplot.show()


if __name__ == '__main__':
    main()
