"""This file contains code for use with "Think Bayes".

By Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from thinkbayes import Suite


class Dice(Suite):
    """Represents hypotheses about which die was rolled."""

    def likelihood(self, data, hypo):
        """Compute the likelihood of the data under the hypothesis.

        hypo: integer number of sides on the die
        data: integer die roll
        """
        if hypo < data:
            return 0
        else:
            return 1 / hypo


def main():
    """main."""
    suite = Dice([4, 6, 8, 12, 20])

    suite.update(6)
    print('After one 6')
    suite.print()

    for roll in [4, 8, 7, 7, 2]:
        suite.update(roll)

    print('After more rolls')
    suite.print()


if __name__ == '__main__':
    main()
