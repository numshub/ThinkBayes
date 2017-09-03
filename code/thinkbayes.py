"""thinkBayes.

This file contains code for use with "Think Bayes", by Allen B. Downey,
available from greenteapress.com.

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

This file contains class definitions for:

Hist: represents a histogram (map from values to integer frequencies).

Pmf: represents a probability mass function (map from values to probs).

_DictWrapper: private parent class for Hist and Pmf.

Cdf: represents a discrete cumulative distribution function

Pdf: represents a continuous probability density function
"""

import bisect
import copy
import logging
import math
import random

import numpy

from scipy.special import erf, erfinv
import scipy.stats

ROOT2 = math.sqrt(2)


def random_seed(x):
    """Initialize the random and numpy.random generators.

    x: int seed
    """
    random.seed(x)
    numpy.random.seed(x)


def odds(p):
    """Compute odds for a given probability.

    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.

    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define Odds(1)
    to be infinity, so that's what this function does.

    p: float 0-1

    Returns: float odds
    """
    if p == 1:
        return float('inf')
    return p / (1 - p)


def probability(o):
    """Compute the probability corresponding to given odds.

    Example: o=2 means 2:1 odds in favor, or 2/3 probability

    o: float odds, strictly positive

    Returns: float probability
    """
    return o / (o + 1)


def probability2(yes, no):
    """Compute the probability corresponding to given odds.

    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.

    yes, no: int or float odds in favor
    """
    return float(yes) / (yes + no)


class Interpolator():
    """Represent a mapping between sorted sequences; performs linear interp.

    Attributes:
        xs: sorted list
        ys: sorted list
    """

    def __init__(self, xs, ys):
        """Init."""
        self.xs = xs
        self.ys = ys

    def lookup(self, x):
        """Look up x and returns the corresponding value of y."""
        return self._Bisect(x, self.xs, self.ys)

    def reverse(self, y):
        """Look up y and returns the corresponding value of x."""
        return self._Bisect(y, self.ys, self.xs)

    def _bisect(self, x, xs, ys):
        """Helper function."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y


class _DictWrapper(object):
    """An object that contains a dictionary."""

    def __init__(self, values=None, name=''):
        """Initialize the distribution.

        hypos: sequence of hypotheses
        """
        self.name = name
        self.d = {}

        # flag whether the distribution is under a log transform
        self.log = False

        if values is None:
            return

        init_methods = [
            self.init_pmf,
            self.init_mapping,
            self.init_sequence,
            self.init_failure, ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue

        if len(self) > 0:
            self.normalize()

    def init_sequence(self, values):
        """Initialize with a sequence of equally-likely values.

        values: sequence of values
        """
        for value in values:
            self.set(value, 1)

    def init_mapping(self, values):
        """Initialize with a map from value to probability.

        values: map from value to probability
        """
        for value, prob in values.items():
            self.set(value, prob)

    def init_pmf(self, values):
        """Initialize with a Pmf.

        values: Pmf object
        """
        for value, prob in values.items():
            self.set(value, prob)

    def init_failure(self, values):
        """Raise an error."""
        raise ValueError('None of the initialization methods worked.')

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def copy(self, name=None):
        """Return a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        Args:
            name: string name for the new Hist
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.name = name if name is not None else self.name
        return new

    def scale(self, factor):
        """Multiplie the values by a factor.

        factor: what to multiply by

        Returns: new object
        """
        new = self.copy()
        new.d.clear()

        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def log(self, m=None):
        """Log transforms the probabilities.

        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.maxLike()

        for x, p in self.d.items():
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiate the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.maxLike()

        for x, p in self.d.items():
            self.set(x, math.exp(p - m))

    def get_dict(self):
        """Get the dictionary."""
        return self.d

    def set_dict(self, d):
        """Set the dictionary."""
        self.d = d

    def values(self):
        """Get an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return list(self.d.keys())

    def items(self):
        """Get an unsorted sequence of (value, freq/prob) pairs."""
        return list(self.d.items())

    def render(self):
        """Generate a sequence of points suitable for plotting.

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return list(zip(*sorted(self.items())))

    def print(self):
        """Print the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.items()):
            print(val, prob)

    def set(self, x, y=0):
        """Set the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def incr(self, x, term=1):
        """Increment the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        """Scale the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def remove(self, x):
        """Remove a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def total(self):
        """Return the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def max_like(self):
        """Return the largest frequency/probability in the map."""
        return max(self.d.values())


class Hist(_DictWrapper):
    """Represent a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def freq(self, x):
        """Get the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def freqs(self, xs):
        """Get frequencies for a sequence of values."""
        return [self.freq(x) for x in xs]

    def is_subset(self, other):
        """is_subset.

        Check whether the values in this histogram are a subset of the
        values in the given histogram.
        """
        for val, freq in self.items():
            if freq > other.freq(val):
                return False
        return True

    def subtract(self, other):
        """Subtract the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)


class Pmf(_DictWrapper):
    """Represent a probability mass function.

    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def prob(self, x, default=0):
        """Get the probability associated with the value x.

        Args:
            x: number value
            default: value to return if the key is not there

        Returns:
            float probability
        """
        return self.d.get(x, default)

    def probs(self, xs):
        """Get probabilities for a sequence of values."""
        return [self.prob(x) for x in xs]

    def make_cdf(self, name=None):
        """Make a Cdf."""
        return make_cdf_from_pmf(self, name=name)

    def prob_greater(self, x):
        """Probability that a sample from this Pmf exceeds x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val > x]
        return sum(t)

    def prob_less(self, x):
        """Probability that a sample from this Pmf is less than x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val < x]
        return sum(t)

    def __lt__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmf_prob_less(self, obj)
        else:
            return self.prob_less(obj)

    def __gt__(self, obj):
        """Greater than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmf_prob_greater(self, obj)
        else:
            return self.prob_greater(obj)

    def __ge__(self, obj):
        """Greater than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self > obj)

    def __eq__(self, obj):
        """Equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmf_prob_equal(self, obj)
        else:
            return self.prob(obj)

    def __ne__(self, obj):
        """Not equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self == obj)

    def normalize(self, fraction=1.0):
        """Normalize this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total

        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def random(self):
        """Choose a random element from this PMF.

        Returns:
            float value from the Pmf
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')

        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def mean(self):
        """Compute the mean of a PMF.

        Returns:
            float mean
        """
        mu = 0.0
        for x, p in self.d.items():
            mu += p * x
        return mu

    def var(self, mu=None):
        """Compute the variance of a PMF.

        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean

        Returns:
            float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, p in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def maximum_likelihood(self):
        """Return the value with the highest probability.

        Returns: float probability
        """
        prob, val = max((prob, val) for val, prob in self.items())
        return val

    def credible_interval(self, percentage=90):
        """Compute the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.makeCdf()
        return cdf.credibleInterval(percentage)

    def __add__(self, other):
        """Compute the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.addPmf(other)
        except AttributeError:
            return self.addConstant(other)

    def add_pmf(self, other):
        """Compute the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 + v2, p1 * p2)
        return pmf

    def add_constant(self, other):
        """Compute the Pmf of the sum a constant and  values from self.

        other: a number

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            pmf.set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """Compute the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 - v2, p1 * p2)
        return pmf

    def max(self, k):
        """Compute the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.makeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


class Joint(Pmf):
    """Represent a joint distribution.

    The values are sequences (usually tuples)
    """

    def marginal(self, i, name=''):
        """Get the marginal distribution of the indicated variable.

        i: index of the variable we want

        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.items():
            pmf.incr(vs[i], prob)
        return pmf

    def conditional(self, i, j, val, name=''):
        """Get the conditional distribution of the indicated variable.

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have

        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.items():
            if vs[j] != val:
                continue
            pmf.incr(vs[i], prob)

        pmf.normalize()
        return pmf

    def max_like_interval(self, percentage=90):
        """Return the maximum-likelihood credible interval.

        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.

        percentage: float between 0 and 100

        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval


def make_joint(pmf1, pmf2):
    """Joint distribution of values from pmf1 and pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        Joint pmf of value pairs
    """
    joint = Joint()
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            joint.set((v1, v2), p1 * p2)
    return joint


def make_hist_from_list(t, name=''):
    """Make a histogram from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this histogram

    Returns:
        Hist object
    """
    hist = Hist(name=name)
    [hist.incr(x) for x in t]
    return hist


def make_hist_from_dict(d, name=''):
    """Make a histogram from a map from values to frequencies.

    Args:
        d: dictionary that maps values to frequencies
        name: string name for this histogram

    Returns:
        Hist object
    """
    return Hist(d, name)


def make_pmf_from_list(t, name=''):
    """Make a PMF from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this PMF

    Returns:
        Pmf object
    """
    hist = make_hist_from_list(t)
    d = hist.get_dict()
    pmf = Pmf(d, name)
    pmf.normalize()
    return pmf


def make_pmf_from_dict(d, name=''):
    """Make a PMF from a map from values to probabilities.

    Args:
        d: dictionary that maps values to probabilities
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = Pmf(d, name)
    pmf.normalize()
    return pmf


def make_pmf_from_items(t, name=''):
    """Make a PMF from a sequence of value-probability pairs.

    Args:
        t: sequence of value-probability pairs
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = Pmf(dict(t), name)
    pmf.normalize()
    return pmf


def make_pmf_from_hist(hist, name=None):
    """Make a normalized PMF from a Hist object.

    Args:
        hist: Hist object
        name: string name

    Returns:
        Pmf object
    """
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.getDict())
    pmf = Pmf(d, name)
    pmf.normalize()
    return pmf


def make_pmf_from_cdf(cdf, name=None):
    """Make a normalized Pmf from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Pmf

    Returns:
        Pmf object
    """
    if name is None:
        name = cdf.name

    pmf = Pmf(name=name)

    prev = 0.0
    for val, prob in cdf.items():
        pmf.incr(val, prob - prev)
        prev = prob

    return pmf


def make_mixture(metapmf, name='mix'):
    """Make a mixture distribution.

    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      name: string name for the new Pmf.

    Returns: Pmf object.
    """
    mix = Pmf(name=name)
    for pmf, p1 in metapmf.items():
        for x, p2 in pmf.items():
            mix.incr(x, p1 * p2)
    return mix


def make_uniform_pmf(low, high, n):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusize)
    n: number of values
    """
    pmf = Pmf()
    for x in numpy.linspace(low, high, n):
        pmf.set(x, 1)
    pmf.normalize()
    return pmf


class Cdf(object):
    """Represent a cumulative distribution function.

    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        name: string used as a graph label.
    """

    def __init__(self, xs=None, ps=None, name=''):
        """Init."""
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def copy(self, name=None):
        """Return a copy of this Cdf.

        Args:
            name: string name for the new Cdf
        """
        if name is None:
            name = self.name
        return Cdf(list(self.xs), list(self.ps), name)

    def make_pmf(self, name=None):
        """Make a Pmf."""
        return make_pmf_from_cdf(self, name=name)

    def values(self):
        """Return a sorted list of values."""
        return self.xs

    def items(self):
        """Return a sorted sequence of (value, probability) pairs.

        Note: in Python3, returns an iterator.
        """
        return list(zip(self.xs, self.ps))

    def append(self, x, p):
        """Add an (x, p) pair to the end of this CDF.

        Note: this us normally used to build a CDF from scratch, not
        to modify existing CDFs.  It is up to the caller to make sure
        that the result is a legal CDF.
        """
        self.xs.append(x)
        self.ps.append(p)

    def shift(self, term):
        """Add a term to the xs.

        term: how much to add
        """
        new = self.copy()
        new.xs = [x + term for x in self.xs]
        return new

    def scale(self, factor):
        """Multiplie the xs by a factor.

        factor: what to multiply by
        """
        new = self.copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def prob(self, x):
        """Return CDF(x), the probability that corresponds to value x.

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.xs[0]:
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def value(self, p):
        """Return InverseCDF(p), the value that corresponds to probability p.

        Args:
            p: number in the range [0, 1]

        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        if p == 0:
            return self.xs[0]
        if p == 1:
            return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def percentile(self, p):
        """Return the value that corresponds to percentile p.

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.value(p / 100.0)

    def random(self):
        """Choose a random value from this distribution."""
        return self.value(random.random())

    def sample(self, n):
        """Generate a random sample from this distribution.

        Args:
            n: int length of the sample
        """
        return [self.random() for i in range(n)]

    def mean(self):
        """Compute the mean of a CDF.

        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def credible_interval(self, percentage=90):
        """Compute the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.value(prob), self.value(1 - prob)
        return interval

    def _round_(self, multiplier=1000.0):
        """_round_.

        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise UnimplementedMethodException()

    def render(self):
        """Generate a sequence of points suitable for plotting.

        An empirical CDF is a step function; linear interpolation
        can be misleading.

        Returns:
            tuple of (xs, ps)
        """
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def max(self, k):
        """Compute the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


def make_cdf_from_items(items, name=''):
    """Make a cdf from an unsorted sequence of (value, frequency) pairs.

    Args:
        items: unsorted sequence of (value, frequency) pairs
        name: string name for this CDF

    Returns:
        cdf: list of (value, fraction) pairs
    """
    runsum = 0
    xs = []
    cs = []

    for value, count in sorted(items):
        runsum += count
        xs.append(value)
        cs.append(runsum)

    total = float(runsum)
    ps = [c / total for c in cs]

    cdf = Cdf(xs, ps, name)
    return cdf


def make_cdf_from_dict(d, name=''):
    """Make a CDF from a dictionary that maps values to frequencies.

    Args:
       d: dictionary that maps values to frequencies.
       name: string name for the data.

    Returns:
        Cdf object
    """
    return make_cdf_from_items(iter(d.items()), name)


def make_cdf_from_hist(hist, name=''):
    """Make a CDF from a Hist object.

    Args:
       hist: Pmf.hist object
       name: string name for the data.

    Returns:
        Cdf object
    """
    return make_cdf_from_items(hist.items(), name)


def make_cdf_from_pmf(pmf, name=None):
    """Make a CDF from a Pmf object.

    Args:
       pmf: Pmf.pmf object
       name: string name for the data.

    Returns:
        Cdf object
    """
    if name is None:
        name = pmf.name
    return make_cdf_from_items(pmf.items(), name)


def make_cdf_from_list(seq, name=''):
    """Create a CDF from an unsorted sequence.

    Args:
        seq: unsorted sequence of sortable values
        name: string name for the cdf

    Returns:
       Cdf object
    """
    hist = make_hist_from_list(seq)
    return make_cdf_from_hist(hist, name)


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class Suite(Pmf):
    """Represent a suite of hypotheses and their probabilities."""

    def update(self, data):
        """Update each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in self.values():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        return self.normalize()

    def log_update(self, data):
        """Update a suite of hypotheses based on new data.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        Note: unlike Update, LogUpdate does not normalize.

        Args:
            data: any representation of the data
        """
        for hypo in self.values():
            like = self.logLikelihood(data, hypo)
            self.incr(hypo, like)

    def update_set(self, dataset):
        """Update each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.values():
                like = self.likelihood(data, hypo)
                self.mult(hypo, like)
        return self.normalize()

    def log_update_set(self, dataset):
        """Update each hypothesis based on the dataset.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: None
        """
        for data in dataset:
            self.logUpdate(data)

    def likelihood(self, data, hypo):
        """Compute the likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def log_likelihood(self, data, hypo):
        """Compute the log likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def print(self):
        """Print the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print(hypo, prob)

    def make_odds(self):
        """Transform from probabilities to odds.

        Values with prob=0 are removed.
        """
        for hypo, prob in self.items():
            if prob:
                self.set(hypo, odds(prob))
            else:
                self.remove(hypo)

    def make_probs(self):
        """Transform from odds to probabilities."""
        for hypo, odds in self.items():
            self.set(hypo, probability(odds))


def make_suite_from_list(t, name=''):
    """Make a suite from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this suite

    Returns:
        Suite object
    """
    hist = make_hist_from_list(t)
    d = hist.getDict()
    return make_suite_from_dict(d)


def make_suite_from_hist(hist, name=None):
    """Make a normalized suite from a Hist object.

    Args:
        hist: Hist object
        name: string name

    Returns:
        Suite object
    """
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.getDict())
    return make_suite_from_dict(d, name)


def make_suite_from_dict(d, name=''):
    """Make a suite from a map from values to probabilities.

    Args:
        d: dictionary that maps values to probabilities
        name: string name for this suite

    Returns:
        Suite object
    """
    suite = Suite(name=name)
    suite.setDict(d)
    suite.normalize()
    return suite


def make_suite_from_cdf(cdf, name=None):
    """Make a normalized Suite from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Suite

    Returns:
        Suite object
    """
    if name is None:
        name = cdf.name

    suite = Suite(name=name)

    prev = 0.0
    for val, prob in cdf.items():
        suite.incr(val, prob - prev)
        prev = prob

    return suite


class Pdf(object):
    """Represent a probability density function (PDF)."""

    def density(self, x):
        """Evaluate this Pdf at x.

        Returns: float probability density
        """
        raise UnimplementedMethodException()

    def make_pmf(self, xs, name=''):
        """Make a discrete version of this Pdf, evaluated at xs.

        xs: equally-spaced sequence of values

        Returns: new Pmf
        """
        pmf = Pmf(name=name)
        for x in xs:
            pmf.set(x, self.density(x))
        pmf.normalize()
        return pmf


class GaussianPdf(Pdf):
    """Represent the PDF of a Gaussian distribution."""

    def __init__(self, mu, sigma):
        """Construct a Gaussian Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def density(self, x):
        """Evaluate this Pdf at x.

        Returns: float probability density
        """
        return eval_gaussian_pdf(x, self.mu, self.sigma)


class EstimatedPdf(Pdf):
    """Represent a PDF estimated by KDE."""

    def __init__(self, sample):
        """Estimate the density function based on a sample.

        sample: sequence of data
        """
        self.kde = scipy.stats.gaussian_kde(sample)

    def density(self, x):
        """Evaluate this Pdf at x.

        Returns: float probability density
        """
        return self.kde.evaluate(x)

    def make_pmf(self, xs, name=''):
        """Make Pmf."""
        ps = self.kde.evaluate(xs)
        pmf = make_pmf_from_items(list(zip(xs, ps)), name=name)
        return pmf


def percentile(pmf, percentage):
    """Compute a percentile of a given Pmf.

    percentage: float 0-100
    """
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.items():
        total += prob
        if total >= p:
            return val


def credible_interval(pmf, percentage=90):
    """Compute a credible interval for a given distribution.

    If percentage=90, computes the 90% CI.

    Args:
        pmf: Pmf object representing a posterior distribution
        percentage: float between 0 and 100

    Returns:
        sequence of two floats, low and high
    """
    cdf = pmf.make_cdf()
    prob = (1 - percentage / 100.0) / 2
    interval = cdf.value(prob), cdf.value(1 - prob)
    return interval


def pmf_prob_less(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2
    return total


def pmf_prob_greater(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2
    return total


def pmf_prob_equal(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2
    return total


def random_sum(dists):
    """Choose a random value from each dist and returns the sum.

    dists: sequence of Pmf or Cdf objects

    returns: numerical sum
    """
    total = sum(dist.random() for dist in dists)
    return total


def sample_sum(dists, n):
    """Draw a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = make_pmf_from_list(random_sum(dists) for i in range(n))
    return pmf


def eval_gaussian_pdf(x, mu, sigma):
    """Compute the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation

    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)


def make_gaussian_pmf(mu, sigma, num_sigmas, n=201):
    """Make a PMF discrete approx to a Gaussian distribution.

    mu: float mean
    sigma: float standard deviation
    num_sigmas: how many sigmas to extend in each direction
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma

    for x in numpy.linspace(low, high, n):
        p = eval_gaussian_pdf(x, mu, sigma)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def eval_binomial_pmf(k, n, p):
    """Evaluate the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)


def eval_poisson_pmf(k, lam):
    """Compute the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    return scipy.stats.poisson.pmf(k, lam)


def make_poisson_pmf(lam, high, step=1):
    """Make a PMF discrete approx to a Poisson distribution.

    lam: parameter lambda in events per unit time
    high: upper bound of the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for k in range(0, high + 1, step):
        p = eval_poisson_pmf(k, lam)
        pmf.set(k, p)
    pmf.normalize()
    return pmf


def eval_exponential_pdf(x, lam):
    """Compute the exponential PDF.

    x: value
    lam: parameter lambda in events per unit time

    returns: float probability density
    """
    return lam * math.exp(-lam * x)


def eval_exponential_cdf(x, lam):
    """Evaluate CDF of the exponential distribution with parameter lam."""
    return 1 - math.exp(-lam * x)


def make_exponential_pmf(lam, high, n=200):
    """Make a PMF discrete approx to an exponential distribution.

    lam: parameter lambda in events per unit time
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = Pmf()
    for x in numpy.linspace(0, high, n):
        p = eval_exponential_pdf(x, lam)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def standard_gaussian_cdf(x):
    """Evaluate the CDF of the standard Gaussian distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution
    #Cumulative_distribution_function

    Args:
        x: float

    Returns:
        float
    """
    return (erf(x / ROOT2) + 1) / 2


def gaussian_cdf(x, mu=0, sigma=1):
    """Evaluate the CDF of the gaussian distribution.

    Args:
        x: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    return standard_gaussian_cdf(float(x - mu) / sigma)


def gaussian_cdf_inverse(p, mu=0, sigma=1):
    """Evaluate the inverse CDF of the gaussian distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function

    Args:
        p: float

        mu: mean parameter

        sigma: standard deviation parameter

    Returns:
        float
    """
    x = ROOT2 * erfinv(2 * p - 1)
    return mu + x * sigma


class Beta(object):
    """Represent a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """

    def __init__(self, alpha=1, beta=1, name=''):
        """Initialize a Beta distribution."""
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def update(self, data):
        """Update a Beta distribution.

        data: pair of int (heads, tails)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def mean(self):
        """Compute the mean of this distribution."""
        return float(self.alpha) / (self.alpha + self.beta)

    def random(self):
        """Generate a random variate from this distribution."""
        return random.betavariate(self.alpha, self.beta)

    def sample(self, n):
        """Generate a random sample from this distribution.

        n: int sample size
        """
        size = n,
        return numpy.random.beta(self.alpha, self.beta, size)

    def eval_pdf(self, x):
        """Evaluate the PDF at x."""
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def make_pmf(self, steps=101, name=''):
        """Return a Pmf of this distribution.

        Note: Normally, we just evaluate the PDF at a sequence
        of points and treat the probability density as a probability
        mass.

        But if alpha or beta is less than one, we have to be
        more careful because the PDF goes to infinity at x=0
        and x=1.  In that case we evaluate the CDF and compute
        differences.
        """
        if self.alpha < 1 or self.beta < 1:
            cdf = self.makeCdf()
            pmf = cdf.makePmf()
            return pmf

        xs = [i / (steps - 1.0) for i in range(steps)]
        probs = [self.evalPdf(x) for x in xs]
        pmf = make_pmf_from_dict(dict(list(zip(xs, probs))), name)
        return pmf

    def make_cdf(self, steps=101):
        """Return the CDF of this distribution."""
        xs = [i / (steps - 1.0) for i in range(steps)]
        ps = [scipy.special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = Cdf(xs, ps)
        return cdf


class Dirichlet(object):
    """Represent a Dirichlet distribution.

    See http://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    def __init__(self, n, conc=1, name=''):
        """Initialize a Dirichlet distribution.

        n: number of dimensions
        conc: concentration parameter (smaller yields more concentration)
        name: string name
        """
        if n < 2:
            raise ValueError('A Dirichlet distribution with '
                             'n<2 makes no sense')

        self.n = n
        self.params = numpy.ones(n, dtype=numpy.float) * conc
        self.name = name

    def update(self, data):
        """Update a Dirichlet distribution.

        data: sequence of observations, in order corresponding to params
        """
        m = len(data)
        self.params[:m] += data

    def random(self):
        """Generate a random variate from this distribution.

        Returns: normalized vector of fractions
        """
        p = numpy.random.gamma(self.params)
        return p / p.sum()

    def likelihood(self, data):
        """Compute the likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float probability
        """
        m = len(data)
        if self.n < m:
            return 0

        x = data
        p = self.random()
        q = p[:m] ** x
        return q.prod()

    def log_likelihood(self, data):
        """Compute the log likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float log probability
        """
        m = len(data)
        if self.n < m:
            return float('-inf')

        x = self.random()
        y = numpy.log(x[:m]) * data
        return y.sum()

    def marginal_beta(self, i):
        """Compute the marginal distribution of the ith element.

        See http://en.wikipedia.org/wiki/Dirichlet_distribution
        #Marginal_distributions

        i: int

        Returns: Beta object
        """
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def predictive_pmf(self, xs, name=''):
        """Make a predictive distribution.

        xs: values to go into the Pmf

        Returns: Pmf that maps from x to the mean prevalence of x
        """
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return make_pmf_from_items(list(zip(xs, ps)), name=name)


def binomial_coef(n, k):
    """Compute the binomial coefficient "n choose k".

    n: number of trials
    k: number of successes

    Returns: float
    """
    return scipy.misc.comb(n, k)


def log_binomial_coef(n, k):
    """Compute the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)
