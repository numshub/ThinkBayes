"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from thinkbayes import Pmf

pmf = Pmf()
pmf.set('Bowl 1', 0.5)
pmf.set('Bowl 2', 0.5)

pmf.mult('Bowl 1', 0.75)
pmf.mult('Bowl 2', 0.5)

pmf.normalize()

print(pmf.prob('Bowl 1'))
