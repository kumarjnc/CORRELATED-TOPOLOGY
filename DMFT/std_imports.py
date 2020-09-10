from dancommon import isiterable, DummyWith, NoPrint, first_index, FileLock, InitDefaultLogger, CheckClusterCancel, avg, DataFromPGZ, irange, iarange, reloadall
from numpy import argmin, argmax, array, log,prod
from danconstants import *
import itertools
import logging
#from Bunch import Bunch

# Force unbuffered stdout (this could be a problem with batch jobs)
# And it doesn't play nice with ipython.
#import os
#import sys
#sys.stdout = os.fdopen(sys.stdout.fileno(),'w',0)
##sys.stderr = sys.stdout
#sys.stderr = os.fdopen(sys.stderr.fileno(),'w',0)

logger = InitDefaultLogger()

# Setup backend for matplotlib
def useagg():
    import matplotlib
    matplotlib.use('Agg')
import matplotlib
matplotlib.rcParams['legend.loc'] = 'best'
matplotlib.rcParams['legend.fancybox'] = True
matplotlib.rcParams['legend.shadow'] = True
matplotlib.rcParams['image.aspect'] = 'auto'

try:
    import os
    if os.uname()[1] not in ['caudoia','pengwyn','pengix','archguest']:
        useagg()
except AttributeError:
    # Windows doesn't have uname
    pass

# Force complex downcast errors
# But only when numpy supports it.
import numpy
if hasattr(numpy,'ComplexWarning'):
    import warnings
    warnings.simplefilter('error',numpy.ComplexWarning)
    # Also force errors for nans and infs
    warnings.filterwarnings('error','invalid value encountered in divide',RuntimeWarning)
    warnings.filterwarnings('error','invalid value encountered in true_divide',RuntimeWarning)
#    warnings.filterwarnings('error','divide by zero encountered in divide',RuntimeWarning)
#    warnings.filterwarnings('error','divide by zero encountered in true_divide',RuntimeWarning)
    #numpy.seterr(divide='raise',invalid='raise')
    numpy.seterr(divide='raise')

# For ease of inserting debugging statements
try:
    from ipdb import set_trace as pdbbreak
    from ipdb import pm as ipm
except:
    pass

#DefineMethodPickle()
