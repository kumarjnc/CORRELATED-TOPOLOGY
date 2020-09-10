from __future__ import print_function

def ConditionNumber(A):
    '''Work out the condition number of a 2x2 matrix. Uses the singular value approach - another option is to use the norm of A by norm of Ainv, or an eigenvalue approach'''

    from numpy.linalg import svd

    u,s,vh = svd(A);

    return s[0]/s[-1];

cond = ConditionNumber;


def isiterable(x):
    if hasattr(x,'__iter__') or type(x) == str:
        return True;
    else:
        return False;

class NoPrint(object):
    global_disable = False
    #@staticmethod
    def __enter__(self):
        if NoPrint.global_disable:
            return
        import sys
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = sys.stderr = open('/dev/null', 'wb')

        import std_imports
        if 'logger' in dir(std_imports):
            import logging
            self.old_logger_level = std_imports.logger.getEffectiveLevel()
            std_imports.logger.setLevel(logging.ERROR)

    #@staticmethod
    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self,'stdout'):
            import sys
            sys.stdout = self.stdout
            sys.stderr = self.stderr

            import std_imports
            if 'logger' in dir(std_imports):
                std_imports.logger.setLevel(self.old_logger_level)

        return False

class DummyWith(object):
    #@staticmethod
    def __enter__(self):
        pass
    #@staticmethod
    def __exit__(self,exc_type, exc_value, traceback):
        return False


class FileLock(object):
    def __init__(self,filename='.lock',timeout=300):
        self.filename = filename
        self.timeout = timeout

    def __enter__(self):
        import os, time
        count = 0
        self.lock_dir = os.getcwd()

        while count < self.timeout:
            try:
                self.fd = os.open(self.filename,os.O_RDONLY | os.O_CREAT | os.O_EXCL)
            except OSError:
                pass
            else:
                break

            time.sleep(0.1)
            count += 0.1

        if count >= self.timeout:
            raise Exception("Reached timeout in FileLock")

    def __exit__(self,exc_type, exc_value, traceback):
        import os
        bak_dir = os.getcwd()

        os.close(self.fd)

        os.chdir(self.lock_dir)
        os.unlink(self.filename)
        os.chdir(bak_dir)

        return False

def MultiProcessingWrapper(func,*args,**kwds):
    try:
        func(*args,**kwds)
    except KeyboardInterrupt:
        return Exception("KeyboardInterrupt found")
    
def MP_PoolAsyncJoin(pool,results,timeout=10,verbose=False):
    pool.close()

    from multiprocessing import TimeoutError
    temp = list(results)
    exception = True
    while exception:
        try: 
            if len(temp) > 0:
                temp[0].get(timeout)
                if verbose:
                    print("Done another")
                temp = temp[1:]
                continue
            #[z.get(timeout) for z in results]
        except TimeoutError:
            exception = True
        except:
            pool.terminate()
            raise
        else:
            exception = False
            pool.join()

def MP_PoolMapAsync(pool,func,iterable,args=()):
    ret = []
    for item in iterable:
        ret += [pool.apply_async(func,(item,) + args)]

    return ret

def CreateInsert(xdata,ydata,data_lims,plot_box,axes=None,min_size=10,style=''):
    '''This function will create an insert of an already existing graph.
    data_lims is (xmin,xmax, ymin,ymax) and in data points.
    plot_box is (left,bottom, width,height) and is in normalised 0->1 points.
    The min_size argument will make sure the width and height of the box to draw around the
    data of the original plot does not go smaller than this value in pixels.
    The style argument is the plot style argument
    
    Example:
    x = linspace(0,100,10000)
    y = sin(x) + x + 6*exp(-(x-30)**2 * 100)
    plot(x,y)
    CreateInsert(x,y,[29,31,28,36],[0.2,0.6,0.2,0.25])
    '''

    import pylab
    if axes == None:
        axes = pylab.gca()

    store_as = axes.get_autoscale_on()
    axes.set_autoscale_on(False)

    xmin,xmax,ymin,ymax = data_lims
    if xmin > xmax:
        xmin,xmax = xmax,xmin
    if ymin > ymax:
        ymin,ymax = ymax,ymin

    box_ll = axes.transData.transform_point((xmin,ymin))
    box_ur = axes.transData.transform_point((xmax,ymax))
    width_px = box_ur[0] - box_ll[0]
    height_px = box_ur[1] - box_ll[1]
    if width_px < min_size:
        box_ll[0] -= (min_size - width_px) / 2
        box_ur[0] += (min_size - width_px) / 2
    if height_px < min_size:
        box_ll[1] -= (min_size - height_px) / 2
        box_ur[1] += (min_size - height_px) / 2
    box_ll = axes.transData.inverted().transform_point(box_ll)
    box_ur = axes.transData.inverted().transform_point(box_ur)
    
    axes.plot([box_ll[0],box_ur[0],box_ur[0],box_ll[0],box_ll[0]],[box_ll[1],box_ll[1],box_ur[1],box_ur[1],box_ll[1]],'k')

    left,bottom,width,height = plot_box
    ax2 = pylab.axes(plot_box)
    ax2.plot(xdata,ydata,style)
    ax2.set_autoscale_on(False)
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylim(ymin,ymax)
    pylab.xticks([])
    pylab.yticks([])

    # Connect the insert to the box
    (left,bot),(right,top) = ax2.bbox.get_points()

    lowerleft = axes.transAxes.inverted().transform_point([left,bot])
    lowerleft = axes.transLimits.inverted().transform_point(lowerleft)
    upperright = axes.transAxes.inverted().transform_point([right,top])
    upperright = axes.transLimits.inverted().transform_point(upperright)


    # Look at top left corner of box in its various positions.
    if (box_ll[0] < lowerleft[0] and box_ur[1] < upperright[1]) or \
       (box_ll[0] > lowerleft[0] and box_ur[1] > upperright[1]):
        axes.plot([box_ll[0],lowerleft[0]],[box_ur[1],upperright[1]],'k:')
    if (box_ur[0] < upperright[0] and box_ur[1] > upperright[1]) or \
       (box_ur[0] > upperright[0] and box_ur[1] < upperright[1]):
        axes.plot([box_ur[0],upperright[0]],[box_ur[1],upperright[1]],'k:')
    if (box_ll[0] < lowerleft[0] and box_ll[1] > lowerleft[1]) or \
       (box_ll[0] > lowerleft[0] and box_ll[1] < lowerleft[1]): 
        axes.plot([box_ll[0],lowerleft[0]],[box_ll[1],lowerleft[1]],'k:')
    if (box_ur[0] < upperright[0] and box_ll[1] < lowerleft[1]) or \
       (box_ur[0] > upperright[0] and box_ll[1] > lowerleft[1]): 
        axes.plot([box_ur[0],upperright[0]],[box_ll[1],lowerleft[1]],'k:')

    pylab.draw()

    axes.set_autoscale_on(store_as)

    return ax2

def first_index(iterable, cond=None):
    '''
    Returns the index for the first item that satisfies cond.  If cond is not 
    specified, then iterable is treated as an iterable of True and False 
    values.'''

    if cond:
        for index,item in enumerate(iterable):
            if cond(item):
                return index
        return None
    else:
        for index,item in enumerate(iterable):
            if item:
                return index
        return None

def danlu(A):
    '''This is slow implementation. Going straight off the
    theoretical description'''

    from numpy import identity, dot
    from numpy.linalg import inv

    L = identity(A.shape[0],A.dtype)
    U = A.copy()

    i = 0
    n = 0

    for n in range(A.shape[0]):
        Ln = identity(A.shape[0],A.dtype)
        for i in range(n+1,A.shape[0]):
            Ln[i,n] = -U[i,n] / U[n,n]

        U = dot(Ln,U)
        L = dot(Ln,L)
        print('U(n)',U)
        print('L(n)',L)
        print('Linv(n)',inv(L))

    L = inv(L)

    return L,U

def InitDefaultLogger():
    import logging

    formatter = logging.Formatter("%(asctime)s:%(levelname).1s:%(funcName)s:%(message)s",datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger = logging.getLogger("Debugger")
    logger.addHandler(console)
    logger.setLevel(logging.DEBUG)
    
    return logger

def CheckClusterCancel():
    import time,os
    cancel_check_interval = 5
    if hasattr(CheckClusterCancel,'last_time'):
        if time.time() - CheckClusterCancel.last_time < cancel_check_interval:
            return

    CheckClusterCancel.last_time = time.time()

    if 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'BATCH':
        if os.path.exists(os.path.expanduser('~/zzclusterdie')):
            raise KeyboardInterrupt

def avg(x):
    #return sum(x) / len(x)
    # Can't use the above with generators

    if not hasattr(x,'__iter__'):
        return x

    num = 0
    tot = None
    for item in x:
        if tot == None:
            import copy
            tot = copy.copy(item)
        else:
            tot += item
        num += 1

    return tot / num


def OpenUniqueFile(prefix,maxnum=99999,filetype='gzip'):
    import os
    from numpy import log10,floor
    num_digits = int(floor(log10(maxnum) + 1))

    if filetype == 'gzip':
        import gzip
        open = gzip.open
    elif filetype == 'builtin':
        pass
    else:
        raise ValueError("Unknown filetype")

    with FileLock():
        for i in range(maxnum+1):
            filename = '{prefix}{i:0{num_digits}}.pickle.gz'.format(**locals())
            if os.path.exists(filename):
                continue

            file = open(filename,'wb')
            return file
        else:
            raise Exception("Couldn't find an unused valid filename")

disable_mp = False
def CreatePool(given_num=False):
    import multiprocessing as mp
    import os

    if disable_mp:
        return 1,FakePool()

    if 'NSLOTS' in os.environ:
        num = int(os.environ['NSLOTS'])
    elif os.uname()[1] in ['caudoia','pengwyn']:
        num = mp.cpu_count()
    else:
        if given_num:
            num = given_num
        else:
            num = 1

    if given_num and type(given_num) == int:
        num = min(given_num,num)

    return num,mp.Pool(num)

class FakePool(object):
    def apply_async(self,func,*args,**kwds):
        #return FakeResult(apply(func,*args,**kwds))
        return FakeResult(func,args,kwds)
    def close(self):
        pass
    def join(self):
        pass

class FakeResult(object):
    #def __init__(self,res):
    #    self.res = res
    #def get(self):
    #    return self.res
    def __init__(self,func,args,kwds):
        self.data = func,args,kwds
    def get(self):
        func,args,kwds = self.data
        return apply(func,*args,**kwds)

def FuncKwdsToDict(**kwds):
    return kwds

def OutputMatrixToText(A,file,precision=12):
    '''This function only outputs a 2D matrix.'''

    close_file_after = False
    if type(file) == str:
        file = open(file,'wb')
        close_file_after = True

    col_complex = [(A[:,col].imag != 0.).any() for col in xrange(A.shape[1])]

    for row in A:
        for col in xrange(A.shape[1]):
            if col_complex[col]:
                file.write("{0:.{2}g}+{1:.{2}g}j ".format(row[col].real,row[col].imag,precision))
            else:
                file.write("{0:.{1}g} ".format(row[col].real,precision))
        file.write("\n")

    if close_file_after:
        file.close()

def DataFromPGZ(filename):
    import pickle,gzip
    file = gzip.open(filename)
    data = pickle.load(file)
    file.close()
    return data

def TestConvergence(old,new):
    diff = old - new
    avg = (old + new)/2.

    eps = abs(diff)/abs(avg)
    eps_avg = sum(eps) / len(eps)
    eps_max = max(eps)
    eps_min = min(eps)

    ind = eps.argmax()

    return eps_avg,eps_min,eps_max,ind

def irange(start,stop=None,step=1):
    '''Return the same as range, inclusive of the end point.'''
    if stop == None:
        start,stop = 0,start

    if (stop-start)%step == 0:
        return range(start,stop,step) + [stop]
    else:
        return range(start,stop,step)

def iarange(start,stop=None,step=1):
    '''Return the same as arange, inclusive of the end point.
    Because of floating point rounding, this assumes the stepsize 
    coincides with the endpoint.'''
    from numpy import r_
    if stop == None:
        start,stop = 0,start

    return r_[start:stop:step,stop]

def reloadall():
    '''Reload all modules where possible.'''
    import sys

    #blacklist = ['matplotlib','IPython','numpy','__main__','scipy','ctypes','enthought','multiprocessing']
    blacklist = ['std_imports']

    for mod in sys.modules:
        try:
            if sys.modules[mod] is not None and \
                all(z not in mod for z in blacklist) and \
                hasattr(sys.modules[mod],'__file__') and 'work3' in sys.modules[mod].__file__:
                    reload(sys.modules[mod])
                    print("Reloaded module",mod)
        except Exception as exc:
            print("Exception ({2}) in module, {0}: {1}".format(mod,sys.modules[mod],exc))
    

def FitExp(x,y,args=(1.,-1.,0.),show_figures=False):
    from numpy import exp,arange
    from scipy.optimize import leastsq

    func = lambda (a,b,c): a*exp(b*x) + c - y
    fit,dummy,infodict,mesg,ier = leastsq(func,args,full_output=True,maxfev=10000)

    print(avg(infodict['fvec']**2))

    func_fit = func(fit) + y
    err = avg(func(fit))
    if show_figures:
        from pylab import figure,plot
        figure()
        plot(x,y)
        plot(x,func_fit)

    return fit,func_fit


def JudgeValidity():
    '''This function tests the sanity and validity of the results from a TISM 
    run.

    **Returns:**
        ``validity,mesg``

        ``validity`` (boolean)
        ``mesg_list`` (list of strings): Reasons when invalid.
    '''

    from numpy import r_,genfromtxt

    info = ReadInfo()
    num_omega = info['num_omega']

    mesg_list = []

    # If there are no iterations, then this is obviously not valid!
    if info['num_iters'] == None:
        mesg_list += ['No iterations']

    # Read the final iteration data.
    with open('../Nup_Ndown_{0}.dat'.format(info['num_iters'])) as file:
        Nup,Ndown,Sx,Sy,Sz = genfromtxt(file,skip_header=1).T

    avg_Nf = avg(Nup + Ndown)
    avg_Sx = avg(Sx)
    avg_Sy = avg(Sy)
    avg_Sz = avg(Sz)

    # Check the final filling against the desired one.
    if info['force_filling'] == None:
        force_filling = 1.0
    else:
        force_filling = info['force_filling']

    tol = 1e-2
    if avg_Nf - force_filling > tol:
        mesg_list += ['Did not reach desired filling']

    # Check for problems with the imaginary part of the diagonal self-energy
    G,SE = AverageData(-5,-1)
    SE_diag = r_[SE[0][num_omega:].flatten(),SE[1][num_omega:].flatten()]

    if SE_diag[SE_diag.imag > 0].imag.sum() / SE_diag.imag.sum() > 1e-2:
        mesg_list += ['Diagonal self-energy should have imaginary part always negative.']


    with open('validity.txt','wb') as file:
        if len(mesg_list) == 0:
            file.write('Valid!\n')
        else:
            for mesg in mesg_list:
                file.write(mesg + '\n')

    if len(mesg_list) == 0:
        return True,['Valid!']
    else:
        return False,mesg_list

