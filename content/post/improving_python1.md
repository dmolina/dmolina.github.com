+++
title = "Improving performance in Python"
date = 2012-07-15
lastmod = 2017-10-10T15:37:08+02:00
tags = ["python", "performance"]
categories = ["programming"]
draft = false
+++

All the source code of this post is available at [github](https://github.com/dmolina/pyreal).

In the previous post, I recognized my predilection for Python. For me, it is a great language for create prototypes in
many areas. For my research work, I usually creates/designs algorithms for continuous optimization using
[evolutionary algorithms](http://en.wikipedia.org/wiki/Evolutionary_algorithm). For these algorithms, languages like C/C++ or Java are widely used, specially for its
good performance (to publish, it is usual to have to make many comparisons between algorithms, so the performance
could be critical. However, for testing new ideas, many authors uses other tools like Mathlab that reduces the
developer time at the cost of a higher computing time.

I agree that Mathlab is great for numerical algorithms, but I still prefer Python over Mathlab, because I'm more confortable
with it, and have many libraries, and it's more simpler to call code in other languages, written in C or Java. That allow us
to increase the performance, and I like to test how much it could be improved.

Several months ago, I start writing my most succesful algorithm, [Memetic Algorithms based on LS Chaining](http://sci2s.ugr.es/EAMHCO/#macmals), in Python. I had several
doubts about the performance, so I start writing one element, an Steady-State Genetic Algorithm, in Python.


## Calling C/C++ code from python {#calling-c-c-code-from-python}

The first challenge I had to tackle was to allow my python program to use the same benchmark functions than other implementations,
[CEC'2005 benchmark](http://sci2s.ugr.es/EAMHCO/#TestF).
This benchmark define the functions to optimize, thus its main funtionality is
evaluate my solutions, when each solution is a vector of real numbers, with a real fitness value.
The benchmark code was implemented (by its authors) in C/C++. So, my python code have to call C++ code.

For doing that, I used the library [boost::python](http://www.boost.org/doc/libs/1_50_0/libs/python/doc/index.html), that is, in my opinion, the simpler way to call C/C++ code, specially
when we uses [numpy](http://numpy.scipy.org/) package.

In my case, it is very simple, because I need few functions:

```python
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/list.hpp>
#include <iostream>
#include "cec2005/cec2005.h"
#include "cec2005/srandom.h"

using namespace boost::python;

Random r(new SRandom(12345679));

void set_function(int fun, int dim) {
    init_cec2005(&r, fun, dim);
}

double evalua(const numeric::array &el) {
   const tuple &shape = extract<tuple>(el.attr("shape"));
   unsigned n = boost::python::extract<unsigned>(shape[0]);
   double *tmp = new double[n];
  for(unsigned int i = 0; i < n; i++)
    {
      tmp[i] = boost::python::extract<double>(el[i]);
    }
  double result = eval_cec2005(tmp, n);
  delete tmp;
  return result;
}
...

BOOST_PYTHON_MODULE(libpycec2005)
{
    using namespace boost::python;
    numeric::array::set_module_and_type( "numpy", "ndarray");
    def("config", &set_function);
    def("evaluate", &evalua);
    ...
}
```

More info in the good [boost::python](http://www.boost.org/doc/libs/1_50_0/libs/python/doc/index.html) documentation.

One we can call C/C++ code, we have implemented the algorithm in python code.
The test code was the following:

```python
from ssga import SSGA
from readargs import ArgsCEC05
import libpycec2005 as cec2005
import numpy

def check_dimension(option, opt, value):
    if value not in [2, 10, 30, 50]:
        raise OptionValueError(
            "option %s: invalid dimensionality value: %r" % (opt, value))

def main():
    """
    Main program
    """
    args = ArgsCEC05()

    if  args.hasError:
        args.print_help_exit()

    fun = args.function
    dim = args.dimension

    print "Function: %d" %fun
    print "Dimension: %d" %dim
    cec2005.config(fun, dim)
    domain = cec2005.domain(fun)
    print "Domain: ", domain
    ea = SSGA(domain=domain, size=60, dim=dim, fitness=cec2005.evaluate)

    for x in xrange(25):
        ea.run(maxeval=dim*10000)
        [bestsol, bestfit] = ea.getBest()
        print "BestSol: ", bestsol
        print "BestFitness: %e" %bestfit
        ea.reset()

if __name__ == "__main__":
    main()
```

This source code run the algorithm 25 times, and in each run the algorithm stops when they are created 10000\*dim solutions.
These conditions are indicated in the [benchmark specification](http://sci2s.ugr.es/EAMHCO/Tech-Report-May-30-05.pdf). The only parameter was the function (-f, used function 1 by
default) and dimension (-d) from 10, 30, 50.


## Profiling the computing time {#profiling-the-computing-time}

How much time it takes? I have changed xrange(25) for xrange(1) and we have run its current version.
The final time was 7 minutes for dimension 10, and 21 minutes for dimension 30 (for only one function).

Because I like to make more interesting things,  that only waiting computing time, I use the profile, only
one run for the function, to detect the functions/method more expensive in computing time.

```bash
python -m cProfile runcec.py -f 1 -d 10
```

The output was the following:

```bash
        2943600 function calls (2943531 primitive calls) in 31.031 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
....
      1    0.001    0.001    0.126    0.126 ssga.py:1(<module>)
    99940    0.561    0.000   17.463    0.000 ssga.py:109(cross)
        1    0.000    0.000    0.000    0.000 ssga.py:123(reset)
        1    5.559    5.559   51.129   51.129 ssga.py:126(run)
        1    0.000    0.000    0.000    0.000 ssga.py:14(__init__)
        1    0.000    0.000    0.000    0.000 ssga.py:158(getBest)
        1    0.000    0.000    0.000    0.000 ssga.py:31(set_mutation_rate)
    99940    0.730    0.000    1.885    0.000 ssga.py:45(mutation)
    12438    0.286    0.000    0.758    0.000 ssga.py:50(mutationBGA)
        1    0.002    0.002    0.002    0.002 ssga.py:77(initPopulation)
   105883    1.101    0.000    5.604    0.000 ssga.py:89(updateWorst)
        1    0.000    0.000    0.000    0.000 ssga.py:9(SSGA)
    99940    1.049    0.000   20.617    0.000 ssga.py:97(getParents)
...

```

With the profile we can observe the most expensive methods in our code:
getParents (20 seconds), crossover operator (17 seconds), and updateWorst (5 seconds).
These methods are the 85% of the computing time, and the first two methods the 74%
of the computing time.


## Optimising the code {#optimising-the-code}

That proves the majority of computing time is due to a minority of the code,
only three methods. If we can optimize these methods, our code could be
improved a lot.

We can uses again the [boost::python](http://www.boost.org/doc/libs/1_50_0/libs/python/doc/index.html) package, but it's a bit tedious to use it. So, we have
used the [cython](http://www.cython.org/) package. With cython we can optimize the source code adding
information about the types.

For instead, Instead of the following code:

```python
import numpy as np

def distance(ind1,ind2):
    """
    Euclidean distance
    ind1 -- first array to compare
    ind2 -- second array to compare

    Return euclidean distance between the individuals

    >>> from numpy.random import rand
    >>> import numpy as np
    >>> dim = 30
    >>> sol = rand(dim)
    >>> distance(sol,sol)
    0.0
    >>> ref=np.zeros(dim)
    >>> dist=distance(sol,ref)
    >>> dist > 0
    True
    >>> dist2 = distance(sol*2,ref)
    >>> 2*dist == dist2
    True
    """
    dif = ind1-ind2
    sum = (dif*dif).sum()
    return math.sqrt(sum)
```

we can write:

```python
cimport numpy as np
cimport cython
DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef np.int_t BTYPE_t

def distance(np.ndarray[DTYPE_t, ndim=1]ind1, np.ndarray[DTYPE_t, ndim=1] ind2):
    """
    Euclidean distance
    ind1 -- first array to compare
    ind2 -- second array to compare

    ....
    """
    cdef np.ndarray[DTYPE_t, ndim=1] dif = ind1-ind2
    cdef double sum = (dif*dif).sum()
    return math.sqrt(sum)
```

We can see that is still very readable. we only have put information about the type
and dimension in the vector parameters and about the variables, using the keyword
cdef.

Let's see as an example the first method, the crossover operator, implemented
in the crossBLX method:

```python
import numpy as np
import math

def crossBLX(mother,parent,domain,alpha):
    """
    crossover operator BLX-alpha

    mother -- mother (first individual)
    parent -- parent (second individual)
    domain -- domain to check
    alpha  -- parameter alpha

    Returns the new children following the expression children = random(x-alpha*dif, y+alpha*dif),
                where dif=abs(x,y) and x=lower(mother,parents), y=upper(mother,parents)

    >>> import numpy as np
    >>> low=-5
    >>> upper = 5
    >>> dim=30
    >>> sol = np.array([1,2,3,2,1])
    >>> crossBLX(sol,sol,[low,upper],0)
    array([ 1.,  2.,  3.,  2.,  1.])
    """
    diff = abs(mother-parent)
    dim = mother.size
    I=diff*alpha
    points = np.array([mother,parent])
    A=np.amin(points,axis=0)-I
    B=np.amax(points,axis=0)+I
    children = np.random.uniform(A,B,dim)
    [low,high]=domain
    return np.clip(children, low, high)

```

We can see that it is very simple to implement using numpy, but it is still very slow. With cython I have
defined directly implement the many operations, the following code:

```python
def crossBLX(np.ndarray[DTYPE_t, ndim=1] mother,np.ndarray[DTYPE_t, ndim=1] parent,list domain, double alpha):
    """
    ...
    """
    cdef np.ndarray[DTYPE_t, ndim=1] C, r
    cdef int low, high, dim
    cdef double x, y
    cdef double I, A, B
    cdef unsigned i
    [low,high]=domain
    dim = mother.shape[0]
    C = np.zeros(dim)
    r = random.randreal(0,1,dim)

    for i in range(dim):
        if mother[i] < parent[i]:
           (x,y) = (mother[i],parent[i])
        else:
           (y,x) = (mother[i],parent[i])

        I = alpha*(y-x)
        A=x-I
        B=y+I

        if (A < low):
            A = low
        if (B > high):
            B = high

        C[i] = A+r[i]*(B-A)

    return C

```

It's true that the source code is more complicated, but it is still very readable.
I have compared the two implementation to make sure both return the same values.


## Measuring the improvement {#measuring-the-improvement}

How much these small changes in the code?
I have profile the source code again and it gives me:

```bash
         1020045 function calls (1019976 primitive calls) in 18.003 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
....
        1    0.001    0.001    0.127    0.127 ssga.py:1(<module>)
    99940    0.425    0.000    2.432    0.000 ssga.py:109(cross)
        1    0.000    0.000    0.000    0.000 ssga.py:123(reset)
        1    5.415    5.415   17.864   17.864 ssga.py:126(run)
        1    0.000    0.000    0.000    0.000 ssga.py:14(__init__)
        1    0.000    0.000    0.000    0.000 ssga.py:158(getBest)
        1    0.000    0.000    0.000    0.000 ssga.py:31(set_mutation_rate)
    99940    0.699    0.000    2.006    0.000 ssga.py:45(mutation)
    12544    0.338    0.000    0.929    0.000 ssga.py:50(mutationBGA)
        1    0.002    0.002    0.002    0.002 ssga.py:77(initPopulation)
   105959    0.775    0.000    1.343    0.000 ssga.py:89(updateWorst)
        1    0.000    0.000    0.000    0.000 ssga.py:9(SSGA)
    99940    0.940    0.000    6.665    0.000 ssga.py:97(getParents)
....

```

We can see the improvement obtained.

| Method           | Python | Cython |
|------------------|--------|--------|
| cross          : | 17.4   | 2.4    |
| getParents     : | 20.6   | 6.6    |
| updateWorst    : | 5.6    | 1.3    |
| Total            | 43.6   | 10.3   |

The new code takes only a 23% of the computing time of the previous code.
With these changes, we have reduced the total time from 51 seconds to 18 code.


## In perspective {#in-perspective}

Now, I run the source code without the profile, and test the source code obtaining the
following time:

| Method      | dim=10 | dim=30      | dim=50 |
|-------------|--------|-------------|--------|
| Python      | 44s    | 3240s (54m) | --     |
| Cython      | 10s    | 28s         | 48s    |
| Improvement | 77%    | 99%         | ---    |

In the following table, we test the maximum time for one and 25 runs (the time depends on the
function used).

| #functions | dim=10  | dim=30  | dim=50 |
|------------|---------|---------|--------|
| 1          | 10s/18s | 28s/40s | 48s/1m |
| 25         | 3/7m    | 15/21m  | 38m/   |

So, the total computing time is 7 minutes for dimension 10, and 21 minutes for dimension 30.
These numbers are very acceptable, specially because we can test in parallel the different functions
in a cluster of computers. Unfortunately, an implementation in Mathlab not only take more time, but
also, for licensing reasons, it could not run in parallel without limit.

In resume, we can uses python code, not only to create experimental prototypes, but also to create
functional prototypes.

And about the possible testing problem, I've been working on it, but I think it is enough for a post,
didn't it? :-)

All the code refered in the post, both in python and cython, is available at [github](https://github.com/dmolina/pyreal), if you want to see it.
