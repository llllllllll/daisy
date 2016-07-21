=========
``daisy``
=========

``dask + lazy = daisy``

A `dask <http://dask.readthedocs.io/en/latest/>`_ backend for lazy_


What is ``daisy``?
------------------

``daisy`` is an experiment to finally use lazy_ for something useful.
``daisy`` is meant to be an alternative to ``dask.delayed`` for automatically
creating computation graphs from functions.


Example
-------

Given the following setup:

.. code-block:: python

   from daisy import autodask, inline, register_get
   from dask import delayed
   from dask.threaded import get
   from lazy import strict
   import numpy as np


   @inline
   def f(a, b):
       return a + b


   def g(a, b):
       return f(f(a, b), f(a, b))

   autodask_g = autodask(g, inline=True)
   delayed_g = delayed(g)


   register_get(get)


   arr = np.arange(1000000)

To start, let's make sure these all do the same thing:

.. code-block:: python

   >>> (g(arr, arr) == delayed_g(arr, arr).compute()).all()
   True

   >>> (g(arr, arr) == autodask_g(arr, arr)).all()
   True

Now we will run some not very scientific profiling runs:

.. code-block:: python

   In [1]: %timeit g(arr, arr)
   100 loops, best of 3: 9.34 ms per loop

   In [2]: %timeit delayed_g(arr, arr).compute()
   100 loops, best of 3: 10.2 ms per loop

   In [3]: %timeit strict(autodask_g(arr, arr))
   100 loops, best of 3: 3.63 ms per loop


Why is this faster?
~~~~~~~~~~~~~~~~~~~

This is a very good case for autodask because we can dramatically reduce the
amount of work we are doing. In the normal function and ``dask.delayed`` cases
we will fall ``f(a, b)`` twice, and then add those together. In the ``autodask``
case will will just directly execute ``a + b`` once, and then add that to
itself. We have totally removed ``f`` from the graph, and instead just use ``+``
directly.

We have used a very large input here to see a speedup. One goal I have is to
reduce the overhead to make this work for smaller inputs and smaller
expressions. I would like to try this with real workloads to see if the amount
of reduced work causes as dramatic of speedups.

More shared work
~~~~~~~~~~~~~~~~

Let's look at a more radical example:

.. code-block:: python

   from daisy import inline, autodask, ltree_to_dask
   from lazy.tree import LTree

   @inline
   def f(a, b):
       return a + b

   @inline
   def g(a, b):
       return a + b + 1

   def h(a, b):
       return f(a, b) + g(a, b)


.. code-block:: python

   In [1]: (h(arr, arr) == autodask_h(arr, arr)).all()
   Out[1]: True

   In [2]: %timeit h(arr, arr)
   100 loops, best of 3: 9.02 ms per loop

   In [3]: %timeit strict(autodask_h(arr, arr))
   100 loops, best of 3: 5.9 ms per loop


The reason this is faster is that we can actually share the work of computing
``a + b`` even though they are in totally separate functions!

.. code-block:: python

   In [4]: from lazy.tree import LTree

   In [5]: from daisy import ltree_to_dask

   In [6]: ltree_to_dask(LTree.parse(autodask_h(arr, arr)))[0]
   Out[6]:
   {'4876ef4b-832a-4058-94f7-29a6fb998ea6': <wrapped-function add>,
    '5a2bee49-2a31-4e01-887f-bfaef7ebb27a': 1,
    'add-39c81b36-ad91-4c2e-93c7-2a74d485fd7b': (<function dask.compatibility.apply>,
     '4876ef4b-832a-4058-94f7-29a6fb998ea6',
     ['add-d581fba1-d73f-42db-8e41-9bff1c803941',
      'add-54f2153f-4cbe-4dfc-babe-cbde4c7d66c1'],
     (dict, [])),
    'add-54f2153f-4cbe-4dfc-babe-cbde4c7d66c1': (<function dask.compatibility.apply>,
     '4876ef4b-832a-4058-94f7-29a6fb998ea6',
     ['add-d581fba1-d73f-42db-8e41-9bff1c803941',
      '5a2bee49-2a31-4e01-887f-bfaef7ebb27a'],
     (dict, [])),
    'add-d581fba1-d73f-42db-8e41-9bff1c803941': (<function dask.compatibility.apply>,
     '4876ef4b-832a-4058-94f7-29a6fb998ea6',
     ['f174fab9-9eb1-4448-991c-5437bd2d709e',
      'f174fab9-9eb1-4448-991c-5437bd2d709e'],
     (dict, [])),
    'f174fab9-9eb1-4448-991c-5437bd2d709e': array([     0,      1,      2, ..., 999997, 999998, 999999])}

The key point here is that we only ever have ``a + b`` once in this graph.


.. _lazy: https://github.com/llllllllll/lazy_python
