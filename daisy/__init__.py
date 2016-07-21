from functools import singledispatch, update_wrapper
from inspect import signature
from uuid import uuid4

import dask
from dask.compatibility import apply
from lazy import thunk, strict
from lazy.bytecode import _mk_lazy_function
from lazy.tree import LTree, Call, Normal
from toolz import valmap, curry


def ltree_to_dask(node):
    """Convert an :class:`lazy.tree.LTree` into a dask task graph.

    Parameters
    ----------
    node : LTree
        The node to convert into a tree.

    Returns
    -------
    dask : dict[str, any]
        The equivalent dask task graph.

    Notes
    -----
    This function does common subexpression folding to produce a minimal graph.
    """
    dask = {}
    scope = {}
    return dask, _ltree_to_dask(node, dask, scope)


@singledispatch
def _ltree_to_dask(node, dask, scope):
    raise NotImplementedError(
        'no dispatch for _ltree_to_dask for type %r: %r' % (
            type(node).__name__,
            node,
        ),
    )


@_ltree_to_dask.register(Normal)
def _(node, dask, scope):
    try:
        return scope[node]
    except:
        name = str(uuid4())
        dask[name] = node.value
        return name


@_ltree_to_dask.register(Call)  # noqa
def _(node, dask, scope):
    def retrieve(term):
        try:
            return scope[term]
        except KeyError:
            scope[term] = ret = _ltree_to_dask(term, dask, scope)
            return ret

    name = '%s-%s' % (node.func, uuid4())
    dask[name] = (
        apply,
        retrieve(node.func),
        list(map(retrieve, node.args)),
        (dict, list(map(list, valmap(retrieve, node.kwargs).items()))),
    )
    scope[node] = name
    return name


class inline(strict):
    """A box that denotes that a function should be inlined in autodask.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Notes
    -----

    ``inline`` can allow non-``autodask`` functions to be inlined into the task
    graph. This is nice if you know that a function is a pure computation of
    its inputs and does not need to scrutinize an input to return a final
    computation.

    Functions cannot be inlined into the graph if they are strict on their
    inputs. This means that to return a final defered computation they must
    scrutinize at least one of the inputs and normalize it to a concrete value.

    There are many operations which will force computation, here are some
    common cases:

    *Branching on the input*

    .. code-block:: python

       def f(x):
           if p(x):
               return x + 1
           else:
               return x - 1

    *Iterating over the input with a ``for`` loop*

    .. code-block:: python

      def f(xs):
         total = 0
         for x in xs:
             total += 0
         return total

    *Explicitly strictly evaluating an input*

    .. code-block:: python

       def f(x):
           return strict(x)

    See Also
    --------
    :func:`daisy.autodask`
    """
    def __init__(self, func):
        self._func = func
        self.__signature__ = signature(func)
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class autodaskthunk(thunk):
    """A thunk which is evaluated with dask.

    Parameters
    ----------
    func : callable
        The code for the closure.
    *args, **kwargs
        The free variables.
    """
    _get = dask.get

    def __strict__(self):
        return __class__._get(*ltree_to_dask(LTree.parse(self)))


def register_get(get):
    """Register the ``get`` function which will be used to evaluate
    ``autodaskthunk`` generated dask graphs.

    By default, :func:`dask.get` will be used.

    Parameters
    ----------
    get : callable[dict, str, any]
        The get function.

    Returns
    -------
    get : callable[dict, str, any]
        The ``get`` function unchanged.
    """
    autodaskthunk._get = get
    return get


_autodask = _mk_lazy_function(autodaskthunk, False)
_inline = inline  # alias because ``inline`` is shadowed below


@curry
def autodask(func, *, inline):
    """Mark that a function should lazily build up a call graph to send to be
    executed by dask.

    Parameters
    ----------
    func : callable
        The function to transform.
    inline : bool
        Should the function be inlined into other ``autodask`` functions?
        This should normally be True unless the function is strict on the
        argument.

    Returns
    -------
    transformed : callable
        ``func`` with the transformations needed to build the call graph.

    Notes
    -----

    ``autodask`` transforms a function to build up a call graph which can be
    executed by dask. This is very similar to :func:`dask.delayed` which
    provides an imperitive API to dask.

    **Functional purity**

    ``autodask`` may only be applied to functions which a pure functions of
    their inputs. This means that a function must always be safe to memoize.

    There is no guarantee about the execution order of ``autodask`` defered
    code. Repeated calls to a function with the same arguments may only be
    computed a single time.

    .. note::

       Things to be on the look out for when checking if a function is pure:

       - IO
       - Mutating structures
       - Reading or writing to shared state (please stop this)
       - Randomness

       IO may be okay if you are alright with only executing the call once
       and in an undefined order. You may force the partial order of execution
       by explicitly passing the results of one IO call into the other calls
       that must follow it.

    **Building up our task graph**

    Unlike :func:`dask.delayed`, ``autodask`` is **lazy by default**. This
    means that ``f(a, b)`` will automatically turn into a dask task graph like:

    .. code-block:: python

       {'name': (f, a, b)}

    .. note::

       ``f``, ``a`` and, ``b`` may also be deferred computations themselves.

    Dask will perform best if we can encode more information into the task
    graph before feeding it to dask. To do this, we can pass ``inline=True``
    to autodask before decorating. If a function is inlineable then instead of
    defering the computation, we will enter the code and add the body of that
    function to the dask graph. For example, imagine we have defined ``f``
    like:

    .. code-block:: python

       @autodask(inline=True)
       def f(a, b):
           return a + b + 1

    When calling this function we know that it is safe to replace the task
    ``(f, a, b)`` with the task graph:

    .. code-block:: python

       {'name_1': (add, a, b),
        'result': (add, 'name_1', 1)}

    This will give dask more information to optimize the expression.
    We can also use this to collapse shared work. For example, imagine we have

    .. code-block:: python

       @autodask(inline=True)
       def g(a, b):
           return f(a, b) + f(a, b)

    Because ``f`` is inlineable, we will enter the code and see what it adds to
    the graph. Because we are doing the same work twice, we can reduce it to
    a more simple task graph that will look more like:

    .. code-block:: python

       {'name_1': (add, a, b),
        'f_result': (add, 'name_1', 1),
        'result': (add, 'f_result', 'f_result')}

    This shows that we will not duplicate the work needed to add compute
    ``f(a, b)`` twice.

    **When it is unsafe to pass ``inline=True``**

    There is no default for ``inline`` because it is a very important decision!
    On the one hand, we almost *always* want to pass ``inline=True``; however,
    when we cannot pass that we will silently get much worse performance.

    Functions cannot be inlined into the graph if they are strict on their
    inputs. This means that to return a final defered computation they must
    scrutinize at least one of the inputs and normalize it to a concrete value.

    There are many operations which will force computation, here are some
    common cases:

    *Branching on the input*

    .. code-block:: python

       def f(x):
           if p(x):
               return x + 1
           else:
               return x - 1

    *Iterating over the input with a ``for`` loop*

    .. code-block:: python

      def f(xs):
         total = 0
         for x in xs:
             total += 0
         return total

    *Explicitly strictly evaluating an input*

    .. code-block:: python

       def f(x):
           return strict(x)

    **Differences with :func:`dask.delayed`**

    *Lazy by default vs eager by default*

    While both ``autodask`` and :func:`dask.delayed` serve the same purpose,
    they go about it in different ways. :func:`dask.delayed` is **strict** by
    default. This means that by default, most functions will be entered
    immediatly instead of creating a task. This can be bad if the function
    does not know how to work with the :class:`dask.delayed.Delayed` object or
    is strict on an input. Here is an example of a function in the
    :func:`dask.delayed` API:

    .. code-block:: python

       @dask.delayed
       def f(a, b):
           # lazy call: this will create a node like ``(f, a, b)`` in the
           # resulting task graph
           c = delayed(g)(a, b)

           # strict call: this will enter the code ``h`` immediatly and add the
           # body to the graph. This may not be safe!
           return h(c, b)

    ``autodask`` takes a different approach and is **lazy** by default. This
    means that by default function calls just create a new task for the graph
    and are not executed eagerly. Here is the same function in the ``autodask``
    API:

    .. code-block:: python

       @autodask
       def f(a, b):
           # lazy call: this will create a node like ``(f, a, b)`` in the
           # resulting task graph **unless ``g`` is an inline function**!
           c = g(a, b)

           # strict call: this will enter the code ``h`` immediatly and add the
           # body to the graph. This may not be safe!
           return inline(h)(c, b)

    One advantage of the ``autodask`` approach is that that the potentially
    unsafe operation is called out explicitly, while we choose a more
    conservative graph construction strategy by default. We also allow
    functions to opt-in to inlining if they know it is safe to do so.

    *Magic*

    ``autodask`` uses much darker magic than :func:`dask.delayed`. This is nice
    because it allows us to do things like translate:

    .. code-block:: python

       @autodask
       def f(a, b):
           return a is b

    into a dask graph like:

    .. code-block:: python

       {'result': (operator.is_, a, b)}

    We can also defer things like comprehensions and even literal construction.

    .. warning::

       The magic required for ``autodask`` may be too much for people. It will
       not be easy to debug! :func:`dask.delayed` is a much more reasonable
       solution for most cases. You have been warned.

    See Also
    --------
    :func:`daisy.inline`
    :func:`lazy.strict`
    :func:`dask.delayed`
    """
    func = _autodask(func)
    if inline:
        func = _inline(func)
    return func
