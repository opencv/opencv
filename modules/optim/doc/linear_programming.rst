Linear Programming
==================

.. highlight:: cpp

optim::solveLP
--------------------
Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).
What we mean here by "linear programming problem" (or LP problem, for short) can be
formulated as:

.. math::
    \mbox{Maximize } c\cdot x\\
    \mbox{Subject to:}\\
    Ax\leq b\\
    x\geq 0

Where :math:`c` is fixed *1*-by-*n* row-vector, :math:`A` is fixed *m*-by-*n* matrix, :math:`b` is fixed *m*-by-*1* column vector and
:math:`x` is an arbitrary *n*-by-*1* column vector, which satisfies the constraints.

Simplex algorithm is one of many algorithms that are designed to handle this sort of problems efficiently. Although it is not optimal in theoretical
sense (there exist algorithms that can solve any problem written as above in polynomial type, while simplex method degenerates to exponential time
for some special cases), it is well-studied, easy to implement and is shown to work well for real-life purposes.

The particular implementation is taken almost verbatim from **Introduction to Algorithms, third edition**
by T. H. Cormen, C. E. Leiserson, R. L. Rivest and Clifford Stein. In particular, the Bland's rule
(`http://en.wikipedia.org/wiki/Bland%27s\_rule <http://en.wikipedia.org/wiki/Bland%27s_rule>`_) is used to prevent cycling.

.. ocv:function:: int optim::solveLP(const Mat& Func, const Mat& Constr, Mat& z)

    :param Func: This row-vector corresponds to :math:`c` in the LP problem formulation (see above). It should contain 32- or 64-bit floating point numbers. As a convenience, column-vector may be also submitted, in the latter case it is understood to correspond to :math:`c^T`.

    :param Constr: *m*-by-*n\+1* matrix, whose rightmost column corresponds to :math:`b` in formulation above and the remaining to :math:`A`. It should containt 32- or 64-bit floating point numbers.

    :param z: The solution will be returned here as a column-vector - it corresponds to :math:`c` in the formulation above. It will contain 64-bit floating point numbers.

    :return: One of the return codes:

::

    //!the return codes for solveLP() function
    enum
    {
        SOLVELP_UNBOUNDED    = -2, //problem is unbounded (target function can achieve arbitrary high values)
        SOLVELP_UNFEASIBLE    = -1, //problem is unfeasible (there are no points that satisfy all the constraints imposed)
        SOLVELP_SINGLE    = 0, //there is only one maximum for target function
        SOLVELP_MULTI    = 1 //there are multiple maxima for target function - the arbitrary one is returned
    };
